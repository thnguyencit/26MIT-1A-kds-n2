"""

trainer.py — CHUẨN BỊ, HUẤN LUYỆN, ĐÁNH GIÁ, LƯU / NẠP MÔ HÌNH  (bước 4 → 7 của workflow)

MỤC LỤC
    Phần A. Siêu tham số ....... HParams
    Phần B. Đánh giá ........... tune_decision, auc, metrics
    Phần C. Chuẩn bị dữ liệu ... Prepared, to_tensors
    Phần D. Huấn luyện ......... train_model, predict
    Phần E. Baseline ........... body_lr
    Phần F. Lưu / nạp .......... save_bundle, Bundle

THẺ TÌM NHANH  (Ctrl+F)
    [RÒ RỈ]      chỗ chống rò rỉ dữ liệu
    [COLD-START] chỗ xử lý khách mới
    [ID DROPOUT] kỹ thuật giả làm khách mới lúc train
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data import BodyEncoder, FitData, Indexer, SizeRegressor
from .model import ModelSpec, OrdinalHybridMF

# Phiên bản định dạng file .pt (để báo lỗi nếu nạp nhầm file)
FORMAT_VERSION = 1


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. SIÊU THAM SỐ
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class HParams:
    """Siêu tham số huấn luyện (giá trị mặc định = giá trị trong báo cáo).

    dim      = 16     số chiều vector ẩn
    hidden   = 64     số nơ-ron lớp ẩn của mạng E(x)
    lr       = 0.003  learning rate của Adam
    l2       = 0.01   hệ số phạt L2
    batch    = 1024   số giao dịch mỗi batch
    epochs   = 30     số epoch tối đa
    patience = 3      dừng sớm nếu AUC val không tăng 3 epoch liền
    """
    dim: int = 16
    hidden: int = 64
    lr: float = 3e-3
    l2: float = 1e-2
    batch: int = 1024
    epochs: int = 30
    patience: int = 3

    def to_dict(self):
        """Đổi sang dict để lưu vào file .pt và config.json."""
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════════════════════

# Các trọng số thử cho lớp chật và rộng (8 x 8 = 64 tổ hợp)
DECISION_GRID = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6]


def tune_decision(p_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Chọn trọng số w = (w_chật, 1, w_rộng) trên VAL để macro-F1 cao nhất."""
    best, bw = -1.0, np.ones(3)
    for a in DECISION_GRID:
        for b in DECISION_GRID:
            w = np.array([a, 1.0, b])
            # dự đoán = argmax(p · w), chấm bằng macro-F1
            f = f1_score(y_val, (p_val * w).argmax(1), average="macro", zero_division=0)
            if f > best:
                best, bw = f, w
    return bw


def auc(y, p) -> float:
    """AUC macro one-vs-rest cho 3 lớp: chỉ số CHÍNH của báo cáo."""
    try:
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        return float("nan")


def metrics(y, p, w=np.ones(3)) -> dict:
    """Tính 3 chỉ số: {"auc", "macro_f1", "acc"}."""
    yhat = (p * w).argmax(1)
    return {"auc": auc(y, p), "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
            "acc": float(accuracy_score(y, yhat))}


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN C. CHUẨN BỊ DỮ LIỆU  (bước 4 + 5)
# ════════════════════════════════════════════════════════════════════════════

class Prepared:
    """BƯỚC 4 + 5. Một lần chia train/val/test đã mã hoá sẵn, dùng CHUNG cho mọi mô hình."""

    def __init__(self, data: FitData, tr, va, te=None):
        self.data = data
        self.idx = {"train": tr, "val": va}
        if te is not None:
            self.idx["test"] = te
        train = data.df.iloc[tr]

        # (1) [RÒ RỈ] HỌC tham số tiền xử lý — CHỈ trên train
        self.indexer = Indexer().fit(train)
        self.body_enc = BodyEncoder().fit(data.body.iloc[tr], train["body_type"])

        # (2) ÁP DỤNG cho mọi tập bằng tham số vừa học
        self.frames = {k: data.df.iloc[ix] for k, ix in self.idx.items()}
        self.X = {k: self.body_enc.transform(data.body.iloc[ix], data.df["body_type"].iloc[ix])
                  for k, ix in self.idx.items()}

        # (3) [RÒ RỈ] Ridge ước lượng size cũng chỉ học trên X["train"]
        self.size_reg = SizeRegressor().fit(self.X["train"], train["category"], train["size"])

        # (4) Đóng gói thành tensor cho từng tập: chỉ số id + size_gap + vector cơ thể + nhãn
        self.tensors = {}
        for k in self.idx:
            f = self.frames[k]
            ix = self.indexer.transform(f)
            ix["gap"] = self.size_reg.gap(self.X[k], f["category"], f["size"])
            self.tensors[k] = to_tensors(ix, self.X[k], f["y"].values)

    def y(self, part):
        """Nhãn thật (0/1/2) của tập "train" / "val" / "test"."""
        return self.frames[part]["y"].values


def to_tensors(ix: dict, X: np.ndarray, y: np.ndarray | None = None) -> dict:
    """Đổi dict mảng numpy thành dict tensor PyTorch; thêm "x" (vector cơ thể) và "y" (nhãn, nếu có)."""
    b = {k: torch.as_tensor(np.array(v)) for k, v in ix.items()}
    b["x"] = torch.as_tensor(np.array(X))
    if y is not None:
        b["y"] = torch.as_tensor(np.array(y), dtype=torch.long)
    return b


def _slice(b: dict, idx) -> dict:
    """Cắt 1 batch: lấy cùng các dòng idx ở mọi tensor trong dict."""
    return {k: v[idx] for k, v in b.items()}


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN D. HUẤN LUYỆN  (bước 6)
# ════════════════════════════════════════════════════════════════════════════

def train_model(prep: Prepared, spec: ModelSpec, hp: HParams, seed: int, on_epoch=None) -> OrdinalHybridMF:
    """BƯỚC 6. Huấn luyện 1 cấu hình mô hình."""
    # (1) Khởi tạo mô hình + bộ tối ưu (cố định seed để chạy lại ra đúng kết quả)
    torch.manual_seed(seed)
    model = OrdinalHybridMF(prep.indexer.sizes(), prep.body_enc.dim, spec, dim=hp.dim, hidden=hp.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    tr, va = prep.tensors["train"], prep.tensors["val"]
    y_va = prep.y("val")
    n = len(tr["y"])
    g = torch.Generator().manual_seed(seed)

    # [ID DROPOUT] chỉ có ý nghĩa khi mô hình có CẢ lịch sử lẫn số đo
    drop = spec.id_drop if (spec.use_id and spec.use_body) else 0.0
    best, best_state, bad, history = -1.0, None, 0, []

    for ep in range(1, hp.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0

        # (2) Duyệt từng batch
        for s in range(0, n, hp.batch):
            b = _slice(tr, perm[s:s + hp.batch])

            # [ID DROPOUT] [COLD-START] ngẫu nhiên ~10% dòng bị đặt user_id = 0 (giả làm khách mới)
            if drop > 0:
                b["user_id"] = b["user_id"] * (torch.rand(len(b["y"]), generator=g) >= drop).long()

            # Lan truyền xuôi -> loss -> lan truyền ngược -> cập nhật
            p = model(b)
            # Loss NLL = -log P(lớp đúng)
            loss = F.nll_loss(torch.log(p), b["y"])
            opt.zero_grad()
            # cộng phạt L2 rồi lan truyền ngược, Adam cập nhật trọng số
            (loss + hp.l2 * model.l2_terms(b)).backward()
            opt.step()
            tot += loss.item() * len(b["y"])

        # (3) Đánh giá trên VAL sau mỗi epoch
        val_auc = auc(y_va, predict(model, va))
        history.append({"epoch": ep, "train_nll": tot / n, "val_auc": val_auc, "sec": time.time() - t0})
        if on_epoch:
            on_epoch(history[-1])

        # (4) Early stopping: cải thiện ít nhất 0.0001 thì lưu lại, ngược lại đếm số lần "bad"
        if val_auc > best + 1e-4:
            best, bad, best_state = val_auc, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= hp.patience:
                break

    # (5) Nạp lại trọng số của epoch tốt nhất
    model.load_state_dict(best_state)
    model.best_val_auc, model.history = best, history
    return model


@torch.no_grad()
def predict(model: OrdinalHybridMF, b: dict, batch: int = 65536) -> np.ndarray:
    """BƯỚC 7. Dự đoán [P(chật), P(vừa), P(rộng)] cho mọi dòng."""
    model.eval()
    n = len(b["x"])
    return np.concatenate([model(_slice(b, slice(s, s + batch))).numpy() for s in range(0, n, batch)])


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN E. BASELINE
# ════════════════════════════════════════════════════════════════════════════

def body_lr(prep: Prepared, seed: int):
    """Baseline Logistic Regression: số đo + size + danh mục + dịp (one-hot)."""
    tr = prep.frames["train"]
    cats = {c: sorted(tr[c].unique()) for c in ["category", "context"]}

    def feats(part):
        """Đặc trưng = [ vector cơ thể | size chuẩn hoá | one-hot danh mục, dịp ]."""
        d = prep.frames[part]
        oh = [np.stack([(d[c] == v).values for v in vals], 1) for c, vals in cats.items() if len(vals) > 1]
        sz = ((d["size"] - prep.indexer.size_mu) / prep.indexer.size_sd).values[:, None]
        return np.hstack([prep.X[part], sz] + [o.astype(float) for o in oh])

    # class_weight="balanced": tăng trọng số lớp hiếm (chật, rộng)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(feats("train"), prep.y("train"))
    return {part: clf.predict_proba(feats(part)) for part in prep.idx}


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN F. LƯU / NẠP MÔ HÌNH
# ════════════════════════════════════════════════════════════════════════════

def save_bundle(path, model, prep: Prepared, spec: ModelSpec, hp: HParams, extra: dict):
    """Lưu vào 1 file .pt mọi thứ cần để gợi ý size về sau (lệnh `sizerec train`)."""
    from .sizing import SizeModel
    train = prep.frames["train"]
    item_sizes = prep.data.df.groupby("item_id")["size"].agg(lambda s: sorted(set(float(x) for x in s))).to_dict()
    sm = SizeModel().fit(train, prep.X["train"], prep.size_reg)
    item_cat = train.groupby("item_id")["category"].agg(lambda s: s.mode().iloc[0]).to_dict()
    torch.save({
        "format": FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "spec": spec.to_dict(),
        "hparams": hp.to_dict(),
        "indexer": prep.indexer.state(),
        "body_encoder": prep.body_enc.state(),
        "size_regressor": prep.size_reg.state(),
        "size_model": sm.state(),
        "item_sizes": {str(k): v for k, v in item_sizes.items()},
        "item_category": {str(k): str(v) for k, v in item_cat.items()},
        "user_counts": {str(k): int(v) for k, v in train["user_id"].value_counts().items()},
        "extra": extra,
    }, path)


class Bundle:
    """Nạp file .pt và dựng lại mô hình + bộ mã hoá để gợi ý cho khách (lệnh `sizerec recommend`)."""

    def __init__(self, path):
        """Đọc file .pt và dựng lại mọi thành phần (không cần dữ liệu train)."""
        # (1) Đọc file .pt (weights_only=True: chỉ nạp dữ liệu, an toàn)
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d.get("format") != FORMAT_VERSION:
            raise ValueError("File mô hình không đúng định dạng của sizerec.")
        self.raw = d

        # (2) Dựng lại từng thành phần từ tham số đã lưu (không cần dữ liệu train)
        self.spec = ModelSpec(**d["spec"])
        self.hp = HParams(**d["hparams"])
        self.indexer = Indexer.from_state(d["indexer"])
        self.body_enc = BodyEncoder.from_state(d["body_encoder"])
        self.size_reg = SizeRegressor.from_state(d["size_regressor"])
        from .sizing import SizeModel
        self.size_model = SizeModel.from_state(d["size_model"])

        # (3) Tạo mạng đúng kích thước rồi nạp trọng số
        self.model = OrdinalHybridMF(self.indexer.sizes(), self.body_enc.dim, self.spec,
                                     dim=self.hp.dim, hidden=self.hp.hidden)
        self.model.load_state_dict(d["state_dict"])
        self.model.eval()

        self.item_sizes = d["item_sizes"]
        self.item_category = d["item_category"]
        self.user_counts = d["user_counts"]
        self.extra = d["extra"]

    def batch(self, rows: pd.DataFrame, body: pd.DataFrame) -> dict:
        """Mã hoá các dòng mới (khách + sản phẩm + size) bằng ĐÚNG tham số đã lưu lúc train."""
        X = self.body_enc.transform(body, rows["body_type"])
        ix = self.indexer.transform(rows)
        ix["gap"] = self.size_reg.gap(X, rows["category"], rows["size"])
        return to_tensors(ix, X)
