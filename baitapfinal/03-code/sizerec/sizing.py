"""

sizing.py — CHỌN SIZE CỤ THỂ ĐỂ ĐỀ XUẤT  (bước 8 của workflow)

MỤC LỤC
    Phần A. Mô hình chọn size .. SizeModel (fit, base_body, target)
    Phần B. Đánh giá ........... nearest, evaluate_size_rec

THẺ TÌM NHANH  (Ctrl+F)
    [COLD-START] chỗ xử lý khách mới
    [HYBRID]     chỗ trộn lịch sử với số đo cơ thể

CÔNG THỨC
    base_body(x)   : size nền ước lượng từ số đo cơ thể (hệ số Ridge của data.SizeRegressor)
    user_base(u)   : size nền của khách học từ lịch sử mua
    cat_coef(c)    : chênh lệch theo danh mục
    item_offset(i) : sản phẩm "chạy" to / nhỏ bao nhiêu

    base   = w · user_base(u) + (1 - w) · base_body(x),    w = n_u / (n_u + 2)
    target = base + cat_coef(c) + item_offset(i)
    size đề xuất = size có sẵn của sản phẩm gần target nhất
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SizeRegressor


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. MÔ HÌNH CHỌN SIZE
# ════════════════════════════════════════════════════════════════════════════

class SizeModel:
    """BƯỚC 8. Mô hình chọn size (tuyến tính, dễ giải thích). Mọi tham số học từ TẬP TRAIN."""

    def fit(self, train: pd.DataFrame, X: np.ndarray, reg: SizeRegressor, k_item: float = 5.0,
            k_user: float = 2.0) -> "SizeModel":
        """Học 4 thành phần của công thức chọn size."""
        self.k_user = k_user

        # (1) Tách hệ số Ridge: [ hệ số chặn | trọng số số đo | hệ số danh mục ]
        coef = np.array(reg.coef)
        n_body = X.shape[1]
        self.intercept, self.w_body = float(coef[0]), coef[1:1 + n_body].tolist()
        self.cat_coef = {c: float(v) for c, v in zip(reg.cats, coef[1 + n_body:])}
        # bước hiệu chỉnh khi khách từng bị chật / rộng = nửa độ lệch chuẩn size
        self.step = float(reg.sd) * 0.5

        # (2) Phần dư = size thực mua - size dự đoán từ cơ thể và danh mục
        base = self.base_body(X)
        cc = train["category"].map(self.cat_coef).fillna(0.0).values
        resid = train["size"].values - base - cc

        # (3) item_offset: chỉ dùng giao dịch "vừa" để biết sản phẩm chạy to hay nhỏ.
        fit = train["y"].values == 1
        d = pd.DataFrame({"item": train["item_id"].values[fit], "r": resid[fit]})
        g = d.groupby("item")["r"].agg(["sum", "count"])
        self.item_offset = ((g["sum"]) / (g["count"] + k_item)).astype(float).to_dict()

        # (4) user_base: size "đúng" của khách trong mỗi lần mua, rồi lấy trung bình.
        adj = np.select([train["y"].values == 0, train["y"].values == 2], [self.step, -self.step], 0.0)
        io = train["item_id"].map(self.item_offset).fillna(0.0).values
        u = pd.DataFrame({"u": train["user_id"].values, "b": train["size"].values - cc - io + adj})
        gu = u.groupby("u")["b"].agg(["mean", "count"])
        self.user_base = gu["mean"].astype(float).to_dict()
        self.user_n = gu["count"].astype(int).to_dict()

        # (5) Dự phòng khi khách không nhập số đo nào: trung vị chung
        self.global_base = float(np.median(train["size"].values - cc - io))
        return self

    def base_body(self, X: np.ndarray) -> np.ndarray:
        """Size nền chỉ từ số đo cơ thể = hệ số chặn + w_body · x."""
        return self.intercept + X @ np.array(self.w_body)

    def target(self, item_id: str, category: str, X_row: np.ndarray | None, user_id: str | None) -> dict:
        """Tính size mục tiêu cho 1 khách + 1 sản phẩm."""
        # (1) Size nền theo cơ thể (không nhập số đo -> dùng trung vị chung)
        has_body = X_row is not None
        b_body = float(self.base_body(X_row[None, :])[0]) if has_body else self.global_base

        # (2) [HYBRID] Trộn với lịch sử. w = n / (n + 2):  n=1 -> 0.33 | n=2 -> 0.5 | n=8 -> 0.8
        #     [COLD-START] khách mới: n = 0 -> chỉ dùng số đo
        n = self.user_n.get(str(user_id), 0) if user_id else 0
        if n:
            w = n / (n + self.k_user) if has_body else 1.0
            base = w * self.user_base[str(user_id)] + (1 - w) * b_body
        else:
            w, base = 0.0, b_body

        # (3) Cộng điều chỉnh danh mục + độ "chạy size" của sản phẩm
        cat, off = self.cat_coef.get(category, 0.0), self.item_offset.get(item_id, 0.0)
        return {"base_body": b_body, "user_base": self.user_base.get(str(user_id)) if n else None,
                "user_weight": w, "category_offset": cat, "item_offset": off, "target": base + cat + off}

    def state(self):
        """Tham số cần lưu vào file .pt."""
        return {k: getattr(self, k) for k in ["k_user", "intercept", "w_body", "cat_coef", "step", "item_offset",
                                              "user_base", "user_n", "global_base"]}

    @classmethod
    def from_state(cls, s):
        """Dựng lại SizeModel từ tham số đã lưu."""
        m = cls()
        for k, v in s.items():
            setattr(m, k, v)
        return m


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. ĐÁNH GIÁ ĐỀ XUẤT SIZE
# ════════════════════════════════════════════════════════════════════════════

def nearest(sizes, target):
    """Chọn size có sẵn gần target nhất (bằng khoảng cách thì chọn size nhỏ hơn)."""
    return min(sizes, key=lambda s: (abs(s - target), s))


def evaluate_size_rec(train: pd.DataFrame, test: pd.DataFrame, X_test: np.ndarray, item_sizes: dict,
                      sm: SizeModel) -> dict:
    """Đo độ chính xác của size đề xuất trên tập test."""
    popular = train.groupby("item_id")["size"].agg(lambda x: x.value_counts().index[0]).to_dict()
    fit = np.where(test["y"].values == 1)[0]
    # stats[cách] = [số giao dịch, số lần trùng đúng, số lần lệch ≤ 1 nấc]
    stats = {m: [0, 0, 0] for m in ["popular", "body", "hybrid"]}

    for j, (uid, item, cat, size) in zip(fit, test[["user_id", "item_id", "category", "size"]].values[fit]):
        # Bỏ sản phẩm chỉ có 1 size (không có gì để chọn)
        sizes = sorted(item_sizes.get(item, []))
        if len(sizes) < 2 or size not in sizes:
            continue

        recs = {"popular": popular.get(item, sizes[len(sizes) // 2]),
                "body": nearest(sizes, sm.target(item, cat, X_test[j], None)["target"]),
                "hybrid": nearest(sizes, sm.target(item, cat, X_test[j], uid)["target"])}
        pos = sizes.index(size)
        for m, rec in recs.items():
            st = stats[m]
            st[0] += 1
            st[1] += rec == size
            st[2] += abs(sizes.index(rec) - pos) <= 1 if rec in sizes else 0

    return {m: {"n": n, "exact": e / max(n, 1), "within_1": w / max(n, 1)} for m, (n, e, w) in stats.items()}
