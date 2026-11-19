"""

data.py — ĐỌC, LÀM SẠCH, CHIA VÀ MÃ HOÁ DỮ LIỆU  (bước 1 → 5 của workflow)

MỤC LỤC
    Phần A. Hằng số ............ FIT_MAP, CUP_ORD, LEAKY_FIELDS
    Phần B. Làm sạch ........... parse_height, parse_weight, parse_num, parse_bust_size
    Phần C. Đọc dữ liệu ........ read_json_lines, body_frame, FitData, load_dataset
    Phần D. Mã hoá ............. BodyEncoder, SizeRegressor, Indexer
    Phần E. Chia dữ liệu ....... split, history_bucket, describe

THẺ TÌM NHANH  (bấm Ctrl+F rồi gõ thẻ)
    [RÒ RỈ]      chỗ chống rò rỉ dữ liệu
    [COLD-START] chỗ xử lý khách mới
    [SIZE_GAP]   đặc trưng size_gap
    [ORDINAL]    nhãn có thứ tự chật < vừa < rộng
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. HẰNG SỐ
# ════════════════════════════════════════════════════════════════════════════

# [ORDINAL] Nhãn có thứ tự: chật (0) < vừa (1) < rộng (2).
FIT_MAP = {"small": 0, "fit": 1, "large": 2}
FIT_NAMES = ["small", "fit", "large"]

# Cup áo ngực -> số tăng dần theo kích cỡ. Các cách viết tương đương có cùng giá trị.
CUP_ORD = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "d+": 4.5, "dd": 5, "dd/e": 5, "ddd/e": 5.5,
           "ddd/f": 6, "f": 6, "dddd/g": 7, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11}

# [RÒ RỈ] Các trường chỉ có SAU khi khách nhận hàng -> KHÔNG được dùng làm đầu vào.
#   review_text, review_summary : nội dung đánh giá (hay viết thẳng "chật quá")
#   rating                      : điểm chấm sau khi mặc
#   quality, length             : ModCloth, khách chấm sau khi mặc
LEAKY_FIELDS = ["review_text", "review_summary", "rating", "quality", "length"]


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. LÀM SẠCH  (bước 2)
#  Mỗi hàm xử lý TỪNG Ô độc lập, không học gì từ dữ liệu -> không gây rò rỉ.
#  Ô không đọc được -> NaN (không xoá dòng).
# ════════════════════════════════════════════════════════════════════════════

def _isnull(v) -> bool:
    """Kiểm tra một ô có bị trống không."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def parse_height(v) -> float:
    """Đổi chiều cao về đơn vị INCH."""
    if _isnull(v):
        return np.nan
    s = str(v).lower().strip()

    # Cách 1: có "cm"  ->  chia 2.54
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", s)
    if m:
        return float(m.group(1)) / 2.54

    # Cách 2: dạng feet'inch  ->  12 * feet + inch
    m = re.search(r"(\d+)\s*(?:ft|')\s*(\d+(?:\.\d+)?)?", s)
    if m:
        return 12.0 * int(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)

    # Cách 3: chỉ có một con số  ->  coi là inch
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def parse_weight(v) -> float:
    """Đổi cân nặng về đơn vị LBS."""
    if _isnull(v):
        return np.nan
    s = str(v).lower()
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    x = float(m.group(0))
    return x * 2.20462 if "kg" in s else x


def parse_num(v) -> float:
    """Lấy con số đầu tiên trong chuỗi."""
    if _isnull(v):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else np.nan


def parse_bust_size(v):
    """Tách cỡ áo ngực RentTheRunway thành (vòng dưới ngực, cup)."""
    if _isnull(v):
        return np.nan, np.nan
    s = str(v).lower().strip()
    # phần số ở đầu = vòng dưới ngực ; phần chữ phía sau = cup
    m = re.match(r"^(\d+)\s*([a-z/+]*)$", s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), CUP_ORD.get(m.group(2), np.nan)


def detect_dataset(path: str) -> str:
    """Đoán tên bộ dữ liệu từ tên file."""
    p = os.path.basename(path).lower()
    if "modcloth" in p:
        return "modcloth"
    if "runway" in p or "rtr" in p:
        return "rtr"
    return os.path.splitext(os.path.splitext(p)[0])[0]


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN C. ĐỌC DỮ LIỆU  (bước 1 + 2)
# ════════════════════════════════════════════════════════════════════════════

def read_json_lines(path: str) -> pd.DataFrame:
    """BƯỚC 1. Đọc file dữ liệu thô."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def body_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """BƯỚC 2. Lấy các số đo cơ thể dạng SỐ từ bảng thô."""
    # Cột không có trong bộ dữ liệu (vd ModCloth không có weight) -> cột toàn NaN
    col = lambda c: raw[c] if c in raw else pd.Series(np.nan, index=raw.index)
    num = pd.DataFrame(index=raw.index)

    # (1) Đổi từng cột sang số
    num["height"] = col("height").map(parse_height)
    num["weight"] = col("weight").map(parse_weight)
    num["age"] = col("age").map(parse_num)

    # (2) Áo ngực: RTR ghi chung "34d"; ModCloth ghi riêng "bra size" + "cup size"
    band_cup = col("bust size").map(parse_bust_size)
    num["bust_band"] = [b for b, _ in band_cup]
    num["cup"] = [c for _, c in band_cup]
    if "bra size" in raw:
        num["bust_band"] = num["bust_band"].fillna(raw["bra size"].map(parse_num))
    if "cup size" in raw:
        num["cup"] = num["cup"].fillna(raw["cup size"].map(lambda x: CUP_ORD.get(str(x).lower(), np.nan)))

    num["bust"] = col("bust").map(parse_num)
    num["waist"] = col("waist").map(parse_num)
    num["hips"] = col("hips").map(parse_num)
    num["shoe"] = col("shoe size").map(parse_num)

    # (3) BMI (đơn vị Mỹ) = cân nặng(lbs) x 703 / chiều cao(inch)²
    num["bmi"] = num["weight"] * 703.0 / num["height"] ** 2

    # (4) Giá trị vô lý -> NaN (KHÔNG xoá dòng)
    num.loc[(num["height"] < 48) | (num["height"] > 84), "height"] = np.nan
    num.loc[~num["bmi"].between(12, 60), "bmi"] = np.nan
    num.loc[num["shoe"] > 15, "shoe"] = np.nan
    return num.astype(float)


@dataclass
class FitData:
    """Hộp chứa bộ dữ liệu đã làm sạch."""
    name: str
    df: pd.DataFrame
    body: pd.DataFrame
    body_cols: list = field(default_factory=list)

    def __len__(self):
        return len(self.df)


def load_dataset(path: str) -> FitData:
    """BƯỚC 1 + 2. Đọc và làm sạch 1 file dữ liệu."""
    # (1) Đọc thô, chỉ giữ dòng có nhãn hợp lệ small / fit / large
    raw = read_json_lines(path)
    raw = raw[raw["fit"].isin(FIT_MAP)].reset_index(drop=True)

    # (2) Bảng giao dịch: CHỈ các cột dưới đây (không có review, rating...)
    df = pd.DataFrame({
        "user_id": raw["user_id"].astype(str),
        "item_id": raw["item_id"].astype(str),
        "size": pd.to_numeric(raw["size"], errors="coerce").fillna(0.0),
        "category": raw["category"].fillna("unknown").astype(str) if "category" in raw else "unknown",
        "context": raw["rented for"].fillna("unknown").astype(str) if "rented for" in raw else "unknown",
        "body_type": raw["body type"].fillna("unknown").astype(str) if "body type" in raw else "unknown",
        "y": raw["fit"].map(FIT_MAP).astype(int),
    })

    # (3) item_size = "item_id|size", vd "2260466|8".
    df["item_size"] = df["item_id"] + "|" + df["size"].map(lambda s: f"{s:g}")

    # (4) Số đo cơ thể; bỏ cột gần như trống (< 1% có giá trị)
    body = body_frame(raw)
    keep = [c for c in body.columns if body[c].notna().mean() > 0.01]
    return FitData(detect_dataset(path), df, body[keep], keep)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN D. MÃ HOÁ  (bước 4 + 5)
#  Mỗi lớp có 2 pha:
#      fit(...)       học tham số          -> [RÒ RỈ] CHỈ gọi trên tập TRAIN
#      transform(...) áp dụng tham số đó   -> cho train, val, test và khách mới
#  state() / from_state(): lưu tham số vào file .pt và nạp lại khi gợi ý.
# ════════════════════════════════════════════════════════════════════════════

class BodyEncoder:
    """BƯỚC 4–5. Biến số đo cơ thể thành vector số cho mạng nơ-ron."""

    def fit(self, body: pd.DataFrame, body_type: pd.Series) -> "BodyEncoder":
        """Học tham số chuẩn hoá từ tập TRAIN."""
        self.cols = list(body.columns)
        self.mu = {c: float(body[c].mean()) if body[c].notna().any() else 0.0 for c in self.cols}
        self.sd = {c: float(body[c].std()) if body[c].notna().sum() > 1 and body[c].std() > 0 else 1.0
                   for c in self.cols}
        # Chỉ tạo cờ thiếu cho cột hay bị thiếu (>= 2%)
        self.miss_cols = [c for c in self.cols if body[c].isna().mean() >= 0.02]
        # Chỉ one-hot khi có từ 2 loại dáng người trở lên
        cats = sorted(body_type.unique())
        self.types = cats if len(cats) > 1 else []
        return self

    @property
    def dim(self) -> int:
        """Số chiều của vector = số số đo + số cờ thiếu + số loại dáng người."""
        return len(self.cols) + len(self.miss_cols) + len(self.types)

    def feature_names(self) -> list:
        """Tên từng chiều của vector (dùng khi in bảng giải thích --explain)."""
        return [f"{c}" for c in self.cols] + [f"{c}_missing" for c in self.miss_cols] + \
               [f"body_type={t}" for t in self.types]

    def transform(self, body: pd.DataFrame, body_type: pd.Series) -> np.ndarray:
        """Áp dụng tham số đã học cho dữ liệu bất kỳ (train, val, test hoặc khách mới)."""
        body = body.reindex(columns=self.cols)
        # (1) z-score, ô thiếu -> 0, cắt [-5, 5]
        z = np.stack([((body[c] - self.mu[c]) / self.sd[c]).fillna(0.0).clip(-5, 5).values
                      for c in self.cols], 1) if self.cols else np.zeros((len(body), 0))
        # (2) cờ thiếu: 1 = ô trống
        miss = body[self.miss_cols].isna().values.astype(float) if self.miss_cols else np.zeros((len(body), 0))
        # (3) one-hot dáng người
        oh = np.stack([(body_type == t).values for t in self.types], 1).astype(float) if self.types \
            else np.zeros((len(body), 0))
        return np.hstack([z, miss, oh]).astype(np.float32)

    def state(self) -> dict:
        """Tham số cần lưu vào file mô hình .pt."""
        return {"cols": self.cols, "miss_cols": self.miss_cols, "mu": self.mu, "sd": self.sd, "types": self.types}

    @classmethod
    def from_state(cls, s: dict) -> "BodyEncoder":
        """Dựng lại BodyEncoder từ tham số đã lưu (không cần fit lại, không cần dữ liệu train)."""
        e = cls()
        e.cols, e.mu, e.sd, e.types = list(s["cols"]), dict(s["mu"]), dict(s["sd"]), list(s["types"])
        e.miss_cols = list(s["miss_cols"])
        return e


class SizeRegressor:
    """[SIZE_GAP] Ước lượng size "tự nhiên" của khách từ số đo cơ thể (Ridge regression)."""

    def fit(self, X: np.ndarray, category: pd.Series, size: pd.Series, alpha: float = 1.0) -> "SizeRegressor":
        """Giải Ridge trên tập TRAIN."""
        # Chỉ tạo hệ số riêng cho danh mục có >= 30 giao dịch
        vc = category.value_counts()
        self.cats = list(vc[vc >= 30].index)

        # Nghiệm đóng: coef = (AᵀA + αI)⁻¹ Aᵀy   (không phạt hệ số chặn)
        A = self._design(X, category)
        y = size.values.astype(float)
        reg = alpha * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(A.T @ A + reg, A.T @ y).tolist()

        # sd phần dư: để chuẩn hoá gap (gap = 1 nghĩa là lệch 1 độ lệch chuẩn)
        self.sd = float(np.std(y - A @ np.array(self.coef))) or 1.0
        return self

    def _design(self, X, category):
        """Tạo ma trận đầu vào A = [ cột toàn số 1 | X_body | one-hot danh mục ]."""
        oh = np.stack([(category == c).values for c in self.cats], 1).astype(float) if self.cats \
            else np.zeros((len(X), 0))
        return np.hstack([np.ones((len(X), 1)), X, oh])

    def predict(self, X, category) -> np.ndarray:
        """Size ước lượng = A · coef."""
        return self._design(X, category) @ np.array(self.coef)

    def gap(self, X, category, size) -> np.ndarray:
        """[SIZE_GAP] Tính (size đặt - size ước lượng) / sd, cắt trong [-5, 5]."""
        return ((size.values - self.predict(X, category)) / self.sd).clip(-5, 5).astype(np.float32)

    def state(self):
        """Tham số cần lưu vào file .pt."""
        return {"cats": self.cats, "coef": self.coef, "sd": self.sd}

    @classmethod
    def from_state(cls, s):
        """Dựng lại từ tham số đã lưu."""
        r = cls()
        r.cats, r.coef, r.sd = list(s["cats"]), list(s["coef"]), s["sd"]
        return r


# Các cột id cần đổi sang chỉ số nguyên để tra bảng Embedding
INDEX_COLS = ["user_id", "item_size", "item_id", "category", "context"]


class Indexer:
    """BƯỚC 4–5. Đổi id (user, sản phẩm, danh mục...) thành chỉ số nguyên. Chỉ số 0 = "chưa biết"."""

    def fit(self, df: pd.DataFrame) -> "Indexer":
        """Học bảng mã từ TRAIN: đánh số 1, 2, 3... cho mỗi id; lưu mu, sd của size để chuẩn hoá."""
        self.maps = {c: {k: i + 1 for i, k in enumerate(df[c].unique())} for c in INDEX_COLS}
        self.size_mu = float(df["size"].mean())
        self.size_sd = float(df["size"].std()) or 1.0
        return self

    def sizes(self) -> dict:
        """Số id đã biết của mỗi cột (để tạo bảng Embedding đúng kích thước)."""
        return {c: len(m) for c, m in self.maps.items()}

    def transform(self, df: pd.DataFrame) -> dict:
        """Đổi id thành chỉ số; id lạ (không có trong train) -> 0."""
        # id lạ -> NaN -> fillna(0) -> chỉ số 0
        out = {c: df[c].map(self.maps[c]).fillna(0).astype(np.int64).values for c in INDEX_COLS}
        out["size_z"] = ((df["size"] - self.size_mu) / self.size_sd).values.astype(np.float32)
        return out

    def state(self) -> dict:
        """Tham số cần lưu vào file .pt."""
        return {"maps": self.maps, "size_mu": self.size_mu, "size_sd": self.size_sd}

    @classmethod
    def from_state(cls, s: dict) -> "Indexer":
        """Dựng lại từ tham số đã lưu."""
        ix = cls()
        ix.maps, ix.size_mu, ix.size_sd = s["maps"], s["size_mu"], s["size_sd"]
        return ix


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN E. CHIA DỮ LIỆU  (bước 3)
# ════════════════════════════════════════════════════════════════════════════

def split(df: pd.DataFrame, protocol: str, seed: int):
    """BƯỚC 3. Chia dữ liệu 80% train / 10% val / 10% test."""
    rng = np.random.RandomState(seed)

    # ── warm: mỗi giao dịch 1 số ngẫu nhiên r trong [0, 1)
    if protocol == "warm":
        r = rng.rand(len(df))
        return np.where(r < .8)[0], np.where((r >= .8) & (r < .9))[0], np.where(r >= .9)[0]
    if protocol != "cold":
        raise ValueError(f"giao thức không hợp lệ: {protocol}")

    users = np.array(df["user_id"].unique(), dtype=object)
    rng.shuffle(users)
    n = len(users)
    tr, va = set(users[:int(.8 * n)]), set(users[int(.8 * n):int(.9 * n)])
    m_tr = df["user_id"].isin(tr).values
    m_va = df["user_id"].isin(va).values
    return np.where(m_tr)[0], np.where(m_va)[0], np.where(~m_tr & ~m_va)[0]


def history_bucket(n_hist: np.ndarray) -> np.ndarray:
    """Xếp nhóm theo số giao dịch của user trong train: "0", "1", "2-4", "5+"."""
    return np.select([n_hist == 0, n_hist == 1, n_hist <= 4], ["0", "1", "2-4"], "5+")


def describe(data: FitData) -> dict:
    """Thống kê mô tả bộ dữ liệu (lệnh `sizerec data`)."""
    df, c = data.df, data.df["user_id"].value_counts()
    return {
        "dataset": data.name,
        "interactions": len(df),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "item_sizes": int(df["item_size"].nunique()),
        "categories": int(df["category"].nunique()),
        "users_with_1_interaction": float((c == 1).mean()),
        "fit_ratio": {k: float((df["y"] == v).mean()) for k, v in FIT_MAP.items()},
        "body_missing": {k: float(data.body[k].isna().mean()) for k in data.body_cols},
    }
