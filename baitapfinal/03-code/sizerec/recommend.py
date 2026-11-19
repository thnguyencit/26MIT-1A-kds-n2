"""

recommend.py — GỢI Ý SIZE CHO MỘT KHÁCH VÀ GIẢI THÍCH  (bước 8, dùng thật)

MỤC LỤC
    Phần A. Nhập số đo ......... BODY_ARGS, body_from_inputs
    Phần B. Gợi ý .............. recommend, _advice
    Phần C. Giải thích ......... body_contributions, TERM_LABELS, FIT_LABELS

THẺ TÌM NHANH  (Ctrl+F)
    [COLD-START] chỗ xử lý khách mới
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import CUP_ORD, FIT_NAMES, body_frame
from .sizing import nearest
from .trainer import Bundle


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. NHẬP SỐ ĐO
# ════════════════════════════════════════════════════════════════════════════

# Tên tham số dòng lệnh  ->  tên cột trong file dữ liệu gốc
BODY_ARGS = {
    "height": "height", "weight": "weight", "age": "age", "bust_size": "bust size", "bra_size": "bra size",
    "cup": "cup size", "bust": "bust", "waist": "waist", "hips": "hips", "shoe": "shoe size",
}


def body_from_inputs(values: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Đổi số đo khách nhập (dòng lệnh hoặc menu) thành bảng số đo."""
    # (1) Tạo "bản ghi thô" 1 dòng giống file dữ liệu gốc (bỏ ô không nhập)
    raw = pd.DataFrame([{BODY_ARGS[k]: v for k, v in values.items() if k in BODY_ARGS and v is not None}])

    # (2) Kiểm tra cup hợp lệ
    if "cup size" in raw and str(raw.loc[0, "cup size"]).lower() not in CUP_ORD:
        raise ValueError(f"cup không hợp lệ: {raw.loc[0, 'cup size']} (ví dụ hợp lệ: a, b, c, d, dd/e)")

    # (3) Làm sạch bằng đúng hàm lúc train
    body = body_frame(raw)
    bt = pd.Series([values.get("body_type") or "unknown"])
    return body, bt


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. GỢI Ý
# ════════════════════════════════════════════════════════════════════════════

def recommend(bundle: Bundle, item_id: str, user_id: str | None = None, body_inputs: dict | None = None,
              context: str = "unknown") -> dict:
    """BƯỚC 8. Gợi ý size cho 1 sản phẩm (hàm chính của lệnh `sizerec recommend`)."""
    # (1) Lấy các size có sẵn của sản phẩm
    item_id = str(item_id)
    if item_id not in bundle.item_sizes:
        raise KeyError(f"Không tìm thấy item_id '{item_id}' trong dữ liệu huấn luyện của mô hình.")
    sizes = bundle.item_sizes[item_id]

    # [COLD-START] khách "cũ" = user_id có trong bảng mã train; ngược lại là khách mới (chỉ số 0)
    known_user = user_id is not None and str(user_id) in bundle.indexer.maps["user_id"]
    body, bt = body_from_inputs(body_inputs or {})

    # (2) Mỗi size 1 dòng: cùng khách, cùng sản phẩm, chỉ khác size
    rows = pd.DataFrame({
        "user_id": [str(user_id) if user_id else "__new__"] * len(sizes),
        "item_id": [item_id] * len(sizes),
        "size": sizes,
        "category": [bundle.item_category.get(item_id, "unknown")] * len(sizes),
        "context": [context] * len(sizes),
        "body_type": list(bt) * len(sizes),
    })
    rows["item_size"] = rows["item_id"] + "|" + rows["size"].map(lambda s: f"{s:g}")
    body_rep = pd.concat([body] * len(sizes), ignore_index=True)
    b = bundle.batch(rows, body_rep)

    # (3) Mô hình hybrid MF: xác suất + các thành phần của điểm t + 2 ngưỡng θ
    with torch.no_grad():
        probs = bundle.model(b).numpy()
        terms = {k: v.numpy() for k, v in bundle.model.terms(b).items()}
        th0, th1 = [float(t) for t in bundle.model.thresholds()]

    # (4) SizeModel: size mục tiêu -> size có sẵn gần nhất (khách không nhập số đo -> X_row = None)
    has_body = any(body[c].notna().iloc[0] for c in body.columns) or bt.iloc[0] != "unknown"
    X1 = bundle.body_enc.transform(body, bt)[0]
    tgt = bundle.size_model.target(item_id, rows["category"].iloc[0], X1 if has_body else None,
                                   str(user_id) if known_user else None)
    rec_size = nearest(sizes, tgt["target"])

    # (5) Bảng xếp hạng: size gần mục tiêu nhất đứng đầu (= size đề xuất)
    ranking = []
    for i, s in enumerate(sizes):
        ranking.append({"size": s, "distance": abs(s - tgt["target"]), "p_small": float(probs[i, 0]),
                        "p_fit": float(probs[i, 1]), "p_large": float(probs[i, 2]),
                        "score_t": float(sum(t[i] for t in terms.values())),
                        "terms": {k: float(v[i]) for k, v in terms.items()}})
    ranking.sort(key=lambda r: (r["distance"], r["size"]))
    best = ranking[0]

    # (6) Lời khuyên + đóng gói kết quả
    advice = _advice(best, ranking)
    filled = [c for c in body.columns if body[c].notna().iloc[0]]
    return {
        "item_id": item_id, "category": rows["category"].iloc[0],
        "user_id": user_id, "user_status": "đã có lịch sử" if known_user else "người dùng mới (cold-start)",
        "user_history": int(bundle.user_counts.get(str(user_id), 0)) if user_id else 0,
        "body_used": {c: float(body[c].iloc[0]) for c in filled},
        "recommended_size": rec_size, "fit_probability": best["p_fit"],
        "risk_small": best["p_small"], "risk_large": best["p_large"], "advice": advice,
        "size_estimate": tgt,
        "thresholds": {"theta_small": th0, "theta_large": th1}, "ranking": ranking,
    }


def _advice(best: dict, ranking: list) -> str | None:
    """Khuyên đổi size CHỈ KHI size kế bên thật sự an toàn hơn."""
    by_size = sorted(ranking, key=lambda r: r["size"])
    i = next(k for k, r in enumerate(by_size) if r["size"] == best["size"])
    miss = lambda r: r["p_small"] + r["p_large"]

    # (1) Xác định hướng rủi ro
    if best["p_small"] - best["p_large"] > 0.05 and best["p_small"] > 0.25:
        risk, j, word, side = "p_small", i + 1, "CHẬT", "lớn hơn"
    elif best["p_large"] - best["p_small"] > 0.05 and best["p_large"] > 0.25:
        risk, j, word, side = "p_large", i - 1, "RỘNG", "nhỏ hơn"
    else:
        return None

    # (2) Soạn câu báo rủi ro
    level = "khá cao" if best[risk] >= 0.4 else "hơi cao"
    msg = f"Mô hình hybrid thấy rủi ro {word} ở size này {level} ({best[risk]:.0%})."

    # (3) Không có size kế bên -> giữ nguyên
    if not 0 <= j < len(by_size):
        return msg + f" Sản phẩm không có size {side}, nên giữ size đề xuất."

    # (4) Size kế bên an toàn hơn thật -> gợi ý cân nhắc; ngược lại giữ nguyên
    nb = by_size[j]
    if nb[risk] < best[risk] and miss(nb) <= miss(best):
        return msg + (f" Nếu phân vân, cân nhắc size {nb['size']:g}: rủi ro {word.lower()} {nb[risk]:.0%}, "
                      f"P(vừa) {nb['p_fit']:.0%}.")
    return msg + (f" Tuy vậy size {side} kế tiếp ({nb['size']:g}) không an toàn hơn "
                  f"(P(vừa) {nb['p_fit']:.0%}), nên vẫn giữ size {best['size']:g}.")


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN C. GIẢI THÍCH
# ════════════════════════════════════════════════════════════════════════════

def body_contributions(bundle: Bundle, body_inputs: dict, top: int = 5) -> list:
    """Cho biết số đo nào ảnh hưởng mạnh nhất tới điểm t (qua phần tuyến tính g(x))."""
    body, bt = body_from_inputs(body_inputs or {})
    x = bundle.body_enc.transform(body, bt)[0]
    w = bundle.model.g.weight.detach().numpy()[0]
    contrib = x * w
    names = bundle.body_enc.feature_names()
    order = np.argsort(-np.abs(contrib))
    return [{"feature": names[i], "value_z": float(x[i]), "contribution": float(contrib[i])}
            for i in order[:top] if abs(contrib[i]) > 1e-6]


# Tên tiếng Việt cho 4 thành phần của điểm t (khớp với model.OrdinalHybridMF.terms)
TERM_LABELS = {
    "item_bias": "Sản phẩm & size (độ lệch chung của item-size)",
    "user_collab": "Lịch sử cộng tác của người dùng (p_u, b_u)",
    "body": "Số đo cơ thể (E(x), g(x))",
    "size_gap": "Chênh lệch size so với size ước lượng từ số đo",
}
FIT_LABELS = dict(zip(FIT_NAMES, ["chật (small)", "vừa (fit)", "rộng (large)"]))
