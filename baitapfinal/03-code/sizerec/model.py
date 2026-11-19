"""

model.py — KIẾN TRÚC MÔ HÌNH ORDINAL HYBRID MATRIX FACTORIZATION  (bước 6 của workflow)

MỤC LỤC
    Phần A. Cấu hình ........... ModelSpec, SPECS, PROPOSED, DESCRIPTIONS
    Phần B. Mô hình ............ OrdinalHybridMF (__init__, thresholds, terms, forward, l2_terms)

THẺ TÌM NHANH  (Ctrl+F)
    [HYBRID]     chỗ kết hợp lịch sử (cộng tác) + số đo cơ thể
    [COLD-START] chỗ xử lý khách mới
    [SIZE_GAP]   đặc trưng size_gap
    [ORDINAL]    đầu ra thứ bậc chật < vừa < rộng

CÔNG THỨC
    u = m_u · p_u + E(x_u)                                   vector người dùng
    v = q_(item,size) + q_item [+ c_cat + c_ctx + size_z·w + gap·w_gap]    vector sản phẩm-size
    t = <u, v> + m_u · b_u + b_(item,size) + b_item + g(x_u) [+ β · gap]   điểm "chật → rộng"
    P(y ≤ k) = sigmoid(θ_k - t)          k = chật, vừa

    m_u = 0 với khách mới (cold-start) hoặc khi bị ID dropout lúc huấn luyện.
    t nhỏ -> chật  |  t lớn -> rộng.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. CẤU HÌNH
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    """Bộ công tắc bật/tắt từng nhóm thông tin của mô hình.

    Ý nghĩa   : chỉ cần 1 lớp OrdinalHybridMF mà tạo được mọi biến thể để so sánh (ablation).
    use_id      : dùng lịch sử cộng tác của người dùng (p_u, b_u)
    use_body    : dùng số đo cơ thể (E, g)
    use_product : dùng đặc trưng sản phẩm (danh mục, dịp, size, size_gap)
    id_drop     : xác suất ẩn user_id lúc train (chỉ có tác dụng khi use_id và use_body)
    frozen=True : tạo xong thì không sửa được, tránh vô tình đổi cấu hình giữa chừng.
    """
    use_id: bool = True
    use_body: bool = True
    use_product: bool = True
    id_drop: float = 0.1

    def to_dict(self):
        """Đổi sang dict để lưu vào file .pt."""
        return asdict(self)


# Các mô hình so sánh (ablation): bỏ dần từng thành phần để thấy đóng góp của nó.
#
#   Tên                         lịch sử  cơ thể  sản phẩm  id_drop
#   ─────────────────────────   ───────  ──────  ────────  ───────
#   mf                             ✓       ✗        ✗        0
#   mf_body                        ✓       ✓        ✗        0.1
#   mf_body_product   (ĐỀ XUẤT)    ✓       ✓        ✓        0.1
#   body_product                   ✗       ✓        ✓        0
#   mf_body_product_nodrop         ✓       ✓        ✓        0
SPECS = {
    "mf": ModelSpec(use_id=True, use_body=False, use_product=False, id_drop=0.0),
    "mf_body": ModelSpec(use_id=True, use_body=True, use_product=False),
    "mf_body_product": ModelSpec(use_id=True, use_body=True, use_product=True),
    "body_product": ModelSpec(use_id=False, use_body=True, use_product=True, id_drop=0.0),
    "mf_body_product_nodrop": ModelSpec(use_id=True, use_body=True, use_product=True, id_drop=0.0),
}
PROPOSED = "mf_body_product"

# Mô tả in ra ở lệnh `sizerec info`
DESCRIPTIONS = {
    "majority": "Luôn dự đoán 'fit' (mốc dưới)",
    "body_lr": "Logistic Regression trên số đo + sản phẩm (thuộc tính tường minh, tuyến tính)",
    "mf": "MF thứ bậc thuần: chỉ latent factor cộng tác (user, item, item-size)",
    "mf_body": "MF + số đo cơ thể tích hợp vào vector ẩn người dùng",
    "mf_body_product": "MF + số đo cơ thể + đặc trưng sản phẩm (MÔ HÌNH ĐỀ XUẤT)",
    "body_product": "Chỉ số đo + sản phẩm, không có latent của người dùng (phi tuyến)",
    "mf_body_product_nodrop": "Mô hình đề xuất nhưng không ID dropout (ablation)",
}


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. MÔ HÌNH
# ════════════════════════════════════════════════════════════════════════════

class OrdinalHybridMF(nn.Module):
    """Mạng PyTorch hiện thực công thức ở đầu file (MÔ HÌNH CHÍNH của đề tài)."""

    def __init__(self, n: dict, d_body: int, spec: ModelSpec, dim: int = 16, hidden: int = 64):
        """Tạo các lớp và tham số của mạng."""
        super().__init__()
        self.spec, self.dim = spec, dim

        # (1) Bảng Embedding: hàng i = vector của id có chỉ số i.
        #     [COLD-START] +1 hàng cho chỉ số 0 ("chưa biết"); padding_idx=0 giữ hàng 0 luôn = 0.
        emb = lambda k, d: nn.Embedding(n[k] + 1, d, padding_idx=0)
        self.P, self.bu = emb("user_id", dim), emb("user_id", 1)
        self.Q_is, self.b_is = emb("item_size", dim), emb("item_size", 1)
        self.Q_i, self.b_i = emb("item_id", dim), emb("item_id", 1)
        self.C_cat = emb("category", dim)
        self.C_ctx = emb("context", dim)

        # (2) Tham số cho size và size_gap
        self.w_size = nn.Parameter(torch.zeros(dim))
        self.w_gap = nn.Parameter(torch.zeros(dim))
        self.beta_gap = nn.Parameter(torch.tensor(0.0))

        # (3) Khởi tạo: vector ẩn ngẫu nhiên nhỏ (std 0.05), độ lệch = 0, hàng 0 = 0
        for e in [self.P, self.Q_is, self.Q_i, self.C_cat, self.C_ctx]:
            nn.init.normal_(e.weight, std=0.05)
        for e in [self.bu, self.b_is, self.b_i]:
            nn.init.zeros_(e.weight)
        with torch.no_grad():
            for e in [self.P, self.Q_is, self.Q_i, self.C_cat, self.C_ctx, self.bu, self.b_is, self.b_i]:
                e.weight[0].zero_()

        # (4) [HYBRID] Phần số đo cơ thể
        self.E = nn.Sequential(nn.Linear(d_body, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, dim))
        self.g = nn.Linear(d_body, 1)

        # (5) [ORDINAL] Hai ngưỡng: θ_0 = theta0 ; θ_1 = theta0 + softplus(dtheta)
        self.theta0 = nn.Parameter(torch.tensor(-1.0))
        self.dtheta = nn.Parameter(torch.tensor(1.0))

    # ────────────────────────────────────────────────────────────────────────
    def thresholds(self):
        """[ORDINAL] Trả về 2 ngưỡng (θ_0, θ_1)."""
        return self.theta0, self.theta0 + F.softplus(self.dtheta)

    def terms(self, b: dict) -> dict:
        """Tính điểm t và tách nó thành 4 phần (để giải thích được)."""
        s = self.spec

        # (1) Vector sản phẩm-size v
        v = self.Q_is(b["item_size"]) + self.Q_i(b["item_id"])
        if s.use_product:
            v = v + self.C_cat(b["category"]) + self.C_ctx(b["context"]) + b["size_z"][:, None] * self.w_size
            if s.use_body:
                # [SIZE_GAP] gap cần số đo nên chỉ dùng khi có cả body
                v = v + b["gap"][:, None] * self.w_gap

        # (2) Độ lệch sản phẩm: b_(item,size) + b_item  (luôn có)
        zero = torch.zeros(v.shape[0], device=v.device)
        out = {"item_bias": self.b_is(b["item_size"])[:, 0] + self.b_i(b["item_id"])[:, 0],
               "user_collab": zero, "body": zero, "size_gap": zero}

        # (3) [SIZE_GAP] ảnh hưởng trực tiếp: β · gap
        if s.use_product and s.use_body:
            out["size_gap"] = self.beta_gap * b["gap"]

        # (4) [HYBRID] Phần lịch sử: <p_u, v> + b_u
        #     [COLD-START] khách mới có user_id = 0 -> P và bu trả về 0 -> phần này = 0
        if s.use_id:
            out["user_collab"] = (self.P(b["user_id"]) * v).sum(-1) + self.bu(b["user_id"])[:, 0]

        # (5) [HYBRID] Phần cơ thể: <E(x), v> + g(x)  -> luôn có giá trị, kể cả khách mới
        if s.use_body:
            out["body"] = (self.E(b["x"]) * v).sum(-1) + self.g(b["x"])[:, 0]
        return out

    def forward(self, b: dict) -> torch.Tensor:
        """[ORDINAL] Đổi điểm t thành [P(chật), P(vừa), P(rộng)]."""
        # (1) Điểm t = tổng 4 phần
        t = sum(self.terms(b).values())

        # (2) Xác suất tích luỹ: c0 = P(y ≤ chật), c1 = P(y ≤ vừa)
        th0, th1 = self.thresholds()
        c0, c1 = torch.sigmoid(th0 - t), torch.sigmoid(th1 - t)

        # (3) P(chật) = c0 ; P(vừa) = c1 - c0 ; P(rộng) = 1 - c1   (clamp để tránh log(0))
        return torch.stack([c0, c1 - c0, 1 - c1], -1).clamp_min(1e-7)

    def l2_terms(self, b: dict) -> torch.Tensor:
        """Phạt L2 để chống overfit."""
        pairs = [(self.Q_is, b["item_size"]), (self.Q_i, b["item_id"]), (self.b_is, b["item_size"]),
                 (self.b_i, b["item_id"])]
        if self.spec.use_id:
            pairs += [(self.P, b["user_id"]), (self.bu, b["user_id"])]
        return sum((e(k) ** 2).sum() for e, k in pairs) / len(b["y"])
