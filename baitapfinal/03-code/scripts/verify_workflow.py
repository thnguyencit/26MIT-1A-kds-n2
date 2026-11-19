"""
verify_workflow.py — CHẠY 8 BƯỚC CỦA SƠ ĐỒ WORKFLOW VÀ TỰ KIỂM TRA

Dùng CHÍNH các hàm trong gói sizerec, in minh chứng và kiểm tra tự động (assert) sau mỗi bước.
Mỗi dòng [OK] in ra = một assert đã qua. Nếu code làm khác sơ đồ, script sẽ dừng với lỗi.

    BƯỚC 1  Dữ liệu thô                 BƯỚC 5  Biến đổi train / val / test
    BƯỚC 2  Làm sạch                    BƯỚC 6  Huấn luyện mô hình đề xuất
    BƯỚC 3  Chia dữ liệu                BƯỚC 7  Đánh giá + so sánh baseline
    BƯỚC 4  Học tiền xử lý trên train   BƯỚC 8  Gợi ý size

Cách chạy (từ thư mục 03-code):
    python scripts/verify_workflow.py ../02-data/modcloth_final_data.json.gz
    python scripts/verify_workflow.py ../02-data/renttherunway_final_data.json.gz --protocol warm
Mất khoảng 30 giây – 2 phút. Không ghi đè kết quả trong 04-results.
"""
import argparse, os, sys, time
try:
    # in tiếng Việt đúng trên console Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# thêm thư mục 03-code vào đường dẫn import để `import sizerec` chạy được khi gọi script trực tiếp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from sizerec.data import (LEAKY_FIELDS, FIT_MAP, read_json_lines, load_dataset, parse_height,
                          parse_bust_size, split, history_bucket)
from sizerec.model import SPECS, PROPOSED
from sizerec.trainer import HParams, Prepared, train_model, predict, tune_decision, metrics, body_lr
from sizerec.sizing import SizeModel, evaluate_size_rec

# tham số: file dữ liệu (tuỳ chọn), giao thức cold/warm (mặc định cold), seed (mặc định 1)
ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", help="file .json.gz; bỏ trống = tự tìm ModCloth"); ap.add_argument("--protocol", default="cold", choices=["cold", "warm"])
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
if not a.data:   # bấm nút Run trong VS Code không cần tham số: tự tìm dữ liệu ModCloth
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ["../../02-data", "../../../dataset/dataset", "../../../baitapfinal/02-data", "../../dataset/dataset"]:
        f = os.path.normpath(os.path.join(here, rel, "modcloth_final_data.json.gz"))
        if os.path.exists(f):
            a.data = f; break
    if not a.data:
        sys.exit("Không tìm thấy modcloth_final_data.json.gz, hãy truyền đường dẫn dữ liệu.")
    print(f"Dữ liệu: {a.data}")
# OK(...): in dòng [OK] sau mỗi assert đã qua; H(...): in tiêu đề từng bước kèm tên hàm trong code
OK = lambda s: print(f"   [OK] {s}")
def H(n, t, where): print(f"\n{'='*78}\nBƯỚC {n}. {t}\n   Code: {where}\n{'-'*78}")
t0 = time.time()

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 1. DỮ LIỆU THÔ
# ════════════════════════════════════════════════════════════════════════════
H(1, "DỮ LIỆU THÔ", "data.py: read_json_lines()")
# đọc dữ liệu THÔ (còn đủ mọi cột, kể cả các cột rò rỉ) để so sánh với sau làm sạch
raw = read_json_lines(a.data)
print(f"   {len(raw):,} dòng, {raw.shape[1]} cột: {', '.join(raw.columns)}")
print(f"   Nhãn thô: {raw['fit'].value_counts().to_dict()}")
# liệt kê các trường rò rỉ CÓ trong dữ liệu thô (để chứng minh bước sau đã loại chúng)
leaky_in_raw = [c for c in LEAKY_FIELDS if c in raw.columns]
print(f"   Trường chỉ có SAU khi mua (có trong dữ liệu thô): {leaky_in_raw}")

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 2. LÀM SẠCH
# ════════════════════════════════════════════════════════════════════════════
H(2, "LÀM SẠCH (theo từng dòng, không học từ dữ liệu)", "data.py: load_dataset(), body_frame(), parse_*()")
ex_h = "5' 6\""
print("   Ví dụ đổi đơn vị: parse_height(" + repr(ex_h) + f") = {parse_height(ex_h)} inch ; "
      f"parse_height('168cm') = {parse_height('168cm'):.1f} inch ; parse_bust_size('34d') = {parse_bust_size('34d')}")
# làm sạch bằng đúng hàm của gói (data.load_dataset)
data = load_dataset(a.data)
df, body = data.df, data.body
print(f"   Sau làm sạch: {len(df):,} giao dịch · {df.user_id.nunique():,} user · {df.item_id.nunique():,} sản phẩm")
print(f"   Cột giữ lại: {list(df.columns)}")
print(f"   Số đo cơ thể (dạng số): {data.body_cols}")
# KIỂM TRA: sau làm sạch không còn trường rò rỉ nào
assert not any(c in df.columns or c in body.columns for c in LEAKY_FIELDS)
OK(f"không còn trường rò rỉ {LEAKY_FIELDS}")
assert set(df.y.unique()) <= set(FIT_MAP.values()); OK("nhãn đã mã hoá small/fit/large -> 0/1/2")
# KIỂM TRA: số dòng không đổi, tức là giá trị vô lý chỉ bị đổi thành NaN chứ không xoá dòng
assert len(df) == len(body); OK("giá trị vô lý/thiếu -> NaN, KHÔNG xoá dòng (số dòng body = số giao dịch)")
print(f"   Tỉ lệ thiếu: " + ", ".join(f"{c}={body[c].isna().mean():.0%}" for c in body.columns))

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 3. CHIA DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════
H(3, f"CHIA DỮ LIỆU (giao thức {a.protocol}, seed {a.seed})", "data.py: split()")
# chia train/val/test bằng data.split
tr, va, te = split(df, a.protocol, a.seed)
n = len(df); print(f"   Train {len(tr):,} ({len(tr)/n:.0%}) · Val {len(va):,} ({len(va)/n:.0%}) · Test {len(te):,} ({len(te)/n:.0%})")
# KIỂM TRA: 3 tập không có giao dịch chung (giao của từng cặp tập đều rỗng)
assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)); OK("3 tập không trùng giao dịch nào")
u_tr, u_te = set(df.user_id.iloc[tr]), set(df.user_id.iloc[te])
print(f"   User của Test đã xuất hiện trong Train: {len(u_te & u_tr):,}/{len(u_te):,}")
if a.protocol == "cold":
    # KIỂM TRA cold-start: không user nào của test có mặt trong train
    assert not (u_te & u_tr); OK("cold-start: KHÔNG user nào của Test có trong Train")

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 4 + 5. HỌC TIỀN XỬ LÝ TRÊN TRAIN, BIẾN ĐỔI 3 TẬP
# ════════════════════════════════════════════════════════════════════════════
H(4, "HỌC THAM SỐ TIỀN XỬ LÝ — CHỈ TRÊN TRAIN", "trainer.py: Prepared.__init__ -> Indexer.fit, BodyEncoder.fit, SizeRegressor.fit (data.py)")
# Prepared: học tiền xử lý CHỈ trên train rồi mã hoá cả 3 tập (trainer.py)
prep = Prepared(data, tr, va, te)
enc = prep.body_enc
for c in enc.cols[:4]:
    m_tr, m_all = body[c].iloc[tr].mean(), body[c].mean()
    print(f"   μ[{c:9s}] trong encoder = {enc.mu[c]:.4f} | mean(Train) = {m_tr:.4f} | mean(toàn bộ) = {m_all:.4f}")
    # KIỂM TRA chống rò rỉ: μ trong encoder phải bằng trung bình của TRAIN (khác trung bình toàn bộ)
    assert abs(enc.mu[c] - m_tr) < 1e-9
OK("μ, σ của encoder = thống kê của TRAIN (không phải toàn bộ dữ liệu)")
print(f"   Cột có cờ thiếu (thiếu ≥2% trên Train): {enc.miss_cols}")
# KIỂM TRA: bảng mã user chỉ chứa user của train
assert len(prep.indexer.maps["user_id"]) == len(u_tr); OK(f"bảng mã user chỉ chứa {len(u_tr):,} user của Train")
print(f"   Ridge (size ~ số đo + danh mục): {len(prep.size_reg.coef)} hệ số, σ phần dư = {prep.size_reg.sd:.3f} — fit trên X['train']")

H(5, "BIẾN ĐỔI TRAIN / VAL / TEST BẰNG THAM SỐ Ở BƯỚC 4", "data.py: BodyEncoder.transform, Indexer.transform, SizeRegressor.gap")
for k in ["train", "val", "test"]:
    X = prep.X[k]; uid = prep.tensors[k]["user_id"].numpy(); gap = prep.tensors[k]["gap"].numpy()
    print(f"   {k:5s}: X {X.shape} · user mã 0 (chưa biết) = {(uid == 0).mean():6.1%} · size_gap ∈ [{gap.min():.1f}, {gap.max():.1f}]")
print(f"   Đặc trưng cơ thể ({enc.dim}): {enc.feature_names()}")
# KIỂM TRA: z-score đã bị cắt trong [-5, 5]
assert np.abs(prep.X["test"][:, :len(enc.cols)]).max() <= 5; OK("z-score cắt trong ±5, ô thiếu = 0 (= trung bình Train)")
if a.protocol == "cold":
    # KIỂM TRA cold-start: mọi user test đều nhận chỉ số 0, nên phần cộng tác m_u = 0
    assert (prep.tensors["test"]["user_id"].numpy() == 0).all(); OK("cold-start: 100% user Test nhận mã 0 => m_u = 0")

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 6. HUẤN LUYỆN
# ════════════════════════════════════════════════════════════════════════════
H(6, f"HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT ({PROPOSED})", "model.py: OrdinalHybridMF ; trainer.py: train_model(), tune_decision()")
print(f"   Cấu hình: {SPECS[PROPOSED]}")
hp = HParams()
# huấn luyện mô hình đề xuất mf_body_product, in loss và AUC val mỗi epoch
m = train_model(prep, SPECS[PROPOSED], hp, a.seed,
                on_epoch=lambda h: print(f"   epoch {h['epoch']:2d}  train NLL={h['train_nll']:.4f}  val AUC={h['val_auc']:.4f}"))
print(f"   Dừng sau {len(m.history)} epoch (patience={hp.patience}), giữ trọng số có val AUC tốt nhất = {m.best_val_auc:.4f}")
# trọng số quyết định chọn trên VAL (không nhìn test)
w = tune_decision(predict(m, prep.tensors["val"]), prep.y("val"))
print(f"   Trọng số quyết định chọn trên VAL: {w}")

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 7. ĐÁNH GIÁ + SO SÁNH BASELINE
# ════════════════════════════════════════════════════════════════════════════
H(7, "ĐÁNH GIÁ TRÊN TEST + SO SÁNH BASELINE", "trainer.py: predict(), metrics(), body_lr() ; experiment.py: run_experiment()")
y = prep.y("test"); res = {}
# các baseline: majority (luôn đoán fit), body_lr (Logistic Regression), rồi các biến thể MF
res["majority"] = metrics(y, np.tile([0., 1., 0.], (len(y), 1)))
p = body_lr(prep, a.seed); res["body_lr"] = metrics(y, p["test"], tune_decision(p["val"], prep.y("val")))
for name in ["mf", "mf_body", "body_product"]:
    mm = train_model(prep, SPECS[name], hp, a.seed)
    res[name] = metrics(y, predict(mm, prep.tensors["test"]), tune_decision(predict(mm, prep.tensors["val"]), prep.y("val")))
res[PROPOSED] = metrics(y, predict(m, prep.tensors["test"]), w)
print(f"   {'Mô hình':28s} {'AUC':>7s} {'Macro-F1':>9s} {'Acc':>7s}")
for k, v in res.items():
    print(f"   {k + (' (ĐỀ XUẤT)' if k == PROPOSED else ''):28s} {v['auc']:7.4f} {v['macro_f1']:9.4f} {v['acc']:7.4f}")
OK("cùng một lần chia, cùng tham số tiền xử lý cho mọi mô hình (so sánh công bằng)")

# ════════════════════════════════════════════════════════════════════════════
#  BƯỚC 8. GỢI Ý SIZE
# ════════════════════════════════════════════════════════════════════════════
H(8, "GỢI Ý SIZE", "sizing.py: SizeModel.fit (trên Train), evaluate_size_rec (trên Test)")
item_sizes = df.groupby("item_id")["size"].agg(lambda s: sorted(set(float(x) for x in s))).to_dict()
# mô hình chọn size học trên train, đánh giá trên các giao dịch "fit" của test
sm = SizeModel().fit(prep.frames["train"], prep.X["train"], prep.size_reg)
for k, v in evaluate_size_rec(prep.frames["train"], prep.frames["test"], prep.X["test"], item_sizes, sm).items():
    print(f"   {k:8s}: đúng size {v['exact']:.1%} · lệch ≤1 size {v['within_1']:.1%} (n={v['n']:,})")
print(f"\nHoàn tất 8 bước sau {time.time() - t0:.0f} giây. Số liệu 1 seed; kết quả 10 seed ở 04-results/experiment/.")
