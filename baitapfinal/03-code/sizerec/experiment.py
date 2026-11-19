"""

experiment.py — THỰC NGHIỆM SO SÁNH MÔ HÌNH + KIỂM ĐỊNH THỐNG KÊ  (bước 7 của workflow)

MỤC LỤC
    Phần A. Chạy thực nghiệm ... run_experiment
    Phần B. Tổng hợp ........... summarize
    Phần C. Kiểm định .......... fast_auc, significance
    Phần D. Ghi kết quả ........ summarize_size, write_outputs, markdown_report

THẺ TÌM NHANH  (Ctrl+F)
    [RÒ RỈ]      chỗ chống rò rỉ dữ liệu
    [KIỂM ĐỊNH]  paired t-test và paired bootstrap
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from .data import FitData, history_bucket, split
from .model import PROPOSED, SPECS
from .sizing import SizeModel, evaluate_size_rec
from .trainer import HParams, Prepared, auc, body_lr, metrics, predict, train_model, tune_decision

# 2 baseline không phải MF + 5 biến thể MF
DEFAULT_MODELS = ["majority", "body_lr", "mf", "mf_body", "body_product", "mf_body_product", "mf_body_product_nodrop"]
# Nhóm số lịch sử của user (dùng khi phân tích warm)
BUCKETS = ["0", "1", "2-4", "5+"]


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. CHẠY THỰC NGHIỆM
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(data: FitData, protocols, seeds, models, hp: HParams, log=print, n_boot: int = 200):
    """BƯỚC 7. Chạy toàn bộ thực nghiệm (hàm chính của lệnh `sizerec experiment`)."""
    rows, store, size_rows = [], {}, []
    item_sizes = data.df.groupby("item_id")["size"].agg(lambda s: sorted(set(float(x) for x in s))).to_dict()

    for protocol in protocols:
        for seed in seeds:
            t0 = time.time()

            # (1) Chia dữ liệu + [RÒ RỈ] học tiền xử lý chỉ trên train, dùng chung cho mọi mô hình
            tr, va, te = split(data.df, protocol, seed)
            prep = Prepared(data, tr, va, te)
            y_va, y_te = prep.y("val"), prep.y("test")

            # Nhóm lịch sử của từng dòng test (user không có trong train -> nhóm "0")
            hist = prep.frames["train"]["user_id"].value_counts()
            bucket = history_bucket(prep.frames["test"]["user_id"].map(hist).fillna(0).astype(int).values)

            # (2) Từng mô hình: xác suất test + trọng số quyết định chọn trên VAL
            preds, weights, extra = {}, {}, {}
            for name in models:
                t1 = time.time()
                if name == "majority":
                    # luôn đoán "vừa" -> mốc dưới, AUC = 0.5
                    preds[name], weights[name] = np.tile([0.0, 1.0, 0.0], (len(y_te), 1)), np.ones(3)
                elif name == "body_lr":
                    p = body_lr(prep, seed)
                    preds[name], weights[name] = p["test"], tune_decision(p["val"], y_va)
                else:
                    m = train_model(prep, SPECS[name], hp, seed)
                    preds[name] = predict(m, prep.tensors["test"])
                    weights[name] = tune_decision(predict(m, prep.tensors["val"]), y_va)
                    extra[name] = {"val_auc": m.best_val_auc, "epochs": len(m.history)}
                log(f"[{data.name}|{protocol}|seed {seed}] {name:24s} test AUC={auc(y_te, preds[name]):.4f} "
                    f"({time.time() - t1:.1f}s)")

            # (3) Chỉ số trên toàn bộ test; warm thì tách thêm theo nhóm lịch sử (nhóm >= 50 dòng)
            for name, p in preds.items():
                base = {"dataset": data.name, "protocol": protocol, "seed": seed, "model": name}
                rows.append({**base, "bucket": "all", "n": len(y_te), **metrics(y_te, p, weights[name]),
                             **extra.get(name, {})})
                if protocol == "warm":
                    for bk in BUCKETS:
                        mk = bucket == bk
                        if mk.sum() >= 50:
                            rows.append({**base, "bucket": bk, "n": int(mk.sum()),
                                         **metrics(y_te[mk], p[mk], weights[name])})
            store[(protocol, seed)] = (y_te, preds, bucket)

            # (4) Đánh giá đề xuất size (SizeModel cũng học trên train)
            sm = SizeModel().fit(prep.frames["train"], prep.X["train"], prep.size_reg)
            for meth, v in evaluate_size_rec(prep.frames["train"], prep.frames["test"], prep.X["test"],
                                             item_sizes, sm).items():
                size_rows.append({"dataset": data.name, "protocol": protocol, "seed": seed, "method": meth, **v})
            log(f"[{data.name}|{protocol}|seed {seed}] xong sau {time.time() - t0:.0f}s")

    # (5) [KIỂM ĐỊNH] mô hình đề xuất so với từng mô hình còn lại
    res = pd.DataFrame(rows)
    sig = significance(res, store, models, n_boot=n_boot, ref=PROPOSED if PROPOSED in models else models[-1])
    return res, sig, pd.DataFrame(size_rows)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. TỔNG HỢP QUA CÁC SEED
# ════════════════════════════════════════════════════════════════════════════

def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Gộp kết quả các seed: trung bình ± độ lệch chuẩn của AUC, macro-F1, acc cho từng mô hình."""
    g = res.groupby(["dataset", "protocol", "bucket", "model"], sort=False)
    out = g[["auc", "macro_f1", "acc"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g["n"].mean().round().astype(int)
    out["seeds"] = g["seed"].nunique()
    return out.reset_index()


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN C. KIỂM ĐỊNH THỐNG KÊ
# ════════════════════════════════════════════════════════════════════════════

def fast_auc(y, p) -> float:
    """AUC macro one-vs-rest tính nhanh bằng thống kê Mann-Whitney (dùng trong bootstrap)."""
    out = []
    for k in range(3):
        pos = y == k
        n1 = pos.sum()
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = stats.rankdata(p[:, k])
        out.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else float("nan")


def significance(res, store, models, ref, n_boot=200, rng_seed=0):
    """[KIỂM ĐỊNH] So mô hình đề xuất (ref) với từng mô hình khác."""
    rows = []
    rng = np.random.RandomState(rng_seed)
    for (dataset, protocol, bucket), grp in res.groupby(["dataset", "protocol", "bucket"]):
        # Bảng AUC: mỗi dòng 1 seed, mỗi cột 1 mô hình
        piv = grp.pivot_table(index="seed", columns="model", values="auc")
        if ref not in piv:
            continue
        keys = [k for k in store if k[0] == protocol]

        for other in models:
            if other == ref or other not in piv:
                continue

            # (1) Paired t-test trên ΔAUC theo seed
            d = (piv[ref] - piv[other]).dropna()
            t_p = float(stats.ttest_rel(piv.loc[d.index, ref], piv.loc[d.index, other]).pvalue) \
                if len(d) > 1 and d.std() > 0 else float("nan")

            # (2) Paired bootstrap
            boot = np.zeros(n_boot)
            used = 0
            for k in keys:
                y, preds, bk = store[k]
                mask = np.ones(len(y), bool) if bucket == "all" else (bk == bucket)
                if mask.sum() < 50:
                    continue
                yy, pa, pb = y[mask], preds[ref][mask], preds[other][mask]
                for b in range(n_boot):
                    # cùng một bộ chỉ số ix cho cả 2 mô hình -> "ghép cặp"
                    ix = rng.randint(0, len(yy), len(yy))
                    boot[b] += fast_auc(yy[ix], pa[ix]) - fast_auc(yy[ix], pb[ix])
                used += 1
            if used == 0:
                continue
            boot /= used

            # (3) p hai phía = 2 x tỉ lệ lần bootstrap ΔAUC "đổi dấu" ; CI95 = phân vị 2.5% và 97.5%
            p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
            rows.append({"dataset": dataset, "protocol": protocol, "bucket": bucket,
                         "model": ref, "vs": other, "delta_auc": float(d.mean()), "delta_std": float(d.std()),
                         "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                         "p_bootstrap": p_boot, "n_boot": n_boot, "p_ttest": t_p, "seeds": int(len(d)),
                         "significant": bool(p_boot < 0.05)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN D. GHI KẾT QUẢ
# ════════════════════════════════════════════════════════════════════════════

def summarize_size(size_df: pd.DataFrame) -> pd.DataFrame:
    """Gộp kết quả đề xuất size qua các seed: trung bình ± độ lệch chuẩn của exact và within_1."""
    g = size_df.groupby(["dataset", "protocol", "method"], sort=False)
    out = g[["exact", "within_1"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g["n"].mean().round().astype(int)
    return out.reset_index()


def write_outputs(outdir, data_stats, res, sig, config: dict, size_df: pd.DataFrame | None = None):
    """Ghi toàn bộ kết quả thực nghiệm ra các file trong thư mục outdir."""
    # (1) Tạo thư mục kết quả
    os.makedirs(outdir, exist_ok=True)
    # (2) Chỉ số từng seed + bảng trung bình ± std
    res.to_csv(os.path.join(outdir, "metrics_per_seed.csv"), index=False)
    summ = summarize(res)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    # (3) Bảng kiểm định + cấu hình chạy
    sig.to_csv(os.path.join(outdir, "significance.csv"), index=False)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "data": data_stats}, f, ensure_ascii=False, indent=2)
    # (4) Kết quả đề xuất size (nếu có)
    size_summ = summarize_size(size_df) if size_df is not None and len(size_df) else None
    if size_summ is not None:
        size_df.to_csv(os.path.join(outdir, "size_rec_per_seed.csv"), index=False)
        size_summ.to_csv(os.path.join(outdir, "size_rec_summary.csv"), index=False)
    # (5) Báo cáo tóm tắt report.md
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(summ, sig, config, size_summ))
    return summ


def markdown_report(summ: pd.DataFrame, sig: pd.DataFrame, config: dict, size_summ=None) -> str:
    """Tạo nội dung report.md gồm 3 bảng: (1) AUC theo giao thức/nhóm ; (2) kiểm định ; (3) đề xuất size."""
    L = [f"# Báo cáo thực nghiệm sizerec ({config.get('dataset')})", "",
         f"- Giao thức: {', '.join(config['protocols'])}; seeds: {config['seeds']}",
         f"- Siêu tham số: {config['hparams']}", "", "## AUC macro (trung bình ± độ lệch chuẩn qua seed)", ""]

    # Bảng 1: AUC theo giao thức / nhóm lịch sử
    for (proto, bucket), g in summ.groupby(["protocol", "bucket"], sort=False):
        L += [f"### {proto} / lịch sử user = {bucket} (n≈{int(g['n'].iloc[0])})", "",
              "| Mô hình | AUC | Macro-F1 | Acc |", "| --- | --- | --- | --- |"]
        for _, r in g.iterrows():
            L.append(f"| {r['model']} | {r['auc_mean']:.4f} ± {np.nan_to_num(r['auc_std']):.4f} | "
                     f"{r['macro_f1_mean']:.4f} | {r['acc_mean']:.4f} |")
        L.append("")

    # Bảng 2: kiểm định
    if len(sig):
        L += ["## Kiểm định: mô hình đề xuất so với từng mô hình", "",
              "| Giao thức | Nhóm | So với | ΔAUC | CI95 bootstrap | p (bootstrap) | p (t-test) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in sig.iterrows():
            L.append(f"| {r['protocol']} | {r['bucket']} | {r['vs']} | {r['delta_auc']:+.4f} | "
                     f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['p_bootstrap']:.3f} | {r['p_ttest']:.3f} |")

    # Bảng 3: đề xuất size
    if size_summ is not None:
        L += ["", "## Đề xuất size trên các giao dịch 'fit' của tập test", "",
              "| Giao thức | Phương pháp | Trùng đúng size | Lệch ≤ 1 size | n |", "| --- | --- | --- | --- | --- |"]
        for _, r in size_summ.iterrows():
            L.append(f"| {r['protocol']} | {r['method']} | {r['exact_mean']:.1%} | {r['within_1_mean']:.1%} | {r['n']} |")
    return "\n".join(L) + "\n"
