"""Vẽ các hình cho báo cáo từ kết quả thực nghiệm.

Cách dùng (đứng trong 03-code):
    python scripts/make_figures.py ../04-results

Đọc   04-results/experiment/{modcloth,rtr}/*.csv và config.json
Ghi   04-results/figures/*.png (300 dpi) và 04-results/figures/data/*.csv (số liệu gốc của từng hình)
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = {"modcloth": "ModCloth", "rtr": "RentTheRunway"}
PROPOSED = "mf_body_product"
N_SEEDS = 10
MODEL_LABELS = {
    "majority": "Majority",
    "body_lr": "Body LR",
    "mf": "MF",
    "mf_body": "MF + Body",
    "body_product": "Body + Product",
    "mf_body_product": "MF + Body + Product (đề xuất)",
    "mf_body_product_nodrop": "Đề xuất, không ID dropout",
}
METHOD_LABELS = {"popular": "Size bán chạy nhất", "body": "Chỉ số đo cơ thể", "hybrid": "Số đo + lịch sử (hybrid)"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, GRID = "#b4b2a9", "#1f1e1d", "#6b6a63", "#e6e4dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(root, name):
    return {d: pd.read_csv(os.path.join(root, "experiment", d, name)) for d in DATASETS
            if os.path.exists(os.path.join(root, "experiment", d, name))}


def save(fig, out, name, data: pd.DataFrame):
    fig.savefig(os.path.join(out, name + ".png"))
    data.to_csv(os.path.join(out, "data", name + ".csv"), index=False)
    plt.close(fig)
    print("  ✓", name + ".png")


def fig_data_overview(root, out):
    rows = []
    for d, lab in DATASETS.items():
        p = os.path.join(root, "experiment", d, "config.json")
        if not os.path.exists(p):
            continue
        c = json.load(open(p, encoding="utf-8"))["data"]
        for k, v in c["fit_ratio"].items():
            rows.append({"dataset": lab, "kind": "fit_ratio", "name": k, "value": v})
        for k, v in c["body_missing"].items():
            rows.append({"dataset": lab, "kind": "body_missing", "name": k, "value": v})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    fr = df[df.kind == "fit_ratio"].pivot(index="name", columns="dataset", values="value").loc[["small", "fit", "large"]]
    fr.index = ["Chật (small)", "Vừa (fit)", "Rộng (large)"]
    fr.plot.bar(ax=axes[0], color=[BLUE, ORANGE][:fr.shape[1]], width=0.7, rot=0, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Phân bố nhãn fit", loc="left", fontweight="bold")
    axes[0].set_ylabel("Tỉ lệ giao dịch")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bm = df[df.kind == "body_missing"].pivot(index="name", columns="dataset", values="value")
    bm = bm.loc[bm.max(axis=1).sort_values().index]
    bm.plot.barh(ax=axes[1], color=[BLUE, ORANGE][:bm.shape[1]], width=0.75, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Tỉ lệ thiếu số đo cơ thể tự khai", loc="left", fontweight="bold")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_ylabel("")
    axes[0].legend(title=None)
    axes[1].legend(title=None, loc="lower right")
    save(fig, out, "fig1_data_overview", df)


def fig_auc_cold(summ, out):
    fig, axes = plt.subplots(1, len(summ), figsize=(6.5 * len(summ), 4.2), layout="constrained")
    axes = axes if len(summ) > 1 else [axes]
    rows = []
    for ax, (d, s) in zip(axes, summ.items()):
        t = s[(s.protocol == "cold") & (s.bucket == "all") & (s.model != "majority")].copy()
        t = t.sort_values("auc_mean")
        colors = [BLUE if m == PROPOSED else GRAY for m in t.model]
        y = range(len(t))
        ax.barh(list(y), t.auc_mean, xerr=t.auc_std, color=colors, height=0.6, error_kw=dict(ecolor=INK, lw=1, capsize=3))
        ax.set_yticks(list(y), [MODEL_LABELS.get(m, m) for m in t.model])
        lo = t.auc_mean.min() - 0.03
        ax.set_xlim(lo, (t.auc_mean + t.auc_std).max() + 0.015)
        for i, (v, sd) in enumerate(zip(t.auc_mean, t.auc_std)):
            ax.text(v + sd + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_title(f"{DATASETS[d]}: cold-start (khách mới)", loc="left", fontweight="bold")
        ax.set_xlabel(f"AUC macro (trung bình ± độ lệch chuẩn, {N_SEEDS} seed)")
        ax.grid(axis="y", visible=False)
        t["dataset"] = DATASETS[d]
        rows.append(t[["dataset", "model", "auc_mean", "auc_std", "macro_f1_mean", "acc_mean"]])
    save(fig, out, "fig2_auc_cold_start", pd.concat(rows))


def fig_auc_history(summ, out):
    models = ["mf", "mf_body", "body_product", PROPOSED]
    colors = dict(zip(models, [YELLOW, AQUA, ORANGE, BLUE]))
    buckets = ["0", "1", "2-4", "5+"]
    fig, axes = plt.subplots(1, len(summ), figsize=(6 * len(summ), 4.2), layout="constrained")
    axes = axes if len(summ) > 1 else [axes]
    rows = []
    for ax, (d, s) in zip(axes, summ.items()):
        t = s[(s.protocol == "warm") & (s.bucket.astype(str).isin(buckets)) & (s.model.isin(models))].copy()
        t["bucket"] = pd.Categorical(t.bucket.astype(str), buckets, ordered=True)
        for m in models:
            r = t[t.model == m].sort_values("bucket")
            ax.plot(range(len(r)), r.auc_mean, marker="o", ms=6, lw=2, color=colors[m], label=MODEL_LABELS[m],
                    zorder=3 if m == PROPOSED else 2)
        ax.set_xticks(range(len(buckets)), [f"{b}" for b in buckets])
        ax.set_xlabel("Số giao dịch của người dùng trong tập huấn luyện (warm)")
        ax.set_ylabel("AUC macro")
        ax.set_title(f"{DATASETS[d]}: AUC theo mức lịch sử", loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
        t["dataset"] = DATASETS[d]
        rows.append(t[["dataset", "bucket", "model", "auc_mean", "auc_std", "n"]])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
    save(fig, out, "fig3_auc_by_history", pd.concat(rows))


def fig_significance(sig, out):
    fig, axes = plt.subplots(1, len(sig), figsize=(7 * len(sig), 4.6), layout="constrained")
    axes = axes if len(sig) > 1 else [axes]
    rows = []
    for ax, (d, s) in zip(axes, sig.items()):
        t = s[(s.bucket.astype(str) == "all") & (s.model == PROPOSED) & (~s.vs.isin(["majority"]))].copy()
        t = t.sort_values(["protocol", "delta_auc"])
        labels = [f"{MODEL_LABELS.get(v, v)} · {p}" for v, p in zip(t.vs, t.protocol)]
        y = list(range(len(t)))
        for i, r in enumerate(t.itertuples()):
            c = BLUE if r.significant else GRAY
            ax.plot([r.ci95_low, r.ci95_high], [i, i], color=c, lw=2)
            ax.plot(r.delta_auc, i, "o", color=c, ms=7, mec="white", mew=1.5)
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.set_xlabel("ΔAUC của mô hình đề xuất so với mô hình so sánh (CI 95%)")
        ax.set_title(f"{DATASETS[d]}: kiểm định paired bootstrap", loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
        t["dataset"] = DATASETS[d]
        rows.append(t[["dataset", "protocol", "vs", "delta_auc", "ci95_low", "ci95_high", "p_bootstrap", "p_ttest",
                       "significant"]])
    fig.supxlabel("Xanh: khác biệt có ý nghĩa thống kê (p < 0.05). Xám: không có ý nghĩa.", fontsize=8.5,
                  color=MUTED, x=0.02, ha="left")
    save(fig, out, "fig4_significance_delta_auc", pd.concat(rows))


def fig_size_rec(sr, out):
    methods = ["popular", "body", "hybrid"]
    colors = dict(zip(methods, [GRAY, ORANGE, BLUE]))
    fig, axes = plt.subplots(1, len(sr), figsize=(6 * len(sr), 4.2), layout="constrained")
    axes = axes if len(sr) > 1 else [axes]
    rows = []
    for ax, (d, s) in zip(axes, sr.items()):
        t = s[s.protocol == "warm"] if (s.protocol == "warm").any() else s
        t = t.set_index("method").loc[[m for m in methods if m in set(t.method)]]
        x = range(2)
        w = 0.26
        for k, m in enumerate(t.index):
            vals = [t.loc[m, "exact_mean"], t.loc[m, "within_1_mean"]]
            pos = [i + (k - 1) * w for i in x]
            ax.bar(pos, vals, width=w - 0.02, color=colors[m], label=METHOD_LABELS[m])
            for p_, v in zip(pos, vals):
                ax.text(p_, v + 0.01, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(list(x), ["Trùng đúng size", "Lệch tối đa 1 size"])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{DATASETS[d]}: độ chính xác đề xuất size (warm)", loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
        tt = t.reset_index()
        tt["dataset"] = DATASETS[d]
        rows.append(tt)
    axes[0].legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fig5_size_recommendation", pd.concat(rows))


def fig_auc_boxplot(root, out):
    per = load(root, "metrics_per_seed.csv")
    if not per:
        return
    fig, axes = plt.subplots(len(per), 2, figsize=(13, 4.3 * len(per)), layout="constrained", squeeze=False)
    rows = []
    for r, (d, df) in enumerate(per.items()):
        for c, proto in enumerate(["cold", "warm"]):
            ax = axes[r][c]
            t = df[(df.protocol == proto) & (df.bucket == "all") & (~df.model.isin(["majority", "body_lr"]))]
            if t.empty:
                ax.set_visible(False)
                continue
            order = t.groupby("model").auc.mean().sort_values().index.tolist()
            data = [t[t.model == m].auc.values for m in order]
            bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none"))
            for patch, m in zip(bp["boxes"], order):
                patch.set_facecolor(BLUE if m == PROPOSED else GRAY)
                patch.set_edgecolor("white")
            for i, v in enumerate(data):
                ax.scatter(v, [i + 1] * len(v), s=9, color=INK, alpha=0.55, zorder=3)
            ax.set_yticks(range(1, len(order) + 1), [MODEL_LABELS.get(m, m) for m in order])
            ax.set_title(f"{DATASETS[d]}: { {'cold': 'cold-start (khách mới)', 'warm': 'warm (chia ngẫu nhiên)'}[proto] }", loc="left", fontweight="bold")
            ax.set_xlabel("AUC macro trên tập test, mỗi điểm là một lần lặp")
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            n = t.seed.nunique()
            rows.append(t[["dataset", "protocol", "seed", "model", "auc", "macro_f1"]])
    fig.suptitle("Phân bố AUC qua {n} lần lặp".format(n=n), x=0.01, ha="left", fontweight="bold")
    save(fig, out, "fig6_auc_boxplot", pd.concat(rows))


def main(root):
    out = os.path.join(root, "figures")
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    summ, sig, sr = load(root, "summary.csv"), load(root, "significance.csv"), load(root, "size_rec_summary.csv")
    global N_SEEDS
    p = os.path.join(root, "experiment", next(iter(summ)), "config.json") if summ else ""
    if p and os.path.exists(p):
        N_SEEDS = len(json.load(open(p, encoding="utf-8")).get("seeds", [])) or N_SEEDS
    if not summ:
        sys.exit(f"Không tìm thấy kết quả trong {root}/experiment/. Hãy chạy thực nghiệm trước.")
    print(f"Vẽ hình vào {out}")
    fig_data_overview(root, out)
    fig_auc_cold(summ, out)
    fig_auc_history(summ, out)
    fig_significance(sig, out)
    fig_size_rec(sr, out)
    fig_auc_boxplot(root, out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "04-results"))
