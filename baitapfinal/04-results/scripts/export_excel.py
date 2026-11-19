"""Gộp toàn bộ kết quả thực nghiệm vào một file Excel.

Cách dùng (đứng trong 03-code):
    python scripts/export_excel.py ../04-results

Ghi 04-results/results_summary.xlsx, mỗi sheet là một bảng:
  README, data_stats, auc_cold, auc_warm_history, summary_<ds>, significance_<ds>, size_rec_<ds>, per_seed_<ds>
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

DATASETS = {"modcloth": "ModCloth", "rtr": "RentTheRunway"}
ORDER = ["majority", "body_lr", "mf", "mf_body", "body_product", "mf_body_product", "mf_body_product_nodrop"]


def read(root, d, name):
    p = os.path.join(root, "experiment", d, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main(root):
    out = os.path.join(root, "results_summary.xlsx")
    sheets = {}
    stats = []
    for d, lab in DATASETS.items():
        p = os.path.join(root, "experiment", d, "config.json")
        if os.path.exists(p):
            c = json.load(open(p, encoding="utf-8"))
            dd = c["data"]
            stats.append({"dataset": lab, "interactions": dd["interactions"], "users": dd["users"],
                          "items": dd["items"], "item_sizes": dd["item_sizes"], "categories": dd["categories"],
                          "users_with_1_interaction": dd["users_with_1_interaction"],
                          **{f"fit_{k}": v for k, v in dd["fit_ratio"].items()},
                          **{f"missing_{k}": v for k, v in dd["body_missing"].items()},
                          "seeds": str(c["seeds"]), "hparams": json.dumps(c["hparams"]), "finished": c["finished"]})
    sheets["data_stats"] = pd.DataFrame(stats)

    cold, hist = [], []
    for d, lab in DATASETS.items():
        s = read(root, d, "summary.csv")
        if s is None:
            continue
        s["model"] = pd.Categorical(s["model"], ORDER, ordered=True)
        c = s[(s.protocol == "cold") & (s.bucket == "all")].sort_values("model")
        cold.append(c.assign(dataset=lab)[["dataset", "model", "auc_mean", "auc_std", "macro_f1_mean", "acc_mean", "n"]])
        w = s[s.protocol == "warm"].pivot_table(index="model", columns="bucket", values="auc_mean", observed=True)
        w = w[[b for b in ["all", "0", "1", "2-4", "5+"] if b in w.columns]].reset_index()
        hist.append(w.assign(dataset=lab))
        sheets[f"summary_{d}"] = s.sort_values(["protocol", "bucket", "model"])
        for name, key in [("significance.csv", "significance"), ("size_rec_summary.csv", "size_rec"),
                          ("metrics_per_seed.csv", "per_seed")]:
            t = read(root, d, name)
            if t is not None:
                sheets[f"{key}_{d}"] = t
    sheets["auc_cold"] = pd.concat(cold) if cold else pd.DataFrame()
    sheets["auc_warm_history"] = pd.concat(hist) if hist else pd.DataFrame()

    readme = pd.DataFrame({"sheet": ["data_stats", "auc_cold", "auc_warm_history", "summary_<ds>", "significance_<ds>",
                                     "size_rec_<ds>", "per_seed_<ds>"],
                           "nội dung": ["Thống kê dữ liệu, tỉ lệ nhãn, tỉ lệ thiếu số đo, cấu hình chạy",
                                        "AUC / Macro-F1 / Acc giao thức cold-start (khách mới), trung bình 10 seed",
                                        "AUC giao thức warm theo số giao dịch lịch sử của người dùng",
                                        "Bảng tổng hợp đầy đủ (summary.csv)",
                                        "ΔAUC mô hình đề xuất so với từng mô hình, CI95, p-value bootstrap và t-test",
                                        "Độ chính xác đề xuất size: trùng đúng size, lệch tối đa 1 size",
                                        "Số liệu chi tiết từng seed"]})
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for name in ["data_stats", "auc_cold", "auc_warm_history"] + sorted(k for k in sheets if "_" in k and k not in
                                                                             ("data_stats", "auc_cold", "auc_warm_history")):
            sheets[name].to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(40, max(10, max(len(str(c.value or "")) for c in col) + 2))
            ws.freeze_panes = "A2"
    print(f"✓ Đã ghi {out} ({len(sheets) + 1} sheet)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "04-results"))
