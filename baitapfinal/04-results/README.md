# 04-results: Kết quả thực nghiệm

Toàn bộ nội dung trong thư mục này được tạo bởi `03-code/scripts/run_all.sh` (Windows: `run_all.bat`). Bản sao các script đã dùng nằm trong `scripts/`.

| Đường dẫn | Nội dung |
| --- | --- |
| `results_summary.xlsx` | Mọi bảng kết quả trong một file Excel (sheet README giải thích từng sheet) |
| `experiment/<ds>/summary.csv` | AUC, Macro-F1, Acc (trung bình ± độ lệch chuẩn 10 seed) theo giao thức, nhóm lịch sử, mô hình |
| `experiment/<ds>/metrics_per_seed.csv` | Số liệu từng seed |
| `experiment/<ds>/significance.csv` | ΔAUC của mô hình đề xuất so với từng mô hình: CI95, p-value bootstrap, p-value t-test |
| `experiment/<ds>/size_rec_summary.csv`, `size_rec_per_seed.csv` | Độ chính xác đề xuất size |
| `experiment/<ds>/report.md` | Báo cáo dạng bảng Markdown |
| `experiment/<ds>/config.json` | Cấu hình chạy và thống kê dữ liệu để tái lập |
| `figures/*.png` | Hình (nhãn tiếng Việt, 300 dpi) |
| `figures/en/*.png` | Hình nhãn tiếng Anh, dùng trong báo cáo LaTeX |
| `figures/data/*.csv` | Số liệu gốc dùng để vẽ từng hình |
| `models/*.pt` | Mô hình đề xuất huấn luyện trên toàn bộ dữ liệu, dùng cho `sizerec recommend` |
| `logs/*.txt` | Log đầy đủ của từng lệnh: thống kê dữ liệu, kiểm thử, thực nghiệm, huấn luyện, ví dụ gợi ý size |

`<ds>` là `modcloth` hoặc `rtr` (RentTheRunway).
