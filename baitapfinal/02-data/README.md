# 02-data: Dữ liệu thực nghiệm

| File | Nguồn | Quy mô sau làm sạch |
| --- | --- | --- |
| `modcloth_final_data.json.gz` | ModCloth (Misra et al., RecSys 2018) | 82.790 giao dịch · 47.958 người dùng · 1.378 sản phẩm |
| `renttherunway_final_data.json.gz` | RentTheRunway (Misra et al., RecSys 2018) | 192.544 giao dịch · 105.571 người dùng · 5.850 sản phẩm |

- Nguồn gốc: https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit
- Trích dẫn: R. Misra, M. Wan, J. McAuley. *Decomposing fit semantics for product size recommendation in metric spaces.* RecSys 2018.
- Định dạng: mỗi dòng là một JSON (JSON Lines), nén gzip. Chương trình đọc trực tiếp file `.gz`, không cần giải nén.
- Nhãn: `fit` ∈ {small, fit, large}, khoảng 70% là `fit`.
- Các trường chỉ có sau khi mua (`review_text`, `review_summary`, `rating`, `quality`, `length`) bị loại để tránh rò rỉ dữ liệu.
- Thống kê chi tiết: xem `04-results/logs/data_modcloth.txt` và `data_rtr.txt`, hoặc chạy `sizerec data <file>`.

Không đưa bản `.json` đã giải nén lên GitHub, vì file RentTheRunway giải nén nặng khoảng 123MB, vượt giới hạn 100MB/file.
