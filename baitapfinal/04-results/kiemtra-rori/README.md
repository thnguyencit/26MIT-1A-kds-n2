# kiemtra-rori: kiểm tra rò rỉ dữ liệu

Thư mục này chứa các script và log mà nhóm dùng để kiểm tra rò rỉ dữ liệu (data leakage) giữa các tập train, val và test của hai bộ dữ liệu ModCloth và RentTheRunway (RTR).

| File | Nội dung |
| --- | --- |
| `kiemtra_rori.py` | Chạy hàm `split()` của sizerec trên dữ liệu gốc với 10 seed. Script kiểm tra: không có dòng nào thuộc hai tập cùng lúc; số khách xuất hiện ở cả train và test; tỉ lệ dòng test có cùng (user, item, size) với một dòng train; và các cột rò rỉ (lời nhận xét, điểm chấm) đã bị loại. |
| `kiemtra_dongtrung.py` | Đếm các dòng trùng nhau hoàn toàn trong file JSON gốc. |
| `chaylai_warm_bodongtrung.py` | Chạy lại giao thức warm (10 seed; các mô hình mf, mf_body, body_product, mf_body_product) hai lần: trên dữ liệu gốc và sau khi bỏ dòng trùng. Sau đó so sánh AUC giữa hai lần chạy. |
| `log_kiemtra_rori.txt` | Kết quả chạy `kiemtra_rori.py` |
| `log_kiemtra_dongtrung.txt` | Kết quả chạy `kiemtra_dongtrung.py` |
| `log_chaylai_modcloth.txt`, `log_chaylai_rtr.txt` | Kết quả chạy `chaylai_warm_bodongtrung.py` (bảng SIG, SUMMARY và bảng chênh lệch orig/dedup) |

## Cách chạy

Chạy từ thư mục `03-code`:

    python -I ../04-results/kiemtra-rori/kiemtra_rori.py . ../02-data/modcloth_final_data.json.gz ../02-data/renttherunway_final_data.json.gz
    python -I ../04-results/kiemtra-rori/kiemtra_dongtrung.py ../02-data/modcloth_final_data.json.gz ../02-data/renttherunway_final_data.json.gz
    python -I ../04-results/kiemtra-rori/chaylai_warm_bodongtrung.py . ../02-data/modcloth_final_data.json.gz 1,2,3,4,5,6,7,8,9,10 4 mc.pkl

Các tham số của `chaylai_warm_bodongtrung.py` lần lượt là: thư mục code, file dữ liệu, danh sách seed, số luồng CPU, file cache. Kết quả chạy lại không ghi đè các kết quả khác trong `04-results`.

## Kết quả

- **Giao thức cold (khách mới) không bị rò rỉ.** Với cả hai bộ dữ liệu, không có khách nào xuất hiện ở cả train và test, hay ở cả train và val. Tỉ lệ dòng test có cùng (user, item, size) với train là 0,00%. Không có cột rò rỉ nào được đưa vào mô hình.
- **Giao thức warm có một ít dòng trùng.** ModCloth có 754 dòng nằm trong các cặp trùng nhau hoàn toàn (377 cặp, trên tổng 82.790 dòng). RTR có 378 dòng như vậy (189 cặp, trên tổng 192.544 dòng). Ở warm, trung bình 0,77% (ModCloth) và 0,21% (RTR) dòng test có cùng (user, item, size) với một dòng train.
- **Bỏ dòng trùng rồi chạy lại warm thì kết quả gần như không đổi.** Trên toàn bộ tập test, AUC của mô hình đề xuất `mf_body_product` thay đổi +0,0002 (ModCloth) và +0,0005 (RTR), tức là dưới 0,001. Các mô hình so sánh và từng nhóm lịch sử thay đổi tối đa khoảng 0,003. Mô hình đề xuất vẫn tốt hơn các mô hình so sánh một cách có ý nghĩa thống kê (p < 0,01).
