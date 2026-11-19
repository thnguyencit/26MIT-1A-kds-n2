# 08-baocao-homnay: bằng chứng kiểm tra rò rỉ dữ liệu (15/11)

Báo cáo chính nằm trong tài liệu "Báo cáo: rà soát code, rò rỉ dữ liệu và viết lại các phần bài báo" (xuất Word bằng menu Export của tài liệu).

| File | Nội dung |
| --- | --- |
| `kiemtra_rori.py` | Chạy `split()` của sizerec trên dữ liệu gốc, 10 seed: đếm khách chung giữa train/test (cold), dòng test có bản trùng trong train (warm), trung vị size |
| `kiemtra_dongtrung.py` | Đếm dòng trùng y hệt trong file JSON gốc |
| `chaylai_warm_bodongtrung.py` | Chạy lại giao thức warm (10 seed, mf, mf_body, body_product, mf_body_product) trên dữ liệu gốc và sau khi bỏ dòng trùng |
| `log_chaylai_modcloth.txt`, `log_chaylai_rtr.txt` | Kết quả hai lần chạy (bảng SIG, SUMMARY, diff) |

Cách chạy (từ thư mục 03-code, đường dẫn dữ liệu tuỳ máy):

    python -I ../08-baocao-homnay/kiemtra_rori.py . ../02-data/modcloth_final_data.json.gz ../02-data/renttherunway_final_data.json.gz
    python -I ../08-baocao-homnay/chaylai_warm_bodongtrung.py . ../02-data/modcloth_final_data.json.gz 1,2,3,4,5,6,7,8,9,10 4 mc.pkl

Kết quả chạy lại KHÔNG ghi đè `04-results`.
