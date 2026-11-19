"""sizerec: Hybrid Matrix Factorization cho gợi ý size thời trang (ModCloth / RentTheRunway).

Bản đồ các file trong gói (theo thứ tự 8 bước của workflow):
  data.py        Bước 1–5: đọc, làm sạch, chia cold/warm, mã hoá (BodyEncoder, SizeRegressor, Indexer)
  model.py       Bước 6  : kiến trúc OrdinalHybridMF (MF + số đo cơ thể + sản phẩm, đầu ra thứ bậc)
  trainer.py     Bước 4–7: Prepared (chống rò rỉ), huấn luyện, dự đoán, chỉ số, lưu/nạp mô hình
  experiment.py  Bước 7  : thực nghiệm nhiều seed, so sánh baseline, kiểm định thống kê
  sizing.py      Bước 8  : chọn size cụ thể (SizeModel) và đánh giá đề xuất size
  recommend.py   Bước 8  : gợi ý size cho 1 khách + giải thích
  cli.py         Giao diện dòng lệnh `sizerec ...`
  interactive.py Menu tương tác khi gõ `sizerec` không tham số
  ui.py          In bảng / màu trên terminal

Chương trình bắt đầu ở cli.main():  python -m sizerec <lệnh>  ->  __main__.py  ->  cli.main()  ->  cmd_<lệnh>()
Mỗi file có MỤC LỤC + THẺ TÌM NHANH ở đầu; mỗi hàm có mục "Ý nghĩa" và "Hoạt động".
"""

__version__ = "1.0.0"
