"""Cho phép chạy `python -m sizerec ...` (chuyển sang cli.main)."""
# sys: dùng sys.exit để trả mã thoát cho hệ điều hành
import sys

# Lấy hàm main (điểm vào chương trình) từ cli.py
from .cli import main

# Chạy main() rồi thoát với mã nó trả về (0 = thành công)
sys.exit(main())
