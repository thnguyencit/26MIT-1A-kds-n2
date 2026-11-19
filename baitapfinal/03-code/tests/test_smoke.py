"""Kiểm thử nhanh toàn bộ pipeline trên dữ liệu giả lập nhỏ (không cần bộ dữ liệu thật).

Chạy:  python -m pytest -q tests      hoặc      python tests/test_smoke.py
"""

import gzip
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sizerec.cli import main  # noqa: E402


def make_synthetic(path, n=3000, seed=0):
    """Tạo dữ liệu giả lập dạng RTR: size "đúng" phụ thuộc cân nặng, nhãn fit theo độ lệch size."""
    rng = random.Random(seed)
    items = {str(1000 + i): rng.choice(["dress", "gown", "top"]) for i in range(40)}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for _ in range(n):
            u = rng.randint(1, 600)
            h = 60 + (u % 12)
            w = 100 + (u % 90)
            true = (w - 100) / 6
            item = rng.choice(list(items))
            size = max(0, round(true + rng.gauss(0, 3)))
            gap = size - true - (int(item) % 3 - 1)
            fit = "large" if gap > 2 else "small" if gap < -2 else "fit"
            f.write(json.dumps({"user_id": str(u), "item_id": item, "size": size, "fit": fit,
                                "category": items[item], "height": f"{h // 12}' {h % 12}\"",
                                "weight": f"{w}lbs", "bust size": "34b", "body type": "hourglass",
                                "rented for": "party", "age": "30"}) + "\n")


def test_pipeline():
    """Chạy lần lượt các lệnh data, experiment --quick, train, inspect, recommend; tất cả phải trả mã 0."""
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "renttherunway_synthetic.json.gz")
        make_synthetic(data)
        assert main(["-q", "data", data]) == 0
        assert main(["-q", "experiment", data, "--quick", "--epochs", "3", "-o", os.path.join(d, "res")]) == 0
        assert os.path.isfile(os.path.join(d, "res", "summary.csv"))
        assert os.path.isfile(os.path.join(d, "res", "significance.csv"))
        model = os.path.join(d, "m.pt")
        assert main(["-q", "train", data, "-o", model, "--epochs", "3"]) == 0
        assert main(["inspect", model]) == 0
        assert main(["recommend", model, "--item", "1001", "--height", "5'5\"", "--weight", "140lbs",
                     "--explain"]) == 0
        assert main(["recommend", model, "--item", "1001", "--user", "5", "--json"]) == 0


if __name__ == "__main__":
    test_pipeline()
    print("OK")
