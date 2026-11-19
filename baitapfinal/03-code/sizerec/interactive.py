"""

interactive.py — MENU TƯƠNG TÁC  (mở khi gõ `sizerec` không tham số, hoặc `sizerec shell`)

MỤC LỤC
    Phần A. Cấu hình ........... DATA_DIRS, MODEL_DIRS, BODY_TYPES, OCCASIONS, HP_DEFAULTS
    Phần B. Hàm hỏi đáp ........ ask, confirm, pick, _run
    Phần C. Tìm file ........... find_datasets, choose_dataset, find_models, choose_model
    Phần D. Chức năng menu ..... action_recommend, action_train, action_experiment, ...
    Phần E. Vòng lặp menu ...... MENU, run_shell

Lưu ý: mọi chức năng đều gọi lại đúng hàm cmd_* của cli.py, nên kết quả giống hệt khi gõ lệnh.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from . import __version__
from .ui import HAS_RICH, UI

if HAS_RICH:
    # có thư viện rich thì dùng ô hỏi đẹp hơn; không có thì dùng input() thường
    from rich.prompt import Confirm, Prompt

# ════════════════════════════════════════════════════════════════════════════
#  PHẦN A. CẤU HÌNH
# ════════════════════════════════════════════════════════════════════════════

# các thư mục tự động tìm file dữ liệu (đường dẫn tương đối so với thư mục đang chạy)
DATA_DIRS = [os.path.join("..", "02-data"), "dataset", "data", os.path.join("..", "..", "dataset", "dataset"), os.path.join("..", "dataset"),
             os.path.join("..", "..", "dataset")]
# các dáng người có trong RentTheRunway (để chọn nhanh)
BODY_TYPES = ["hourglass", "athletic", "pear", "petite", "full bust", "straight & narrow", "apple"]
# các dịp thuê (rented for) của RentTheRunway
OCCASIONS = ["wedding", "formal affair", "party", "everyday", "work", "date", "vacation", "other"]
# các thư mục tự động tìm file mô hình .pt
MODEL_DIRS = ["models", os.path.join("..", "04-results", "models")]


def find_models():
    """Tìm các file mô hình .pt trong các thư mục MODEL_DIRS."""
    return sorted({p for d in MODEL_DIRS for p in glob.glob(os.path.join(d, "*.pt"))})


# siêu tham số mặc định, giống cli._add_hparams
HP_DEFAULTS = dict(dim=16, lr=3e-3, l2=1e-2, batch=1024, epochs=30, patience=3, id_drop=0.1)


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN B. HÀM HỎI ĐÁP
# ════════════════════════════════════════════════════════════════════════════
def ask(text, default=None, choices=None, allow_empty=False):
    """Hỏi người dùng 1 giá trị; lặp lại tới khi hợp lệ."""
    while True:
        if HAS_RICH:
            v = Prompt.ask(f"[bold]{text}[/bold]", default=default, choices=choices, show_choices=bool(choices))
        else:
            hint = f" [{default}]" if default is not None else ""
            v = input(f"{text}{hint}: ").strip() or default
            if choices and v not in choices:
                print(f"  Chọn một trong: {', '.join(choices)}")
                continue
        v = (v or "").strip()
        if v or allow_empty:
            return v or None


def confirm(text, default=True):
    """Hỏi Có/Không; trả về True/False."""
    if HAS_RICH:
        return Confirm.ask(f"[bold]{text}[/bold]", default=default)
    v = input(f"{text} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not v else v in ("y", "yes", "c", "co", "có")


def pick(ui: UI, title, options, default=1):
    """In danh sách lựa chọn đánh số 1, 2, 3... và trả về giá trị của mục được chọn."""
    ui.table(["#", title], [(i, lab) for i, (lab, _) in enumerate(options, 1)], justify=["right", "left"])
    i = ask("Chọn số", default=str(default), choices=[str(i) for i in range(1, len(options) + 1)])
    return options[int(i) - 1][1]


def _hp_ns(**kw):
    """Tạo Namespace giống kết quả argparse, có sẵn siêu tham số mặc định,"""
    return argparse.Namespace(**{**HP_DEFAULTS, **kw})


def _run(ui, fn, ns):
    """Chạy 1 hàm cmd_* mà không làm thoát menu khi có lỗi (bắt SystemExit, Ctrl+C, lỗi khác)."""
    try:
        fn(ns, ui)
        return True
    except SystemExit:
        return False
    except KeyboardInterrupt:
        ui.warn("Đã dừng thao tác hiện tại.")
        return False
    except Exception as ex:  # pragma: no cover
        ui.error(f"{type(ex).__name__}: {ex}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN C. TÌM FILE DỮ LIỆU / MÔ HÌNH
# ════════════════════════════════════════════════════════════════════════════
def find_datasets():
    """Tìm file dữ liệu .json / .json.gz trong các thư mục DATA_DIRS (bỏ trùng, ưu tiên bản .gz)."""
    out = []
    for d in DATA_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.json.gz")) + glob.glob(os.path.join(d, "*.json"))):
            if os.path.isfile(p) and os.path.abspath(p) not in [os.path.abspath(x) for x in out]:
                out.append(p)
    # (nếu có cả abc.json và abc.json.gz thì chỉ giữ bản .gz)
    # ưu tiên .gz, bỏ bản .json trùng tên
    names = {os.path.basename(p).replace(".gz", "") for p in out if p.endswith(".gz")}
    return [p for p in out if p.endswith(".gz") or os.path.basename(p) not in names]


def choose_dataset(ui):
    """Cho người dùng chọn bộ dữ liệu từ danh sách tìm được, hoặc nhập đường dẫn khác."""
    found = find_datasets()
    opts = [(f"{os.path.basename(p)}  ({os.path.dirname(p) or '.'})", p) for p in found]
    opts.append(("Nhập đường dẫn khác...", None))
    p = pick(ui, "Bộ dữ liệu", opts)
    while p is None or not os.path.isfile(p):
        if p is not None:
            ui.error(f"Không tìm thấy file: {p}")
        p = ask("Đường dẫn file .json / .json.gz")
    return p


def choose_model(ui):
    """Cho người dùng chọn file mô hình .pt; chưa có mô hình nào thì đề nghị huấn luyện ngay."""
    models = find_models()
    if not models:
        ui.warn("Chưa có mô hình nào trong thư mục models\\. Hãy huấn luyện trước (mục 2).")
        if confirm("Huấn luyện ngay bây giờ?"):
            return action_train(ui)
        return None
    if len(models) == 1:
        ui.log(f"Dùng mô hình {models[0]}")
        return models[0]
    return pick(ui, "Mô hình", [(m, m) for m in models])


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN D. CÁC CHỨC NĂNG CỦA MENU
# ════════════════════════════════════════════════════════════════════════════
def action_recommend(ui):
    """Menu 1: gợi ý size."""
    from .data import parse_height, parse_weight
    from .trainer import Bundle
    from .cli import cmd_recommend
    path = choose_model(ui)
    if not path:
        return
    b = Bundle(path)
    # xem mô hình dùng những số đo nào để chỉ hỏi đúng các số đo đó
    cols = set(b.body_enc.cols)
    # có cân nặng hoặc dáng người -> mô hình RentTheRunway; ngược lại là ModCloth
    is_rtr = "weight" in cols or bool(b.body_enc.types)
    examples = list(b.item_sizes.items())[:5]
    ui.log("Ví dụ item_id: " + ", ".join(f"{k} ({b.item_category.get(k, '?')}, size {', '.join(f'{s:g}' for s in v)})"
                                         for k, v in examples[:3]))
    customer = None
    while True:
        item = ask("item_id sản phẩm")
        if item not in b.item_sizes:
            ui.error(f"Không có item_id '{item}' trong mô hình.")
            continue
        if customer is None or not confirm("Dùng lại thông tin khách vừa nhập?"):
            customer = _ask_customer(ui, b, is_rtr, cols, parse_height, parse_weight)
        ns = argparse.Namespace(model=path, item=item, top=10, json=False,
                                explain=confirm("Hiển thị giải thích chi tiết?", default=True), **customer)
        ui.print()
        _run(ui, cmd_recommend, ns)
        if not confirm("Gợi ý cho sản phẩm khác?", default=True):
            break


def _ask_customer(ui, b, is_rtr, cols, parse_height, parse_weight):
    """Hỏi thông tin khách."""
    c = dict(user=None, height=None, weight=None, age=None, bust_size=None, bra_size=None, cup=None, bust=None,
             waist=None, hips=None, shoe=None, body_type=None, occasion=None)
    kind = pick(ui, "Loại khách", [("Khách mới (nhập số đo cơ thể)", "new"),
                                   ("Khách cũ (có user_id)", "old"),
                                   ("Khách cũ và cập nhật số đo", "both")])
    if kind in ("old", "both"):
        while True:
            u = ask("user_id")
            if u in b.indexer.maps["user_id"]:
                c["user"] = u
                ui.ok(f"Tìm thấy user {u}: {b.user_counts.get(u, 0)} giao dịch trong dữ liệu huấn luyện")
                break
            ui.warn(f"user_id {u} chưa có trong dữ liệu huấn luyện -> sẽ xử lý như khách mới.")
            if confirm("Vẫn dùng user_id này?", default=False):
                c["user"] = u
                break
    if kind == "old":
        return c
    ui.print("[dim]Nhấn Enter để bỏ qua mục không biết.[/dim]")
    while True:
        h = ask("Chiều cao (vd 168cm, 5'6\")", allow_empty=True)
        if h is None or parse_height(h) == parse_height(h):
            c["height"] = h
            break
        ui.error("Không đọc được chiều cao.")
    if is_rtr:
        while True:
            w = ask("Cân nặng (vd 59kg, 130lbs)", allow_empty=True)
            if w is None or parse_weight(w) == parse_weight(w):
                c["weight"] = w
                break
            ui.error("Không đọc được cân nặng.")
        # RentTheRunway: cỡ áo ngực dạng 34b
        c["bust_size"] = ask("Cỡ áo ngực (vd 34b)", allow_empty=True)
        bt = pick(ui, "Dáng người", [(t, t) for t in BODY_TYPES] + [("Không rõ", None)], default=len(BODY_TYPES) + 1)
        c["body_type"] = bt
        age = ask("Tuổi", allow_empty=True)
        c["age"] = float(age) if age and age.replace(".", "", 1).isdigit() else None
        if confirm("Nhập dịp sử dụng (rented for)?", default=False):
            c["occasion"] = pick(ui, "Dịp sử dụng", [(o, o) for o in OCCASIONS])
    else:
        bra = ask("Cỡ vòng dưới ngực / bra size (vd 34)", allow_empty=True)
        c["bra_size"] = float(bra) if bra and bra.replace(".", "", 1).isdigit() else None
        c["cup"] = ask("Cup (a, b, c, d, dd/e, ...)", allow_empty=True)
        for k, lab in [("hips", "Vòng hông (inch)"), ("waist", "Vòng eo (inch)"), ("bust", "Vòng ngực (inch)"),
                       ("shoe", "Cỡ giày US")]:
            if k in cols:
                v = ask(lab, allow_empty=True)
                c[k] = float(v) if v and v.replace(".", "", 1).isdigit() else None
    return c


def action_train(ui):
    """Menu 2: chọn dữ liệu, chọn nơi lưu, gọi cli.cmd_train để huấn luyện mô hình đề xuất."""
    from .cli import cmd_train
    from .data import detect_dataset
    data = choose_dataset(ui)
    out = ask("Lưu mô hình vào", default=os.path.join("models", f"{detect_dataset(data)}.pt"))
    ns = _hp_ns(data=data, model="mf_body_product", output=out, val_frac=0.1, seed=1)
    return out if _run(ui, cmd_train, ns) else None


def action_experiment(ui):
    """Menu 3: chạy thực nghiệm (nhanh ~1 phút / đầy đủ 10 seed / tuỳ chỉnh số seed) qua cli.cmd_experiment."""
    from .cli import cmd_experiment
    from .experiment import DEFAULT_MODELS
    from .data import detect_dataset
    data = choose_dataset(ui)
    mode = pick(ui, "Chế độ", [("Chạy nhanh: cold-start, 1 seed (khoảng 1 phút)", "quick"),
                               ("Đầy đủ: cold + warm, 10 seed, kiểm định thống kê (khoảng 15 đến 40 phút)", "full"),
                               ("Tuỳ chỉnh số seed", "custom")])
    seeds = list(range(1, 11))
    if mode == "custom":
        n = ask("Số seed", default="3", choices=[str(i) for i in range(1, 11)])
        seeds = list(range(1, int(n) + 1))
    out = ask("Thư mục kết quả", default=os.path.join("results", f"{detect_dataset(data)}_{mode}"))
    ns = _hp_ns(data=data, protocols=["cold", "warm"], seeds=seeds, models=list(DEFAULT_MODELS), boot=200,
                quick=mode == "quick", out_dir=out)
    _run(ui, cmd_experiment, ns)


def action_data(ui):
    """Menu 4: thống kê bộ dữ liệu (cli.cmd_data)."""
    from .cli import cmd_data
    _run(ui, cmd_data, argparse.Namespace(data=choose_dataset(ui), json=False))


def action_inspect(ui):
    """Menu 5: xem thông tin mô hình đã lưu (cli.cmd_inspect)."""
    from .cli import cmd_inspect
    m = choose_model(ui)
    if m:
        _run(ui, cmd_inspect, argparse.Namespace(model=m))


def action_info(ui):
    """Menu 6: thông tin hệ thống và các mô hình (cli.cmd_info)."""
    from .cli import cmd_info
    _run(ui, cmd_info, argparse.Namespace())


# ════════════════════════════════════════════════════════════════════════════
#  PHẦN E. VÒNG LẶP MENU CHÍNH
# ════════════════════════════════════════════════════════════════════════════

# danh sách chức năng của menu chính: (tên hiển thị, hàm xử lý)
MENU = [
    ("Gợi ý size cho khách hàng", action_recommend),
    ("Huấn luyện mô hình", action_train),
    ("Chạy thực nghiệm so sánh + kiểm định thống kê", action_experiment),
    ("Thống kê bộ dữ liệu", action_data),
    ("Xem thông tin mô hình đã lưu", action_inspect),
    ("Thông tin hệ thống & các mô hình", action_info),
]


def run_shell(ui: UI) -> int:
    """Vòng lặp menu chính."""
    if not sys.stdin.isatty():
        ui.error("Chế độ tương tác cần một terminal. Dùng `sizerec -h` để xem các lệnh.")
        return 2
    ui.header(f"sizerec {__version__} · Hệ thống gợi ý size thời trang",
              "Hybrid Matrix Factorization: latent cộng tác + số đo cơ thể tự khai · Ctrl+C để quay lại menu")
    models = find_models()
    ui.log(f"Thư mục làm việc: {os.getcwd()} · {len(find_datasets())} bộ dữ liệu · {len(models)} mô hình")
    while True:
        ui.print()
        ui.table(["#", "Chức năng"], [(i, lab) for i, (lab, _) in enumerate(MENU, 1)] + [(0, "Thoát")],
                 title="MENU CHÍNH", justify=["right", "left"])
        try:
            c = ask("Chọn chức năng", default="1", choices=[str(i) for i in range(0, len(MENU) + 1)])
            if c == "0":
                ui.ok("Tạm biệt!")
                return 0
            # gọi hàm của mục đã chọn (số thứ tự bắt đầu từ 1 nên trừ 1)
            MENU[int(c) - 1][1](ui)
        except KeyboardInterrupt:
            ui.print()
            ui.warn("Quay lại menu chính.")
        except EOFError:
            return 0
