"""
ui.py — IN RA MÀN HÌNH (thông báo, bảng, khung tiêu đề)

Có thư viện `rich` thì in màu và bảng đẹp; không có thì in văn bản thuần.
Các file khác chỉ gọi qua lớp UI, nên đổi cách hiển thị chỉ cần sửa ở đây.
"""

from __future__ import annotations

import sys

try:
    # rich: thư viện in màu và bảng đẹp trên terminal (không bắt buộc)
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.markup import escape
    from rich.table import Table

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

# Windows: đổi console sang UTF-8 để hiện đúng tiếng Việt
if sys.platform == "win32":  # tiếng Việt trên console Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class UI:
    """Gom mọi lệnh in ra màn hình vào một chỗ (thông báo, bảng, khung tiêu đề)."""

    def __init__(self, quiet: bool = False, color: bool = True):
        """quiet=True: ẩn các dòng log ; color=False: tắt màu."""
        self.quiet = quiet
        # console in ra màn hình chính; err in ra luồng lỗi (stderr)
        self.console = Console(highlight=False, no_color=not color) if HAS_RICH else None
        self.err = Console(stderr=True, highlight=False, no_color=not color) if HAS_RICH else None

    # ────────────────────────────────────────────────────────────────────────
    # In văn bản
    def print(self, msg: str = "", style: str | None = None):
        """In 1 dòng (có rich thì giữ định dạng màu, không có thì bỏ thẻ định dạng)."""
        if self.console:
            self.console.print(msg, style=style)
        else:
            print(_strip(msg))

    # log: chữ mờ, bị ẩn khi chạy với -q (quiet)
    def log(self, msg: str):
        """In dòng log mờ; bị ẩn khi chạy với -q."""
        if not self.quiet:
            self.print(f"[dim]{escape(msg) if HAS_RICH else msg}[/dim]")

    # step / ok / warn: dòng có biểu tượng › ✓ ! ở đầu
    def step(self, msg: str):
        """In dòng bắt đầu bằng › (một bước đang làm)."""
        self.print(f"[bold cyan]›[/bold cyan] {msg}")

    def ok(self, msg: str):
        """In dòng bắt đầu bằng ✓ (thành công)."""
        self.print(f"[bold green]✓[/bold green] {msg}")

    def warn(self, msg: str):
        """In dòng bắt đầu bằng ! (cảnh báo)."""
        self.print(f"[bold yellow]![/bold yellow] {msg}")

    def error(self, msg: str):
        """In thông báo lỗi ra luồng lỗi (stderr)."""
        if self.err:
            self.err.print(f"[bold red]Lỗi:[/bold red] {msg}")
        else:
            print(f"Lỗi: {_strip(msg)}", file=sys.stderr)

    def header(self, title: str, subtitle: str = ""):
        """In khung tiêu đề (tên + mô tả phụ)."""
        if self.console:
            self.console.print(Panel.fit(f"[bold]{title}[/bold]" + (f"\n[dim]{subtitle}[/dim]" if subtitle else ""),
                                         border_style="cyan"))
        else:
            line = "=" * max(len(title), len(subtitle), 20)
            print(f"{line}\n{title}\n{subtitle}\n{line}" if subtitle else f"{line}\n{title}\n{line}")

    # ────────────────────────────────────────────────────────────────────────
    # In bảng
    def table(self, columns: list, rows: list, title: str | None = None, highlight_row=None, justify=None):
        """In bảng; highlight_row(i, r) trả True thì tô đậm dòng đó."""
        justify = justify or ["left"] + ["right"] * (len(columns) - 1)
        if self.console:
            t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold", header_style="bold cyan")
            for c, j in zip(columns, justify):
                t.add_column(c, justify=j)
            for i, r in enumerate(rows):
                t.add_row(*[escape(str(x)) for x in r], style="bold green" if highlight_row and highlight_row(i, r) else None)
            self.console.print(t)
        else:
            # không có rich: tự canh cột theo độ dài chữ dài nhất mỗi cột
            widths = [max(len(_strip(str(c))), *(len(_strip(str(r[i]))) for r in rows)) if rows else len(c)
                      for i, c in enumerate(columns)]
            if title:
                print(f"\n{_strip(title)}")
            fmt = lambda vals: "  ".join(_strip(str(v)).ljust(w) if j == "left" else _strip(str(v)).rjust(w)
                                         for v, w, j in zip(vals, widths, justify))
            print(fmt(columns))
            print("  ".join("-" * w for w in widths))
            for r in rows:
                print(fmt(r))
            print()

    def kv(self, pairs: list, title: str | None = None):
        """In bảng 2 cột "Mục | Giá trị"."""
        self.table(["Mục", "Giá trị"], pairs, title=title, justify=["left", "left"])


def _strip(s: str) -> str:
    """Bỏ thẻ định dạng [bold]...[/bold] của rich khi in văn bản thuần."""
    import re
    return re.sub(r"\[/?[a-z ]+[a-z0-9 _#=-]*\]", "", s)
