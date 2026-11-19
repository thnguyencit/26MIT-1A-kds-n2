# Hướng dẫn chạy thực nghiệm

**Đề tài:** Sử dụng Machine Learning (Matrix Factorization kết hợp thông số cơ thể tự kê khai) để dự đoán kích cỡ (size)
phù hợp nhất cho người mua, nhằm giảm tỉ lệ trả hàng do lệch size trong thương mại điện tử thời trang.

**Mô hình đề xuất:** Hybrid Ordinal Matrix Factorization, viết tắt `mf_body_product`. Mô hình tích hợp trực tiếp hai nguồn thông tin:
- các nhân tố ẩn cộng tác, gồm user, item và item-size;
- số đo cơ thể do khách tự khai, gồm chiều cao, cân nặng, vòng ngực, eo, hông và dáng người.

Mô hình tập trung vào khách mới (cold-start). Thực nghiệm dùng hai bộ dữ liệu ModCloth và RentTheRunway.

Công cụ được đóng gói thành giao diện dòng lệnh `sizerec`, có thêm chế độ menu tương tác.

---

## 1. Cấu trúc thư mục

```
baitapfinal/
├── 01-huongdan.md          ← file này
├── 02-data/                dữ liệu ModCloth, RentTheRunway (.json.gz) + README nguồn dữ liệu
├── 03-code/                toàn bộ mã nguồn và script chạy thực nghiệm
│   ├── sizerec/            gói Python (CLI, mô hình, huấn luyện, thực nghiệm, gợi ý size)
│   ├── tests/              kiểm thử pipeline trên dữ liệu giả lập
│   ├── scripts/            run_all.sh, run_all.bat, run_quick.sh, demo_recommend.sh,
│   │                       make_figures.py, make_figures_en.py, export_excel.py
│   ├── pyproject.toml, requirements.txt
│   └── sizerec.code-workspace   cấu hình VS Code (11 cấu hình chạy/debug)
├── 04-results/             toàn bộ kết quả đã chạy
│   ├── experiment/{modcloth,rtr}/   CSV kết quả, kiểm định, report.md, config.json
│   ├── logs/               log .txt của mọi lệnh đã chạy
│   ├── figures/            hình PNG 300 dpi (figures/en: bản tiếng Anh) + data/*.csv (số liệu gốc)
│   ├── models/             mô hình đã huấn luyện (.pt) dùng cho gợi ý size
│   ├── scripts/            bản sao các script đã dùng để tạo kết quả
│   └── results_summary.xlsx  tổng hợp mọi bảng kết quả vào một file Excel
└── 05-report/              báo cáo PDF, source LaTeX (Overleaf), kết quả Turnitin
```

## 2. Cài đặt môi trường

Yêu cầu: Python 3.9 trở lên (đã kiểm tra trên 3.11). Không cần GPU: toàn bộ thực nghiệm chạy trên CPU.

**Windows (PowerShell)**, đứng trong `baitapfinal\03-code`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # hoặc: pip install torch
pip install -r requirements.txt
pip install -e .            # đăng ký lệnh `sizerec`
```

**Linux / macOS**, đứng trong `baitapfinal/03-code`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Kiểm tra cài đặt:

```bash
sizerec info                 # phiên bản, môi trường, danh sách mô hình
python tests/test_smoke.py   # kiểm thử toàn bộ pipeline trên dữ liệu giả lập, in "OK"
```

Nếu không chạy `pip install -e .`, thay `sizerec` bằng `python -m sizerec` (khi đứng trong `03-code`).

## 3. Chạy lại toàn bộ thực nghiệm

Mọi lệnh dưới đây chạy khi đứng trong `03-code`, với venv đang bật.

| Mục đích | Linux / macOS / Git Bash | Windows (cmd / PowerShell) | Thời gian* |
| --- | --- | --- | --- |
| Chạy thử nhanh (ModCloth, cold, 1 seed) | `bash scripts/run_quick.sh` | `sizerec experiment ..\02-data\modcloth_final_data.json.gz --quick -o ..\04-results\quick\modcloth` | khoảng 1 phút |
| **Chạy toàn bộ** và tái tạo `04-results` | `bash scripts/run_all.sh` | `scripts\run_all.bat` | 20-30 phút |
| Chỉ huấn luyện, demo, vẽ hình, xuất Excel (dùng lại kết quả thực nghiệm đã có) | `SKIP_EXPERIMENT=1 bash scripts/run_all.sh` | `scripts\run_all.bat skip` | khoảng 3 phút |
| Chỉ vẽ lại hình | `python scripts/make_figures.py ../04-results` | như bên trái | vài giây |
| Chỉ xuất lại Excel | `python scripts/export_excel.py ../04-results` | như bên trái | vài giây |

\* Đo trên CPU 4 luồng. ModCloth đầy đủ mất khoảng 4 phút, RentTheRunway đầy đủ khoảng 12 phút.

`run_all.sh` thực hiện lần lượt các bước:

1. `sizerec info`, rồi `sizerec data` cho từng bộ dữ liệu (thống kê, tỉ lệ thiếu số đo). Kết quả ghi vào `logs/info.txt` và `logs/data_*.txt`.
2. Kiểm thử pipeline. Kết quả ghi vào `logs/smoke_test.txt`.
3. Chạy `sizerec experiment` trên từng bộ dữ liệu. Thực nghiệm gồm:
   - 7 mô hình, 2 giao thức (cold và warm), 10 seed;
   - paired bootstrap 200 lần và paired t-test.

   Kết quả ghi vào `experiment/<ds>/` và `logs/experiment_*.txt`.
4. Chạy `sizerec train` để huấn luyện mô hình đề xuất trên toàn bộ dữ liệu. Kết quả ghi vào `models/*.pt` và `logs/train_*.txt`.
5. Chạy `scripts/demo_recommend.sh` để chạy các ví dụ gợi ý size. Kết quả ghi vào `logs/demo_recommend.txt`.
6. Chạy `make_figures.py` và `export_excel.py`, tạo ra `figures/*.png`, `figures/data/*.csv` và `results_summary.xlsx`.

Các seed được cố định (1 đến 5). Vì vậy chạy lại sẽ cho cùng cách chia dữ liệu, và kết quả trùng với `04-results` (sai khác rất nhỏ do khác phiên bản thư viện hoặc CPU).

## 4. Các lệnh của công cụ `sizerec`

| Lệnh | Việc làm |
| --- | --- |
| `sizerec` hoặc `sizerec shell` | Mở **menu tương tác**: gợi ý size, huấn luyện, thực nghiệm, thống kê dữ liệu |
| `sizerec info` | Phiên bản, môi trường, danh sách mô hình |
| `sizerec data <file>` | Thống kê bộ dữ liệu |
| `sizerec experiment <file> [-o thư_mục] [--quick] [--seeds ...] [--models ...]` | Thực nghiệm so sánh + kiểm định thống kê |
| `sizerec train <file> -o <x.pt>` | Huấn luyện và lưu mô hình đề xuất |
| `sizerec inspect <x.pt>` | Xem thông tin mô hình đã lưu |
| `sizerec recommend <x.pt> --item ID [--user ID] [số đo...] [--explain] [--json]` | Gợi ý size cho khách mới (theo số đo) hoặc khách cũ (theo user_id) |

Gõ `sizerec <lệnh> -h` để xem đầy đủ tuỳ chọn. Ví dụ gợi ý size:

```bash
# Khách mới (cold-start), chỉ có số đo tự khai
sizerec recommend ../04-results/models/rtr.pt --item 126335 --height 165cm --weight 57kg --bust-size 34b --body-type petite --explain
# Khách cũ, dùng lịch sử mua
sizerec recommend ../04-results/models/rtr.pt --item 126335 --user 691468 --explain
# Đầu ra JSON
sizerec recommend ../04-results/models/rtr.pt --item 126335 --height 168cm --weight 59kg --json
```

Chế độ menu tự tìm dữ liệu trong `../02-data` và mô hình trong `../04-results/models`. Chương trình hỏi lần lượt item_id, loại khách và từng số đo. Mục nào không biết thì nhấn Enter để bỏ qua.

**VS Code:** mở file `03-code/sizerec.code-workspace` (File > Open Workspace from File). Sau đó vào Run and Debug (Ctrl+Shift+D), chọn một cấu hình rồi nhấn F5.
- Có 11 cấu hình, đi từ menu, thống kê dữ liệu, thực nghiệm, huấn luyện, gợi ý size, đến kiểm thử.
- Interpreter mặc định là `03-code/.venv`. Nếu dùng venv khác, chọn lại bằng Ctrl+Shift+P > Python: Select Interpreter.

## 5. Phương pháp

### 5.1 Các mô hình so sánh

Tất cả mô hình dùng chung một cách chia dữ liệu và cùng hàm loss hồi quy thứ bậc.

| Tên | Thành phần |
| --- | --- |
| `majority` | Luôn đoán "fit" |
| `body_lr` | Logistic Regression trên số đo, size, danh mục |
| `mf` | MF thứ bậc thuần: latent user, item, item-size |
| `mf_body` | MF + số đo cơ thể tích hợp vào vector ẩn người dùng |
| `body_product` | Số đo + đặc trưng sản phẩm, không có latent người dùng |
| **`mf_body_product`** | **Đề xuất: MF + số đo cơ thể + đặc trưng sản phẩm** |
| `mf_body_product_nodrop` | Ablation: mô hình đề xuất nhưng không dùng ID dropout |

### 5.2 Công thức mô hình đề xuất

Nhãn là thứ bậc: small < fit < large.

```
u = m_u · p_u + E(x_u)
v = q_(item,size) + q_item + c_category + c_context + size_z · w + gap · w_gap
t = <u, v> + m_u · b_u + b_(item,size) + b_item + g(x_u) + β · gap
P(y ≤ k) = σ(θ_k − t)
```

Trong đó:
- `p_u` là nhân tố ẩn học từ lịch sử.
- `x_u` là số đo tự khai. `E` là MLP chiếu `x_u` vào cùng không gian với `p_u`.
- `m_u` bằng 0 với khách mới, khi đó dự đoán dựa hoàn toàn vào số đo.
- `gap` là độ chênh giữa size đã đặt và size ước lượng từ số đo (Ridge theo danh mục).
- Khi huấn luyện, ID dropout 10% giúp mô hình học cách dự đoán cho khách chưa có lịch sử.

### 5.3 Giao thức đánh giá

- **cold:** chia 80/10/10 theo người dùng. Người dùng trong tập test chưa từng xuất hiện, mô phỏng khách mới.
- **warm:** chia ngẫu nhiên theo giao dịch. Kết quả được tách theo số giao dịch của người dùng trong tập train: 0, 1, 2-4, 5+.

### 5.4 Chỉ số và kiểm định

- **Chỉ số:**
  - AUC macro (one-vs-rest) là chỉ số chính.
  - Macro-F1 được tính sau khi chọn trọng số quyết định trên tập val, vì khoảng 70% nhãn là fit.
- **Kiểm định:** so sánh mô hình đề xuất với từng mô hình còn lại bằng hai cách:
  - paired t-test trên AUC theo 10 seed;
  - paired bootstrap trên tập test, cho ra khoảng tin cậy 95% của ΔAUC và p-value.

### 5.5 Đề xuất size

Dữ liệu fit là dữ liệu quan sát, nên có thiên lệch chọn mẫu: khách tự chọn size gần với cơ thể mình. Vì thế, size được chọn bằng một mô hình kích cỡ đơn điệu, diễn giải được. Size mục tiêu được tính từ:
- size ước lượng từ số đo;
- size nền theo lịch sử mua của khách (nếu có);
- độ lệch theo danh mục và mức "chạy to/nhỏ" của sản phẩm.

Mô hình hybrid ước lượng rủi ro chật hoặc rộng ở size được chọn. Chương trình chỉ khuyên đổi size khi size bên cạnh thật sự an toàn hơn.

Chất lượng đề xuất được đo trên các giao dịch "fit" của tập test bằng hai chỉ số:
- tỉ lệ trùng đúng size khách mặc vừa;
- tỉ lệ lệch không quá 1 size.

Đây là chỉ số gần nhất với mục tiêu giảm trả hàng.

### 5.6 Chống rò rỉ dữ liệu

Các trường chỉ có sau khi mua (`review_text`, `review_summary`, `rating`, `quality`, `length`) bị loại.

- Mỗi seed tạo một cách chia train/val/test riêng. Ở cùng seed, mọi mô hình dùng chung cách chia.
- Mọi thành phần cần "học" đều chỉ được ước lượng trên phần train của từng lần chia:
  - thống kê chuẩn hoá z-score và cột báo thiếu của số đo;
  - bảng ánh xạ ID;
  - chuẩn hoá size;
  - hồi quy ridge ước lượng size (dùng cho đặc trưng gap);
  - độ lệch to/nhỏ của sản phẩm trong mô hình chọn size.
- Tập val chỉ dùng để dừng sớm và chọn trọng số lớp cho Macro-F1.
- Tập test chỉ dùng một lần để tính kết quả cuối.
- Thông tin duy nhất lấy từ toàn bộ dữ liệu là danh sách size mà mỗi sản phẩm có bán. Cửa hàng biết thông tin này trước khi bán, nên không gây rò rỉ.
- Trước khi chạy đầy đủ, toàn bộ quy trình đã được thử trên một phần nhỏ dữ liệu (`scripts/run_quick.sh`, `tests/test_smoke.py`).

## 6. Kết quả chính (trung bình 10 seed)

Bảng đầy đủ: `04-results/results_summary.xlsx` và `04-results/experiment/*/report.md`.

**AUC macro, giao thức cold-start (khách mới):**

| Mô hình | ModCloth | RentTheRunway |
| --- | --- | --- |
| body_lr | 0.585 ± 0.003 | 0.620 ± 0.004 |
| mf | 0.661 ± 0.006 | 0.722 ± 0.003 |
| mf_body | 0.674 ± 0.006 | 0.724 ± 0.004 |
| body_product | 0.678 ± 0.007 | 0.729 ± 0.004 |
| **mf_body_product (đề xuất)** | **0.679 ± 0.007** | **0.731 ± 0.004** |
| mf_body_product_nodrop (ablation) | 0.679 ± 0.007 | 0.731 ± 0.003 |

**Độ chính xác đề xuất size (warm, các giao dịch "fit" của tập test):**

| Phương pháp | ModCloth: trùng size | ModCloth: lệch ≤ 1 size | RTR: trùng size | RTR: lệch ≤ 1 size |
| --- | --- | --- | --- | --- |
| Size bán chạy nhất | 25.5 ± 0.6% | 62.4 ± 0.6% | 20.9 ± 0.4% | 51.5 ± 0.3% |
| Chỉ số đo cơ thể | 44.0 ± 0.7% | 87.7 ± 0.3% | 39.7 ± 0.4% | 79.0 ± 0.3% |
| **Số đo + lịch sử (hybrid)** | **51.0 ± 0.5%** | **91.7 ± 0.3%** | **44.5 ± 0.5%** | **82.3 ± 0.3%** |

**Nhận xét:**
- Mô hình đề xuất hơn có ý nghĩa thống kê so với `majority`, `body_lr`, `mf` và `mf_body` ở cả hai bộ dữ liệu.
- So với `body_product` (không có latent người dùng):
  - ở cold-start, mức chênh nhỏ: +0.001 trên ModCloth (không có ý nghĩa thống kê, p = 0.08) và +0.002 trên RTR (p < 0.01);
  - ở warm, mô hình đề xuất hơn có ý nghĩa ở cả hai bộ (p < 0.001). Mức chênh rõ nhất với khách có lịch sử 2-4 giao dịch (+0.015 trên ModCloth, +0.007 trên RTR) và 5+ giao dịch (+0.013 và +0.014).
- Như vậy, thành phần cộng tác phát huy tác dụng khi khách đã có lịch sử. Với khách mới, số đo cơ thể và đặc trưng sản phẩm đóng vai trò chính.
- Không dùng ID dropout cho kết quả tương đương (p > 0.28).
- Boxplot qua 10 lần lặp (`fig6_auc_boxplot.png`) cho thấy thứ hạng các mô hình ổn định giữa các lần chia dữ liệu.

Các hình trong `04-results/figures/`:

| Hình | Nội dung |
| --- | --- |
| `fig1_data_overview.png` | Phân bố nhãn fit và tỉ lệ thiếu số đo |
| `fig2_auc_cold_start.png` | AUC cold-start của các mô hình |
| `fig3_auc_by_history.png` | AUC theo số giao dịch lịch sử của người dùng |
| `fig4_significance_delta_auc.png` | ΔAUC và CI 95% của mô hình đề xuất so với từng mô hình |
| `fig5_size_recommendation.png` | Độ chính xác đề xuất size |
| `fig6_auc_boxplot.png` | Boxplot AUC qua 10 lần lặp của từng mô hình (cold và warm) |

## 7. Tài liệu tham khảo

- R. Misra, M. Wan, J. McAuley. *Decomposing fit semantics for product size recommendation in metric spaces.* RecSys 2018.
- V. Sembium, R. Rastogi, A. Saroop, S. Merugu. *Recommending product sizes to customers.* RecSys 2017.
- Y. Koren, R. Bell, C. Volinsky. *Matrix factorization techniques for recommender systems.* IEEE Computer, 2009.
