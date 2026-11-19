# Bảng tra số dòng (bản chú thích gọn, 15/11/2026)

**Cách đọc code:** mỗi file mở đầu bằng **MỤC LỤC** và **THẺ TÌM NHANH**. Mỗi hàm có 1 dòng mô tả ngay dưới `def`. Trong thân hàm chỉ chú thích các bước chính (1)(2)(3) và các dòng quan trọng (có thẻ bên dưới).

Trong VS Code: `Ctrl+F` gõ thẻ để nhảy tới chỗ quan trọng, `Ctrl+G` gõ số dòng để nhảy tới dòng.

| Thẻ | Ý nghĩa |
| --- | --- |
| `[RÒ RỈ]` | chỗ chống rò rỉ dữ liệu |
| `[COLD-START]` | chỗ xử lý khách mới |
| `[SIZE_GAP]` | đặc trưng size_gap |
| `[ORDINAL]` | đầu ra thứ bậc chật < vừa < rộng |
| `[HYBRID]` | chỗ kết hợp lịch sử + số đo cơ thể |
| `[ID DROPOUT]` | kỹ thuật giả làm khách mới lúc train |
| `[KIỂM ĐỊNH]` | paired t-test và paired bootstrap |

**Logic không đổi**: cây cú pháp (AST) của mọi file giống hệt bản gốc, `tests/test_smoke.py` chạy OK. Cột "Dòng gốc" là số dòng của bản chưa có chú thích (bản trước khi thêm chú thích). Từ 17/11, file docx và pptx trong `07-code-khop-workflow` đã dùng số dòng MỚI.

## data.py (bước 1–5: đọc, làm sạch, chia, mã hoá)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `FIT_MAP` | 23 | **36** |
| `LEAKY_FIELDS` | 27 | **47** |
| `parse_height` | 35–47 | **61–79** |
| `parse_weight` | 50–59 | **82–91** |
| `parse_bust_size` | 69–77 | **102–111** |
| `read_json_lines` | 90–93 | **128–132** |
| `body_frame` | 96–118 | **135–167** |
| `load_dataset` | 132–147 | **182–205** |
| `BodyEncoder` | 151–190 | **216–265** |
| `BodyEncoder.fit` | 154–163 | **219–230** |
| `BodyEncoder.transform` | 173–180 | **242–253** |
| `SizeRegressor` | 193–228 | **268–311** |
| `SizeRegressor.fit` | 199–208 | **271–286** |
| `SizeRegressor.gap` | 218–219 | **298–300** |
| `Indexer` | 234–258 | **318–348** |
| `Indexer.transform` | 246–249 | **332–337** |
| `split` | 261–275 | **355–372** |
| `history_bucket` | 278–279 | **375–377** |
| `describe` | 282–294 | **380–393** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| Lọc nhãn hợp lệ | 134 | **186** |
| Tạo item_size | 144 | **200** |
| μ học trên train | 156 | **222** |
| Nghiệm Ridge | 206 | **282** |
| Id lạ -> 0 | 247 | **335** |
| Chia cold theo user | 272 | **369** |

## trainer.py (bước 4–7: chống rò rỉ, huấn luyện, đánh giá)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `HParams` | 22–33 | **43–65** |
| `tune_decision` | 40–50 | **76–86** |
| `auc` | 53–57 | **89–94** |
| `metrics` | 60–63 | **97–101** |
| `Prepared` | 67–90 | **108–140** |
| `train_model` | 106–143 | **161–217** |
| `predict` | 146–150 | **220–225** |
| `body_lr` | 154–167 | **232–247** |
| `save_bundle` | 171–190 | **254–274** |
| `Bundle` | 193–219 | **277–313** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| Indexer.fit (chỉ train) | 76 | **119** |
| BodyEncoder.fit (chỉ train) | 77 | **120** |
| SizeRegressor.fit (chỉ train) | 81 | **128** |
| ID dropout | 124 | **188** |
| Loss NLL | 126 | **193** |
| Early stopping | 135 | **207** |
| Nạp bản tốt nhất | 141 | **215** |

## model.py (bước 6: kiến trúc)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `ModelSpec` | 22–30 | **38–56** |
| `SPECS` | 34–40 | **68–74** |
| `OrdinalHybridMF` | 53–112 | **93–186** |
| `OrdinalHybridMF.__init__` | 54–76 | **96–130** |
| `OrdinalHybridMF.thresholds` | 79–80 | **133–135** |
| `OrdinalHybridMF.terms` | 82–99 | **137–166** |
| `OrdinalHybridMF.forward` | 101–105 | **168–178** |
| `OrdinalHybridMF.l2_terms` | 107–112 | **180–186** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| padding_idx=0 | 57 | **103** |
| Hàng 0 = vector 0 | 72 | **122** |
| Mạng E(x) | 73 | **125** |
| Phần lịch sử user_collab | 96 | **161** |
| Phần cơ thể body | 98 | **165** |
| Sigmoid thứ bậc | 104 | **175** |
| 3 xác suất | 105 | **178** |

## sizing.py (bước 8: chọn size)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `SizeModel.fit` | 29–51 | **39–73** |
| `SizeModel.target` | 56–67 | **79–97** |
| `nearest` | 81–82 | **117–119** |
| `evaluate_size_rec` | 85–108 | **122–146** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| item_offset | 43 | **61** |
| Hiệu chỉnh theo nhãn | 44 | **64** |
| Trọng số lịch sử | 61 | **89** |

## recommend.py (bước 8: gợi ý cho 1 khách)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `body_from_inputs` | 19–25 | **36–48** |
| `recommend` | 28–75 | **55–116** |
| `_advice` | 78–101 | **119–147** |
| `body_contributions` | 104–113 | **154–163** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| Khách cũ hay mới | 34 | **65** |
| Gọi mô hình hybrid MF | 48 | **83** |
| Chọn size đề xuất | 55 | **92** |

## experiment.py (bước 7: thực nghiệm, kiểm định)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `run_experiment` | 22–66 | **41–100** |
| `summarize` | 69–75 | **107–114** |
| `fast_auc` | 78–89 | **121–132** |
| `significance` | 92–131 | **135–180** |
| `write_outputs` | 142–156 | **196–216** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| Paired t-test | 108 | **152** |
| p bootstrap | 125 | **174** |

## cli.py (điểm vào chương trình)

| Hàm / lớp | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| `cmd_data` | 94–108 | **110–126** |
| `cmd_experiment` | 111–138 | **129–162** |
| `cmd_train` | 175–202 | **203–235** |
| `cmd_inspect` | 205–217 | **238–252** |
| `cmd_recommend` | 220–273 | **255–313** |
| `build_parser` | 277–350 | **319–393** |
| `main` | 353–375 | **396–421** |

| Dòng code quan trọng | Dòng gốc | Dòng MỚI |
| --- | --- | --- |
| Dòng chạy lệnh a.fn(a, ui) | 371 | **417** |
