# Hướng dẫn trình bày code (bản dễ hiểu)

Đọc từ trên xuống. Mỗi bước có 4 phần:

1. **Chuyện ở cửa hàng**: bước này giống việc gì ngoài đời.
2. **Code ở đâu**: mở file nào, bấm `Ctrl+G` rồi gõ số dòng.
3. **Đọc code bằng lời**: từng dòng code nghĩa là gì.
4. **Câu nói mẫu**: đọc khi chiếu cho thầy xem.

---

## 0. Cả đề tài nói gì? (đọc trước, chưa cần mở code)

Hãy tưởng tượng một **cửa hàng cho thuê váy online**. Khách không thử đồ được, nên hay thuê nhầm size. Mặc chật hoặc rộng thì khách trả hàng, cửa hàng mất tiền.

Ở cửa hàng thật, một **chị bán hàng lâu năm** đoán size rất giỏi nhờ 2 thứ:
- **Nhớ khách cũ:** "Chị Lan lần trước mặc size 8 bị chật, lần này nên lấy 10."
- **Nhìn dáng người:** "Khách mới cao 1m65, nặng 55kg thì chắc mặc size 6."

Đề tài này dạy **máy tính** làm việc của chị bán hàng đó. Máy học từ hàng trăm nghìn lần thuê đồ trong quá khứ, mỗi lần có ghi khách mặc **chật**, **vừa** hay **rộng**.

Vài từ sẽ gặp:

| Từ | Nghĩa đời thường |
| --- | --- |
| Dữ liệu | sổ ghi chép các lần thuê đồ |
| Mô hình | "bộ não" của máy, giống kinh nghiệm của chị bán hàng |
| Huấn luyện (train) | cho máy đọc sổ cũ để rút kinh nghiệm |
| Tập train / val / test | sổ để học / sổ để tự kiểm tra trong lúc học / sổ để thi cuối kỳ |
| Khách mới (cold-start) | khách lần đầu đến, máy chưa biết gì về lịch sử của họ |
| AUC | điểm thi của máy: 0.5 là đoán bừa, 1.0 là đoán đúng hết |

---

## Mở đầu: chương trình bắt đầu chạy từ đâu?

**Chuyện ở cửa hàng:** Cửa hàng có một quầy lễ tân. Khách nói "tôi muốn thuê đồ" hay "tôi muốn xem số liệu" thì lễ tân chỉ đúng phòng.

**Code ở đâu:** `sizerec/cli.py`, `Ctrl+G` **417**

```python
a.fn(a, ui)
```

**Đọc code bằng lời:** "Người dùng gõ lệnh gì thì chạy đúng phần việc đó." Ví dụ gõ `train` thì máy đi học, gõ `recommend` thì máy gợi ý size.

**Câu nói mẫu:**
> "Thưa thầy, chương trình bắt đầu ở hàm `main` trong file cli.py. Dòng 417 giống quầy lễ tân: người dùng gõ lệnh nào thì chuyển tới đúng phần xử lý lệnh đó."

---

## Bước 1. Lấy sổ ghi chép (dữ liệu thô)

**Chuyện ở cửa hàng:** Lấy cuốn sổ ghi các lần thuê đồ. Mỗi dòng ghi: ai thuê, váy nào, size mấy, mặc thấy thế nào. Dòng nào không ghi kết quả thì bỏ qua.

**Code ở đâu:** `sizerec/data.py`, `Ctrl+G` **185**

```python
raw = read_json_lines(path)
raw = raw[raw["fit"].isin(FIT_MAP)].reset_index(drop=True)
```

**Đọc code bằng lời:**
- Dòng 1: mở file dữ liệu, đọc từng dòng thành một bảng.
- Dòng 2: chỉ giữ dòng có kết quả là "chật", "vừa" hoặc "rộng".

**Câu nói mẫu:**
> "Em dùng 2 bộ dữ liệu thật là ModCloth và RentTheRunway. Mỗi dòng là một lần thuê, có ghi khách mặc chật, vừa hay rộng. Đây là thứ máy cần học để đoán."

---

## Bước 2. Làm sạch sổ + che đáp án

**Chuyện ở cửa hàng:**
- Khách ghi chiều cao lung tung: "5 feet 6", "168cm"... nên phải đổi hết về **một đơn vị**.
- Sổ còn có **lời nhận xét sau khi mặc**, ví dụ "váy chật quá". Nếu cho máy đọc thì giống **cho học sinh xem đáp án trước khi thi**, nên phải **che đi**.

**Code ở đâu:** `sizerec/data.py`, `Ctrl+G` **47**, rồi **161**, **164**

```python
LEAKY_FIELDS = ["review_text", "review_summary", "rating", "quality", "length"]
num["bmi"] = num["weight"] * 703.0 / num["height"] ** 2
num.loc[(num["height"] < 48) | (num["height"] > 84), "height"] = np.nan
```

**Đọc code bằng lời:**
- Dòng 47: danh sách 5 cột bị che (nhận xét, điểm chấm...). Máy không bao giờ được xem các cột này.
- Dòng 161: tính chỉ số BMI từ cân nặng và chiều cao.
- Dòng 164: chiều cao vô lý, ví dụ dưới 1m22 hoặc trên 2m13, coi như gõ nhầm và để trống.

**Câu nói mẫu:**
> "Bước này em làm 2 việc. Một là đổi số đo về cùng đơn vị và bỏ các số gõ nhầm. Hai là che các cột như lời nhận xét hay điểm chấm, vì chúng chỉ có sau khi khách mặc thử. Cho máy xem thì giống cho xem đáp án trước, kết quả sẽ đẹp giả."

---

## Bước 3. Chia sổ thành 3 phần

**Chuyện ở cửa hàng:** Muốn biết chị bán hàng giỏi thật hay không, ta không hỏi lại những khách chị đã gặp. Ta cho chị **học** 80% sổ, **tự kiểm tra** 10%, và **thi** 10% còn lại.

Có 2 kiểu thi:
- **Thi với khách quen (warm):** khách trong bài thi có thể đã từng thuê trước đó.
- **Thi với khách mới (cold):** khách trong bài thi **chưa từng** xuất hiện trong phần học. Đây là bài khó, giống khách lần đầu ghé cửa hàng.

**Code ở đâu:** `sizerec/data.py`, `Ctrl+G` **360**, rồi **369**

```python
if protocol == "warm":
    r = rng.rand(len(df))
...
tr, va = set(users[:int(.8 * n)]), set(users[int(.8 * n):int(.9 * n)])
```

**Đọc code bằng lời:**
- Dòng 360–361 (warm): mỗi lần thuê bốc thăm một số ngẫu nhiên để chia vào học, kiểm tra hay thi.
- Dòng 369 (cold): chia theo **người**. 80% số khách vào phần học, 10% vào kiểm tra, 10% còn lại để thi. Một người chỉ nằm trong đúng một phần.

**Câu nói mẫu:**
> "Em chia dữ liệu 80/10/10. Kiểu cold chia theo người, nên khách trong bài thi là khách hoàn toàn mới. Đây là tình huống khó nhất và cũng hay gặp nhất ngoài thực tế."

---

## Bước 4. Chỉ học từ phần "học"

**Chuyện ở cửa hàng:** Chị bán hàng ghi nhớ những thứ như "khách trung bình cao bao nhiêu" hay "danh sách khách quen". Những điều này chỉ được rút ra từ **phần sổ để học**, không được nhìn trộm bài thi.

**Code ở đâu:** `sizerec/trainer.py`, `Ctrl+G` **119**

```python
self.indexer  = Indexer().fit(train)
self.body_enc = BodyEncoder().fit(data.body.iloc[tr], train["body_type"])
self.size_reg = SizeRegressor().fit(self.X["train"], train["category"], train["size"])
```

**Đọc code bằng lời:** chữ `fit` nghĩa là "học". Cả 3 dòng đều chỉ học từ `train`, tức phần sổ để học.
- Dòng 119: lập **danh bạ khách và váy**, đánh số 1, 2, 3...
- Dòng 120: ghi nhớ **số đo trung bình** của khách, ví dụ chiều cao trung bình.
- Dòng 128: học công thức "người có số đo thế này thường mặc size mấy".

**Câu nói mẫu:**
> "Đây là chỗ quan trọng nhất để kết quả trung thực. Mọi thứ máy ghi nhớ đều chỉ lấy từ phần học. Phần thi không hề được nhìn trước."

---

## Bước 5. Dịch thông tin sang con số

**Chuyện ở cửa hàng:** Máy chỉ hiểu con số, nên phải "dịch":
- Tên khách → số thứ tự trong danh bạ. **Khách lạ chưa có trong danh bạ → số 0**, nghĩa là "chưa biết người này".
- Chiều cao → cao hơn hay thấp hơn trung bình bao nhiêu.
- Thêm một con số: **khách đặt size to hay nhỏ so với dáng người** (gọi là size_gap).

**Code ở đâu:** `sizerec/trainer.py`, `Ctrl+G` **134**

```python
ix = self.indexer.transform(f)
ix["gap"] = self.size_reg.gap(self.X[k], f["category"], f["size"])
```

**Đọc code bằng lời:** chữ `transform` nghĩa là "dịch sang số".
- Dòng 134: đổi tên khách và tên váy thành số thứ tự. Khách lạ thành 0.
- Dòng 135: tính size_gap. Ví dụ dáng người hợp size 8 mà đặt size 12, thì gap dương lớn, nghĩa là dễ bị **rộng**.

**Câu nói mẫu:**
> "Máy chỉ hiểu số, nên em đổi mọi thông tin sang số. Khách mới nhận số 0, tức máy biết đây là khách chưa có lịch sử. Em thêm size_gap để cho máy biết khách đặt size to hay nhỏ so với cơ thể."

---

## Bước 6. "Bộ não" đoán chật / vừa / rộng, và cách nó học

**Chuyện ở cửa hàng:** Chị bán hàng đoán dựa trên 2 nguồn rồi cộng lại thành một "điểm":
- **Trí nhớ về khách** (lịch sử): khách mới thì phần này bằng 0.
- **Dáng người** (số đo): ai cũng có, kể cả khách mới.

Điểm thấp thì đoán **chật**, điểm ở giữa thì **vừa**, điểm cao thì **rộng**. Giống thước đo có 2 vạch chia làm 3 vùng.

**Code ở đâu:** `sizerec/model.py`, `Ctrl+G` **161**

```python
out["user_collab"] = (self.P(b["user_id"]) * v).sum(-1) + ...
out["body"] = (self.E(b["x"]) * v).sum(-1) + self.g(b["x"])[:, 0]
c0, c1 = torch.sigmoid(th0 - t), torch.sigmoid(th1 - t)
```

**Đọc code bằng lời:**
- Dòng 161: phần điểm từ **trí nhớ về khách**. Khách mới có số 0 nên phần này bằng 0.
- Dòng 165: phần điểm từ **dáng người**. Phần này luôn có.
- Dòng 175: so tổng điểm với **2 vạch** để ra 3 xác suất: bao nhiêu phần trăm chật, vừa, rộng.

**Cách nó học:** `sizerec/trainer.py`, `Ctrl+G` **188**

```python
b["user_id"] = b["user_id"] * (torch.rand(len(b["y"]), generator=g) >= drop).long()
loss = F.nll_loss(torch.log(p), b["y"])
if val_auc > best + 1e-4:
```

- Dòng 188: khi học, cứ 10 lần thì khoảng 1 lần máy **cố tình "quên" khách là ai**. Nhờ vậy nó tập đoán chỉ bằng dáng người, và gặp khách mới thật vẫn đoán được.
- Dòng 193: tính "đoán sai nhiều hay ít" để máy tự sửa.
- Dòng 207: học xong mỗi vòng thì tự kiểm tra. Ba vòng liền không tiến bộ thì dừng, tránh học vẹt.

**Câu nói mẫu:**
> "Mô hình giống chị bán hàng: cộng điểm từ trí nhớ về khách và từ dáng người, rồi so với 2 vạch để ra chật, vừa hay rộng. Vì có phần dáng người nên khách mới vẫn đoán được. Em có thử thêm cách cho máy thỉnh thoảng giả vờ không biết khách là ai lúc học. Kết quả cho thấy cách này không tạo khác biệt đáng kể. Điểm mạnh chính đến từ việc kết hợp lịch sử với số đo."

**Lưu ý nếu thầy hỏi:** mô hình bỏ "giả vờ quên" (`mf_body_product_nodrop`) cho điểm gần như bằng mô hình đề xuất. Trên RTR cold, chênh lệch là 0.0003 với p = 0.46, tức không có ý nghĩa thống kê. Nên nói thật như câu mẫu ở trên, đừng nói đây là điểm làm mô hình tốt hơn.

---

## Bước 7. Chấm điểm và so với cách làm khác

**Chuyện ở cửa hàng:** Cho nhiều "người bán hàng" cùng làm **một bài thi** để so ai giỏi hơn:
- Người lười: lúc nào cũng nói "vừa".
- Người chỉ nhìn dáng (Logistic Regression).
- Người chỉ nhớ khách (MF thuần).
- Người dùng cả hai: **mô hình của em**.

Thi lại 10 lần với 10 cách chia khác nhau, rồi dùng thống kê kiểm tra xem hơn nhau **thật** hay chỉ **ăn may**.

**Code ở đâu:** `sizerec/experiment.py`, `Ctrl+G` **51**, **61**, **174**

```python
tr, va, te = split(data.df, protocol, seed)
for name in models:
p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
```

**Đọc code bằng lời:**
- Dòng 51: chia sổ **một lần**, mọi người dùng chung để công bằng.
- Dòng 61: lần lượt cho từng "người bán hàng" làm bài.
- Dòng 174: tính p-value. **p < 0.05 nghĩa là hơn thật, không phải may mắn.**

**Câu nói mẫu:**
> "Em so 7 cách làm trên cùng một bài thi, lặp 10 lần. Với khách mới trên RentTheRunway, mô hình của em đạt AUC 0.7305. Cách chỉ nhìn dáng người đạt 0.6195, cách chỉ nhớ khách đạt 0.7220. Kiểm định thống kê cho thấy chênh lệch này là thật."

---

## Bước 8. Gợi ý size cho một khách

**Chuyện ở cửa hàng:** Khách hỏi "váy này tôi mặc size mấy?".
- Từ dáng người, chị đoán một size, ví dụ 8.
- Nếu là khách quen từng thuê size 10 vừa, chị nghiêng về 10. Khách thuê càng nhiều lần thì chị càng tin lịch sử.
- Váy này hay chạy nhỏ thì cộng thêm một chút.
- Chọn size cửa hàng **có sẵn** gần nhất.

**Code ở đâu:** `sizerec/sizing.py`, `Ctrl+G` **89**

```python
w = n / (n + self.k_user)
```

**Đọc code bằng lời:** `n` là số lần khách đã thuê. Thuê 1 lần thì tin lịch sử 33%, 2 lần thì 50%, 8 lần thì 80%. Khách mới có n = 0 nên chỉ dựa vào dáng người.

**Code ở đâu:** `sizerec/recommend.py`, `Ctrl+G` **65**, **83**, **92**

```python
known_user = user_id is not None and str(user_id) in bundle.indexer.maps["user_id"]
probs = bundle.model(b).numpy()
rec_size = nearest(sizes, tgt["target"])
```

- Dòng 65: khách này là khách quen hay khách mới?
- Dòng 83: với mỗi size, máy cho biết bao nhiêu phần trăm chật, vừa, rộng.
- Dòng 92: chọn size có sẵn gần nhất, đó là **size đề xuất**.

**Câu nói mẫu:**
> "Khi khách hỏi size, máy đoán từ dáng người, rồi điều chỉnh theo lịch sử nếu là khách quen, và theo việc váy hay chạy to hay nhỏ. Sau đó máy chọn size có sẵn gần nhất và báo kèm rủi ro chật hoặc rộng. Với khách mới trên RentTheRunway, máy chọn trúng đúng size 40.1% số lần, và lệch không quá 1 size 79.1% số lần."

---

## Nếu thầy muốn xem chạy thật

Mở Terminal trong VS Code, đứng ở thư mục `03-code`, gõ:

```
python scripts/verify_workflow.py ../02-data/modcloth_final_data.json.gz
```

Máy sẽ chạy lần lượt **BƯỚC 1 → BƯỚC 8** và in `[OK]` sau mỗi lần tự kiểm tra thành công. Mất khoảng 30 giây đến 2 phút.

> "Đây là script chạy lại đúng 8 bước em vừa trình bày. Mỗi dòng OK là một lần máy tự kiểm tra code làm đúng như sơ đồ."
