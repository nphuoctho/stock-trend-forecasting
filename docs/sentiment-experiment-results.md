# Bản ghi thực nghiệm phân loại cảm xúc

> **Trạng thái:** Kết quả xác thực chéo theo tầng trên 1.306 nhãn đã rà soát là kết quả sẵn sàng đưa vào báo cáo. Các phần về 306 mẫu bên dưới chỉ là bản ghi lịch sử của đường cơ sở và ablation.
>
> **Nguồn số liệu hiện tại:** [`../outputs/sentiment-cv-merged/cv_results.json`](../outputs/sentiment-cv-merged/cv_results.json) và [`cv_results.csv`](../outputs/sentiment-cv-merged/cv_results.csv), được trích từ archive Kaggle đã kiểm tra toàn vẹn.

## Kết quả hiện tại: xác thực chéo theo tầng trên 1.306 nhãn

Lần chạy dùng `title_context + head_tail`, 5 epoch, kích thước lô 16, hạt giống gốc
`42` và `inverse_frequency`. Tập có 194 `NEGATIVE`, 771 `NEUTRAL` và 341
`POSITIVE`. Hai tầng làm giàu chỉ dùng để huấn luyện; năm outer holdout cộng lại gồm
656 dòng với 59 `NEGATIVE`.

| Chỉ số | Trung bình 5 fold | Độ lệch chuẩn |
| --- | ---: | ---: |
| Macro-F1 | 0.729764 | 0.042252 |
| Accuracy | 0.812457 | 0.020328 |
| Balanced accuracy | 0.768920 | 0.049668 |

| Lớp | F1 trung bình | Recall trung bình | Số mẫu outer holdout |
| --- | ---: | ---: | ---: |
| `NEGATIVE` | 0.603609 | 0.680303 | 59 |
| `NEUTRAL` | 0.881565 | 0.835858 | 463 |
| `POSITIVE` | 0.704119 | 0.790598 | 134 |

Lần chạy này đánh giá chất lượng bộ phân loại, không chọn checkpoint bằng chỉ số
outer holdout và chưa sinh lại đặc trưng cảm xúc cho nhánh dự báo giá. Bước kế tiếp là
tinh chỉnh lại trên toàn bộ tập với quy tắc validation-only hoặc tổ hợp năm checkpoint,
chấm lại kho tin theo mốc cắt thông tin, rồi chạy lại dự báo và đối chứng.

## Thực nghiệm lịch sử trên 306 mẫu

Lần chạy không trọng số và lần chạy có trọng số kiểm tra ảnh hưởng của ba dạng đầu vào và ba chiến lược giữ token của PhoBERT trên tập nhãn tin tức tài chính Vietstock đã được rà soát. Ma trận gồm:

- dạng đầu vào: `title`, `context`, `title_context`;
- chiến lược cắt: `head`, `tail`, `head_tail`;
- năm fold xác thực chéo phân tầng;
- ba và năm epoch cho mỗi fold;
- hạt giống: hạt giống gốc `42`, mỗi fold dùng `42 + chỉ số fold`, nên fold 1 đến fold 5
  của mỗi cấu hình dùng lần lượt `43, 44, 45, 46, 47` (đọc từ
  `fold-*/manifest.json`). Lần chạy không trọng số 3 epoch ghi `seed = 42` ở cấp
  cấu hình. Checkpoint đang dùng cho suy luận là fold 1, hạt giống `43`.

Mục tiêu của lần chạy không trọng số là kiểm tra quy trình và tạo đường cơ sở. Hai lần chạy có trọng số cho thấy cải thiện rõ ở Macro-F1 và balanced accuracy, nhưng vẫn chưa đủ điều kiện để chọn mô hình cảm xúc cuối cùng vì lớp `NEGATIVE` còn được nhận diện rất yếu.

## 2. Bằng chứng đầu vào và khả năng tái lập

Thông tin dưới đây được đọc trực tiếp từ các tệp `cv_results*.json` trong thư mục `outputs/`:

| Thành phần          | Giá trị                                                            |
| ------------------- | ------------------------------------------------------------------ |
| Mô hình nền         | `vinai/phobert-base`                                               |
| Số mẫu đầy đủ       | 306                                                                |
| Thứ tự nhãn         | `0 = NEGATIVE`, `1 = NEUTRAL`, `2 = POSITIVE`                      |
| Phân bố nhãn đầy đủ | `NEGATIVE=31`, `NEUTRAL=204`, `POSITIVE=71`                        |
| Số fold             | 5                                                                  |
| Số epoch            | 3                                                                  |
| Kích thước lô       | 16                                                                 |
| Độ dài tối đa       | 256 token                                                          |
| Tốc độ học          | `2e-5`                                                             |
| Suy giảm trọng số   | `0.01`                                                             |
| Tỷ lệ khởi động     | `0.1`                                                              |
| Hạt giống           | `42`                                                               |
| Cân bằng lớp        | `none`                                                             |
| Python              | `3.12.13`                                                          |
| NumPy               | `2.0.2`                                                            |
| Transformers        | `5.15.1`                                                           |
| PyTorch             | `2.10.0+cu128`                                                     |
| Mã băm tệp nhãn     | `e988f9cf41b4291a112d6299cd9daa6e6e7fb25b3e281956387629c8de99b643` |

Mã băm nguồn nhãn giống nhau trong các tệp kết quả, nên các cấu hình đã dùng cùng một phiên bản tệp dữ liệu theo bằng chứng được lưu trong kết quả.

Quy trình xác thực chéo được cài đặt tại [`src/stf/sentiment/experiments.py`](../src/stf/sentiment/experiments.py), còn thứ tự nhãn được cố định tại [`src/stf/sentiment/labels.py`](../src/stf/sentiment/labels.py). Tập outer holdout chỉ dùng để đánh giá; một phần của outer train được dùng làm validation để chọn checkpoint.

## 3. Ánh xạ tệp kết quả

Hai tệp không có hậu tố là kết quả chạy riêng cấu hình `title_context + head_tail`:

- [`outputs/cv_results.json`](../outputs/cv_results.json)
- [`outputs/cv_results.csv`](../outputs/cv_results.csv)

Chín cặp tệp có hậu tố là kết quả của ma trận ablation:

| Tệp JSON/CSV     | Dạng đầu vào    | Chiến lược cắt |
| ---------------- | --------------- | -------------- |
| `cv_results (1)` | `context`       | `head`         |
| `cv_results (2)` | `context`       | `head_tail`    |
| `cv_results (3)` | `context`       | `tail`         |
| `cv_results (4)` | `title`         | `head`         |
| `cv_results (5)` | `title`         | `head_tail`    |
| `cv_results (6)` | `title`         | `tail`         |
| `cv_results (7)` | `title_context` | `head`         |
| `cv_results (8)` | `title_context` | `head_tail`    |
| `cv_results (9)` | `title_context` | `tail`         |

`cv_results.json` và `cv_results (8).json` là cùng một kết quả. Hai tệp CSV tương ứng cũng là bản sao. Vì vậy, khi tổng hợp số liệu, chỉ tính một trong hai bản.

Bảng xếp hạng đầy đủ nằm tại [`outputs/ablation_summary.csv`](../outputs/ablation_summary.csv).

Kết quả có trọng số được lưu trong hai tệp nén:

- `outputs/title_context__head_tail__cw-inverse_frequency__e5.zip`: chạy riêng cấu hình `title_context + head_tail`, có đầy đủ `fold-*/best/`.
- `outputs/ablation__cw-inverse_frequency__e5.zip`: ma trận chín cấu hình, chỉ lưu số liệu và manifest.

## 4. Kết quả tổng hợp

### 4.1. Các cấu hình có nội dung ngữ cảnh

Sáu cấu hình `context` và `title_context` đều có cùng số liệu tổng hợp:

| Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
| -----: | ------------------: | ------------: | ------------------: | ----------------: |
|    306 |            0.266667 |      0.002923 |            0.666737 |          0.333333 |

Ba chiến lược cắt cho cùng một con số. Nguyên nhân là cơ học, không phải bằng chứng
tương đương: xem mục 4.3.

### 4.2. Các cấu hình chỉ dùng tiêu đề

Ba cấu hình `title` đều có cùng số liệu tổng hợp:

| Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
| -----: | ------------------: | ------------: | ------------------: | ----------------: |
|    301 |            0.265333 |      0.002981 |            0.661202 |          0.333333 |

Khi tạo đầu vào, mã chọn văn bản cuối rồi khử trùng lặp trên chính văn bản đó. Vì vậy, các cấu hình chỉ dùng tiêu đề được ghi nhận với 301 mẫu, trong khi các cấu hình có ngữ cảnh được ghi nhận với 306 mẫu. Hai nhóm không được đánh giá trên cùng số lượng mẫu, do đó không nên diễn giải chênh lệch nhỏ giữa chúng như một ưu thế chắc chắn của dạng đầu vào.

### 4.3. Chiến lược cắt token không được kích hoạt trên tệp nhãn này

Ba chiến lược `head`, `tail`, `head_tail` cho kết quả trùng nhau đến từng chữ số vì
chúng **không bao giờ được gọi** trên tệp nhãn hiện dùng. Đo bằng chính tokenizer của
checkpoint (`models/sentiment/selected`) trên 306 mẫu của `to_label_r1.csv`:

| Dạng đầu vào    | Trung vị | Phân vị 90 | Tối đa | Số mẫu > 256 token |
| --------------- | -------: | ---------: | -----: | -----------------: |
| `title`         |       22 |         32 |     71 |              0/306 |
| `context`       |      109 |        132 |    206 |              0/306 |
| `title_context` |      130 |        160 |    229 |              0/306 |

Không một mẫu nào đạt tới giới hạn 256 token, nên ba chiến lược tạo ra cùng một chuỗi
token đầu vào và bắt buộc cho cùng một kết quả. Nguyên nhân gốc là trường `body_preview`
trong `to_label_r1.csv` bị chặn ở 400 ký tự khi tệp được sinh; `make_indomain_sample.py`
hiện chặn ở 2000 ký tự, nên một tệp sinh lại sẽ không còn tính chất này.

Vì vậy, **không được báo cáo lần chạy này như một so sánh giữa các chiến lược cắt**. Kết
luận đúng là: ở độ dài văn bản của tệp nhãn hiện tại, việc cắt không phát sinh, nên thí
nghiệm không có khả năng phân biệt ba chiến lược. Muốn kiểm định thật, phải gán nhãn lại
trên nội dung bài đầy đủ; khi nối `to_label_r1.url` với `articles.parquet`, 296/306 mẫu
có nội dung đầy đủ và khoảng **19% vượt 256 từ** — số token BPE luôn lớn hơn hoặc bằng số
từ, nên tỷ lệ vượt 256 *token* còn cao hơn 19%. Như vậy chiến lược cắt sẽ thực sự có hiệu
lực trên tệp gán nhãn lại, nhưng cỡ hiệu ứng dự kiến vẫn nhỏ.

### 4.3.1. Sàn nhiễu của lần chạy có trọng số

Trong lần chạy 5 epoch, `context + head` và `context + head_tail` cho `0.495236`, còn
`context + tail` cho `0.493463`. Vì đầu vào token là như nhau và hạt giống mỗi fold cũng
như nhau giữa các cấu hình, khác biệt `0.0018` này **không thể** do chiến lược cắt. Đây là
dao động không tất định giữa các lần chạy trên GPU. Con số đó là **sàn nhiễu đo được của
quy trình, khoảng `0.002` Macro-F1**, và mọi chênh lệch nhỏ hơn mức này không được diễn
giải là khác biệt thực.

`ablation_summary.csv` đặt `context + tail` ở dòng đầu chỉ vì cần một thứ tự phá hòa khi
mọi cấu hình có cùng `macro_f1_mean`. Đây không phải bằng chứng rằng cấu hình đó tốt hơn.

### 4.4. Kết quả có trọng số lớp — 3 epoch

Lần chạy này dùng `--class-weighting inverse_frequency`, với trọng số được tính riêng từ phần huấn luyện của từng fold. Các cấu hình đều dùng 5 fold, 3 epoch, kích thước lô 16 và hạt giống `42`.

| Dạng đầu vào | Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `title_context` | 306 | 0.506284 | 0.058599 | 0.741724 | 0.544866 |
| `title` | 301 | 0.500477 | 0.074720 | 0.690710 | 0.516532 |
| `context + head` | 306 | 0.480635 | 0.076234 | 0.718826 | 0.514628 |
| `context + head_tail` | 306 | 0.480635 | 0.076234 | 0.718826 | 0.514628 |
| `context + tail` | 306 | 0.476660 | 0.069313 | 0.715547 | 0.509866 |

Ba chiến lược cắt token của `title_context` cho cùng kết quả trong lần chạy này. Cấu hình `title_context + head_tail` được giữ làm cấu hình đại diện vì đã có checkpoint tốt nhất cho từng fold.

### 4.5. Kết quả có trọng số lớp — 5 epoch

Lần chạy này giữ nguyên dữ liệu, hạt giống và cơ chế tính trọng số của lần chạy 3 epoch, chỉ tăng số epoch từ 3 lên 5.

| Dạng đầu vào | Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `title` | 301 | 0.529995 | 0.096751 | 0.724044 | 0.535977 |
| `title_context + head` | 306 | 0.517616 | 0.070196 | 0.735325 | 0.547288 |
| `title_context + head_tail` | 306 | 0.517616 | 0.070196 | 0.735325 | 0.547288 |
| `title_context + tail` | 306 | 0.516692 | 0.071090 | 0.732047 | 0.545662 |
| `context + head` | 306 | 0.495236 | 0.063874 | 0.732047 | 0.533041 |
| `context + head_tail` | 306 | 0.495236 | 0.063874 | 0.732047 | 0.533041 |
| `context + tail` | 306 | 0.493463 | 0.062131 | 0.725595 | 0.529232 |

So với lần chạy 3 epoch, cấu hình đại diện `title_context + head_tail` chỉ tăng nhẹ Macro-F1 từ `0.506284` lên `0.517616` và balanced accuracy từ `0.544866` lên `0.547288`. `title` có Macro-F1 cao hơn, nhưng hai dạng đầu vào dùng số lượng mẫu khác nhau nên không được so sánh như một kết luận chắc chắn.

## 5. Phân tích theo lớp của baseline không trọng số

Các tệp CSV cho thấy cùng một kiểu dự đoán ở tất cả các cấu hình:

| Lớp        |     F1 quan sát được | Recall quan sát được |
| ---------- | -------------------: | -------------------: |
| `NEGATIVE` |                  0.0 |                  0.0 |
| `NEUTRAL`  | khoảng 0.78 đến 0.80 |                  1.0 |
| `POSITIVE` |                  0.0 |                  0.0 |

Mô hình đang dự đoán toàn bộ mẫu kiểm tra là `NEUTRAL`. Kết luận này phù hợp với ba dấu hiệu độc lập:

1. `balanced_accuracy` bằng đúng `0.333333` ở mọi fold, tương ứng với mức chỉ nhận diện một lớp trong bài toán ba lớp.
2. F1 và recall của `NEGATIVE` và `POSITIVE` bằng không ở mọi cấu hình.
3. `accuracy` khoảng 0.66, gần với tỷ lệ lớp `NEUTRAL` trong dữ liệu.

Do đó, `accuracy` hiện tại không thể được dùng làm chỉ số đại diện cho chất lượng mô hình. Chỉ số cần ưu tiên là `macro_f1`, `balanced_accuracy` và F1/recall từng lớp.

### 5.2. Phân tích lần chạy có trọng số — 3 epoch

Ở cấu hình đại diện `title_context + head_tail`, F1 và recall trung bình qua 5 fold là:

| Lớp | F1 trung bình | Recall trung bình |
| --- | ---: | ---: |
| `NEGATIVE` | 0.0444 | 0.0333 |
| `NEUTRAL` | 0.8550 | 0.8432 |
| `POSITIVE` | 0.6194 | 0.7581 |

So với baseline, hai lớp thiểu số không còn bị bỏ qua hoàn toàn. Tuy nhiên, recall của `NEGATIVE` vẫn quá thấp để dùng mô hình cho suy luận toàn bộ kho tin.

### 5.3. Phân tích lần chạy có trọng số — 5 epoch

Ở cấu hình đại diện `title_context + head_tail`, F1 và recall trung bình qua 5 fold là:

| Lớp | F1 trung bình | Recall trung bình |
| --- | ---: | ---: |
| `NEGATIVE` | 0.0767 | 0.0619 |
| `NEUTRAL` | 0.8625 | 0.8333 |
| `POSITIVE` | 0.6136 | 0.7467 |

Recall `NEGATIVE` theo từng fold là `0.1429`, `0.0000`, `0.0000`, `0.0000`, `0.1667`; ba trên năm fold không nhận diện đúng mẫu `NEGATIVE` nào. Vì chỉ có 31 mẫu `NEGATIVE`, một mẫu đúng hoặc sai đã làm recall của một fold thay đổi khoảng 16–17 điểm phần trăm.

## 6. Đánh giá trạng thái thực nghiệm

### Đã xác nhận

- Notebook đã chạy đủ ma trận chín cấu hình.
- Mỗi cấu hình đã hoàn thành năm fold.
- Kết quả JSON và CSV có đầy đủ số liệu tổng hợp.
- Tất cả cấu hình dùng cùng hạt giống, siêu tham số và mã băm tệp nhãn.
- Chiến lược cắt token không được kích hoạt trên tệp nhãn này (mục 4.3), nên lần chạy
  không phân biệt được ba chiến lược.
- Cơ chế lưu trữ mới không làm ma trận ablation đầy đĩa.
- Lần chạy 5 epoch đã hoàn thành đủ chín cấu hình và lần chạy riêng cấu hình đại diện đã lưu các thư mục `fold-*/best/`.

### Chưa được xác nhận

- Chưa có cấu hình nào cho thấy khả năng phân loại cân bằng cả ba lớp; ở cấu hình đại diện 5 epoch, lớp `NEGATIVE` vẫn có recall trung bình chỉ `0.0619` và bằng 0 ở ba trên năm fold.
- Chưa thể kết luận `context` tốt hơn `title`, vì số lượng mẫu sau chuẩn bị đầu vào khác nhau.
- Chưa có bằng chứng cho phép suy rộng chất lượng cảm xúc ra ngoài phạm vi 306 mẫu này.

## 7. Quyết định nghiên cứu

**Chốt cấu hình cảm xúc để phục vụ nhánh dự báo giá, và ghi rõ giới hạn của nó.** Lần chạy
5 epoch chỉ cải thiện nhẹ so với 3 epoch:

- Cấu hình `title_context + head_tail`: Macro-F1 từ `0.506284` lên `0.517616`;
- Balanced accuracy từ `0.544866` lên `0.547288`;
- F1 `NEGATIVE` từ `0.0444` lên `0.0767`;
- Recall `NEGATIVE` đạt `0.0619`, nhưng bằng 0 ở ba trên năm fold.

Tập dữ liệu hiện có 306 mẫu, gồm `31 NEGATIVE`, `204 NEUTRAL` và `71 POSITIVE`. Với 5 fold,
mỗi fold chỉ có khoảng sáu mẫu `NEGATIVE`, nên các ước lượng recall của lớp này có độ biến
động rất lớn. Tăng thêm epoch không giải quyết được giới hạn này.

### 7.1. Checkpoint được chọn và quy tắc chọn

Nhánh dự báo giá cần một bộ sinh xác suất cảm xúc để trả lời câu hỏi trung tâm của đồ án.
Vì vậy checkpoint dùng cho suy luận được chọn từ lần chạy có trọng số 5 epoch của cấu hình
`title_context + head_tail`, theo quy tắc **fold có Macro-F1 gần nhất với trung bình xác
thực chéo**:

| Fold | Macro-F1 | Lệch so với trung bình `0.517616` |
| ---: | -------: | --------------------------------: |
|    1 | 0.553464 |                        **0.035848** |
|    2 | 0.464726 |                          0.052890 |
|    3 | 0.559429 |                          0.041813 |
|    4 | 0.422852 |                          0.094764 |
|    5 | 0.587607 |                          0.069991 |

Fold 1 được chọn và giải nén vào `models/sentiment/selected/`. Quy tắc này cố ý **không**
chọn fold có điểm cao nhất (fold 5): chọn theo điểm cao nhất trên tập holdout là chọn trên
tập kiểm tra và sẽ cho một ước lượng lạc quan. Fold đại diện cho hiệu năng trung bình là
lựa chọn không chệch hơn.

### 7.2. Khớp độ dài đầu vào giữa huấn luyện và suy luận

Checkpoint được huấn luyện trên `title` cộng `body_preview` bị chặn 400 ký tự, trong khi
`articles.parquet` giữ nội dung đầy đủ (trung vị 429 ký tự, phân vị 90 là 2218 ký tự). Nếu
suy luận trên nội dung đầy đủ, đầu vào sẽ dài hơn hẳn miền huấn luyện. Do đó `score-news`
được chạy với `--context-chars 400` để giữ độ dài ngữ cảnh trong đúng miền của checkpoint.

### 7.3. Giới hạn phải nêu kèm mọi kết quả dùng đặc trưng cảm xúc

- Recall `NEGATIVE` là `0.0619`; tin xấu bị nhận diện rất yếu.
- Đặc trưng đưa vào nhánh dự báo là **phân phối xác suất mềm**, không phải nhãn cứng, nên
  nhiễu ở lớp thiểu số làm suy giảm tín hiệu thay vì tạo nhãn sai dứt khoát.
- Mọi kết luận về đóng góp của cảm xúc phải được phát biểu kèm chất lượng bộ sinh cảm xúc
  này; một kết quả "không cải thiện" có thể do tín hiệu yếu, do bộ sinh yếu, hoặc cả hai.
  Đồ án không được quy kết nguyên nhân khi chưa có bằng chứng tách bạch.

### 7.4. Việc còn lại để nâng chất lượng cảm xúc

Mở rộng tập nhãn, đặc biệt lớp `NEGATIVE`, và bổ sung tập hạt giống CafeF (999 mẫu,
`564 POSITIVE / 249 NEUTRAL / 186 NEGATIVE`) vào **phần huấn luyện của từng fold**, giữ 306
mẫu in-domain làm tập đánh giá duy nhất. Nếu gộp thẳng thành một tập 1.305 mẫu rồi chia
5-fold, phần lớn mẫu kiểm tra sẽ là tiêu đề CafeF và con số Macro-F1 sẽ bị lạc quan. CafeF
chỉ có tiêu đề nên phải chạy ở dạng `title`, hoặc ghi rõ là đầu vào trộn độ dài. Ngoài ra,
nhãn CafeF mô tả sắc thái tiêu đề, còn hướng dẫn gán nhãn của đồ án mô tả tác động giá kỳ
vọng ngắn hạn; nếu dùng, phải nêu đây là một giả định chuyển miền.

## 8. Tệp cần lưu trữ

Để tái kiểm tra số liệu, giữ:

- `ablation_summary.csv`;
- một cặp `cv_results.json` và `cv_results.csv` cho mỗi cấu hình;
- hai tệp nén của lần chạy có trọng số;
- bản notebook đã chạy;
- mã nguồn ở nhánh có cơ chế lưu checkpoint an toàn;
- tệp nhãn gốc hoặc mã băm của tệp nhãn.

Tệp `title_context__head_tail__cw-inverse_frequency__e5.zip` đã chứa các thư mục `fold-*/best/` dùng cho suy luận thử nghiệm. Tệp `ablation__cw-inverse_frequency__e5.zip` không chứa mô hình tốt nhất vì ma trận ablation chỉ lưu số liệu.

Các tệp nén của lần chạy 3 epoch đã được thay thế sau khi lưu kết quả mới; số liệu 3 epoch vẫn được ghi lại trong các bảng so sánh ở trên.

## 9. Tài liệu và mã thực thi liên quan

- Quy trình chạy notebook: [`notebooks/training-guide.md`](../notebooks/training-guide.md)
- Ghi chú notebook: [`notebooks/README.md`](../notebooks/README.md)
- Lệnh và giao diện thực thi: [`src/stf/cli.py`](../src/stf/cli.py)
- Chuẩn bị dữ liệu và khử trùng lặp: [`src/stf/sentiment/dataset.py`](../src/stf/sentiment/dataset.py)
- Xác thực chéo và ablation: [`src/stf/sentiment/experiments.py`](../src/stf/sentiment/experiments.py)
- Huấn luyện và lưu mô hình: [`src/stf/sentiment/model.py`](../src/stf/sentiment/model.py)
