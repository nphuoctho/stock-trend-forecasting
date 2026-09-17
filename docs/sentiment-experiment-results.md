# Bản ghi thực nghiệm phân loại cảm xúc

> **Trạng thái:** Bản ghi bằng chứng cho ba lần chạy trên Kaggle, chưa phải kết quả cuối cùng để đưa vào báo cáo.
>
> **Nguồn số liệu:** Các tệp JSON/CSV và tệp nén trong [`../outputs/`](../outputs/), được tải về sau khi chạy `sentiment-cv` và `sentiment-ablation`.

## 1. Mục đích và phạm vi

Lần chạy không trọng số và lần chạy có trọng số kiểm tra ảnh hưởng của ba dạng đầu vào và ba chiến lược giữ token của PhoBERT trên tập nhãn tin tức tài chính Vietstock đã được rà soát. Ma trận gồm:

- dạng đầu vào: `title`, `context`, `title_context`;
- chiến lược cắt: `head`, `tail`, `head_tail`;
- năm fold xác thực chéo phân tầng;
- ba và năm epoch cho mỗi fold;
- hạt giống ngẫu nhiên `42`.

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

Không có bằng chứng cho thấy `head`, `tail` hoặc `head_tail` tốt hơn trong nhóm này.

### 4.2. Các cấu hình chỉ dùng tiêu đề

Ba cấu hình `title` đều có cùng số liệu tổng hợp:

| Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
| -----: | ------------------: | ------------: | ------------------: | ----------------: |
|    301 |            0.265333 |      0.002981 |            0.661202 |          0.333333 |

Khi tạo đầu vào, mã chọn văn bản cuối rồi khử trùng lặp trên chính văn bản đó. Vì vậy, các cấu hình chỉ dùng tiêu đề được ghi nhận với 301 mẫu, trong khi các cấu hình có ngữ cảnh được ghi nhận với 306 mẫu. Hai nhóm không được đánh giá trên cùng số lượng mẫu, do đó không nên diễn giải chênh lệch nhỏ giữa chúng như một ưu thế chắc chắn của dạng đầu vào.

### 4.3. Không chọn cấu hình đứng đầu một cách máy móc

`ablation_summary.csv` đặt `context + tail` ở dòng đầu vì các cấu hình có cùng `macro_f1_mean` và việc sắp xếp cần một thứ tự phá hòa. Đây không phải bằng chứng rằng `context + tail` tốt hơn `context + head`, `context + head_tail` hoặc các cấu hình `title_context`.

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
- Các chiến lược cắt token không tạo khác biệt quan sát được trong lần chạy này.
- Cơ chế lưu trữ mới không làm ma trận ablation đầy đĩa.
- Lần chạy 5 epoch đã hoàn thành đủ chín cấu hình và lần chạy riêng cấu hình đại diện đã lưu các thư mục `fold-*/best/`.

### Chưa được xác nhận

- Chưa có cấu hình nào cho thấy khả năng phân loại cân bằng cả ba lớp; ở cấu hình đại diện 5 epoch, lớp `NEGATIVE` vẫn có recall trung bình chỉ `0.0619` và bằng 0 ở ba trên năm fold.
- Chưa thể kết luận `context` tốt hơn `title`, vì số lượng mẫu sau chuẩn bị đầu vào khác nhau.
- Chưa nên dùng các kết quả này để sinh đặc trưng cảm xúc cho nhánh dự báo giá.
- Chưa có mô hình cuối được chọn cho suy luận toàn bộ kho tin.

## 7. Quyết định nghiên cứu

Không chọn mô hình cuối để gán nhãn toàn bộ kho tin. Lần chạy 5 epoch chỉ cải thiện nhẹ so với 3 epoch:

- Cấu hình `title_context + head_tail`: Macro-F1 từ `0.506284` lên `0.517616`;
- Balanced accuracy từ `0.544866` lên `0.547288`;
- F1 `NEGATIVE` từ `0.0444` lên `0.0767`;
- Recall `NEGATIVE` đạt `0.0619`, nhưng bằng 0 ở ba trên năm fold.

Tập dữ liệu hiện có 306 mẫu, gồm `31 NEGATIVE`, `204 NEUTRAL` và `71 POSITIVE`. Với 5 fold, mỗi fold chỉ có khoảng sáu mẫu `NEGATIVE`, nên các ước lượng recall của lớp này có độ biến động rất lớn.

Ưu tiên tiếp theo là kiểm tra thủ công 31 mẫu `NEGATIVE` và mở rộng tập dữ liệu, đặc biệt bổ sung các mẫu `NEGATIVE` và `POSITIVE`. Không nên tiếp tục tăng epoch trước khi giải quyết giới hạn về số lượng và độ ổn định của nhãn.

Sau khi có thêm dữ liệu, chạy lại cấu hình `title_context + head_tail` với `inverse_frequency`, giữ nguyên hạt giống và quy trình xác thực chéo. Chỉ sau khi recall `NEGATIVE` ổn định hơn mới chọn mô hình để chạy `score-news`; các đặc trưng cảm xúc chưa được đưa vào nhánh dự báo giá ở thời điểm này.

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
