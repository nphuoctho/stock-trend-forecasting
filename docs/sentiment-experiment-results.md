# Bản ghi thực nghiệm phân loại cảm xúc

> **Trạng thái:** Bản ghi bằng chứng cho một lần chạy trên Kaggle, không phải kết quả cuối cùng để đưa vào báo cáo.
>
> **Nguồn số liệu:** Các tệp JSON/CSV trong [`../outputs/`](../outputs/), được tải về sau khi chạy `sentiment-cv` và `sentiment-ablation`.

## 1. Mục đích và phạm vi

Lần chạy này kiểm tra ảnh hưởng của ba dạng đầu vào và ba chiến lược giữ token của PhoBERT trên tập nhãn tin tức tài chính Vietstock đã được rà soát. Ma trận gồm:

- dạng đầu vào: `title`, `context`, `title_context`;
- chiến lược cắt: `head`, `tail`, `head_tail`;
- năm fold xác thực chéo phân tầng;
- ba epoch cho mỗi fold;
- hạt giống ngẫu nhiên `42`.

Mục tiêu của lần chạy là kiểm tra quy trình, tạo số liệu so sánh ban đầu và phát hiện cấu hình phù hợp. Lần chạy **chưa đủ điều kiện để chọn mô hình cảm xúc cuối cùng**, vì mô hình không nhận diện được hai lớp thiểu số.

## 2. Bằng chứng đầu vào và khả năng tái lập

Thông tin dưới đây được đọc trực tiếp từ các tệp `cv_results*.json` trong thư mục `outputs/`:

| Thành phần | Giá trị |
|---|---|
| Mô hình nền | `vinai/phobert-base` |
| Số mẫu đầy đủ | 306 |
| Thứ tự nhãn | `0 = NEGATIVE`, `1 = NEUTRAL`, `2 = POSITIVE` |
| Phân bố nhãn đầy đủ | `NEGATIVE=31`, `NEUTRAL=204`, `POSITIVE=71` |
| Số fold | 5 |
| Số epoch | 3 |
| Kích thước lô | 16 |
| Độ dài tối đa | 256 token |
| Tốc độ học | `2e-5` |
| Suy giảm trọng số | `0.01` |
| Tỷ lệ khởi động | `0.1` |
| Hạt giống | `42` |
| Cân bằng lớp | `none` |
| Python | `3.12.13` |
| NumPy | `2.0.2` |
| Transformers | `5.15.1` |
| PyTorch | `2.10.0+cu128` |
| Mã băm tệp nhãn | `e988f9cf41b4291a112d6299cd9daa6e6e7fb25b3e281956387629c8de99b643` |

Mã băm nguồn nhãn giống nhau trong các tệp kết quả, nên các cấu hình đã dùng cùng một phiên bản tệp dữ liệu theo bằng chứng được lưu trong kết quả.

Quy trình xác thực chéo được cài đặt tại [`src/stf/sentiment/experiments.py`](../src/stf/sentiment/experiments.py), còn thứ tự nhãn được cố định tại [`src/stf/sentiment/labels.py`](../src/stf/sentiment/labels.py). Tập outer holdout chỉ dùng để đánh giá; một phần của outer train được dùng làm validation để chọn checkpoint.

## 3. Ánh xạ tệp kết quả

Hai tệp không có hậu tố là kết quả chạy riêng cấu hình `title_context + head_tail`:

- [`outputs/cv_results.json`](../outputs/cv_results.json)
- [`outputs/cv_results.csv`](../outputs/cv_results.csv)

Chín cặp tệp có hậu tố là kết quả của ma trận ablation:

| Tệp JSON/CSV | Dạng đầu vào | Chiến lược cắt |
|---|---|---|
| `cv_results (1)` | `context` | `head` |
| `cv_results (2)` | `context` | `head_tail` |
| `cv_results (3)` | `context` | `tail` |
| `cv_results (4)` | `title` | `head` |
| `cv_results (5)` | `title` | `head_tail` |
| `cv_results (6)` | `title` | `tail` |
| `cv_results (7)` | `title_context` | `head` |
| `cv_results (8)` | `title_context` | `head_tail` |
| `cv_results (9)` | `title_context` | `tail` |

`cv_results.json` và `cv_results (8).json` là cùng một kết quả. Hai tệp CSV tương ứng cũng là bản sao. Vì vậy, khi tổng hợp số liệu, chỉ tính một trong hai bản.

Bảng xếp hạng đầy đủ nằm tại [`outputs/ablation_summary.csv`](../outputs/ablation_summary.csv).

## 4. Kết quả tổng hợp

### 4.1. Các cấu hình có nội dung ngữ cảnh

Sáu cấu hình `context` và `title_context` đều có cùng số liệu tổng hợp:

| Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
|---:|---:|---:|---:|---:|
| 306 | 0.266667 | 0.002923 | 0.666737 | 0.333333 |

Không có bằng chứng cho thấy `head`, `tail` hoặc `head_tail` tốt hơn trong nhóm này.

### 4.2. Các cấu hình chỉ dùng tiêu đề

Ba cấu hình `title` đều có cùng số liệu tổng hợp:

| Số mẫu | Macro-F1 trung bình | Độ lệch chuẩn | Accuracy trung bình | Balanced accuracy |
|---:|---:|---:|---:|---:|
| 301 | 0.265333 | 0.002981 | 0.661202 | 0.333333 |

Khi tạo đầu vào, mã chọn văn bản cuối rồi khử trùng lặp trên chính văn bản đó. Vì vậy, các cấu hình chỉ dùng tiêu đề được ghi nhận với 301 mẫu, trong khi các cấu hình có ngữ cảnh được ghi nhận với 306 mẫu. Hai nhóm không được đánh giá trên cùng số lượng mẫu, do đó không nên diễn giải chênh lệch nhỏ giữa chúng như một ưu thế chắc chắn của dạng đầu vào.

### 4.3. Không chọn cấu hình đứng đầu một cách máy móc

`ablation_summary.csv` đặt `context + tail` ở dòng đầu vì các cấu hình có cùng `macro_f1_mean` và việc sắp xếp cần một thứ tự phá hòa. Đây không phải bằng chứng rằng `context + tail` tốt hơn `context + head`, `context + head_tail` hoặc các cấu hình `title_context`.

## 5. Phân tích theo lớp

Các tệp CSV cho thấy cùng một kiểu dự đoán ở tất cả các cấu hình:

| Lớp | F1 quan sát được | Recall quan sát được |
|---|---:|---:|
| `NEGATIVE` | 0.0 | 0.0 |
| `NEUTRAL` | khoảng 0.78 đến 0.80 | 1.0 |
| `POSITIVE` | 0.0 | 0.0 |

Mô hình đang dự đoán toàn bộ mẫu kiểm tra là `NEUTRAL`. Kết luận này phù hợp với ba dấu hiệu độc lập:

1. `balanced_accuracy` bằng đúng `0.333333` ở mọi fold, tương ứng với mức chỉ nhận diện một lớp trong bài toán ba lớp.
2. F1 và recall của `NEGATIVE` và `POSITIVE` bằng không ở mọi cấu hình.
3. `accuracy` khoảng 0.66, gần với tỷ lệ lớp `NEUTRAL` trong dữ liệu.

Do đó, `accuracy` hiện tại không thể được dùng làm chỉ số đại diện cho chất lượng mô hình. Chỉ số cần ưu tiên là `macro_f1`, `balanced_accuracy` và F1/recall từng lớp.

## 6. Đánh giá trạng thái thực nghiệm

### Đã xác nhận

- Notebook đã chạy đủ ma trận chín cấu hình.
- Mỗi cấu hình đã hoàn thành năm fold.
- Kết quả JSON và CSV có đầy đủ số liệu tổng hợp.
- Tất cả cấu hình dùng cùng hạt giống, siêu tham số và mã băm tệp nhãn.
- Các chiến lược cắt token không tạo khác biệt quan sát được trong lần chạy này.
- Cơ chế lưu trữ mới không làm ma trận ablation đầy đĩa.

### Chưa được xác nhận

- Chưa có cấu hình nào cho thấy khả năng phân loại thực sự cả ba lớp.
- Chưa thể kết luận `context` tốt hơn `title`, vì số lượng mẫu sau chuẩn bị đầu vào khác nhau.
- Chưa nên dùng các kết quả này để sinh đặc trưng cảm xúc cho nhánh dự báo giá.
- Chưa có mô hình cuối được chọn cho suy luận toàn bộ kho tin.

## 7. Quyết định nghiên cứu

Không chọn mô hình cuối từ ma trận này. Đây là kết quả nền không dùng trọng số lớp, và nó cho thấy mất cân bằng nhãn đang chi phối quá trình học.

Bước thực nghiệm tiếp theo cần dùng tùy chọn:

```text
--class-weighting inverse_frequency
```

Trọng số phải được tính riêng từ phần huấn luyện của từng fold. Không được dùng outer holdout để tính trọng số hoặc điều chỉnh dữ liệu đánh giá.

Nên chạy trước một cấu hình đại diện, chẳng hạn `title_context + head_tail`, với trọng số nghịch đảo tần suất. Chỉ tiếp tục chạy ma trận đầy đủ nếu F1 và recall của cả `NEGATIVE` và `POSITIVE` tăng lên rõ ràng.

Sau khi chọn được cấu hình từ ma trận có trọng số, chạy lại `sentiment-cv` riêng cho cấu hình đó để giữ các thư mục `fold-*/best/` dùng cho `score-news`. Ma trận `sentiment-ablation` chỉ lưu số liệu và bản kê nguồn nhằm giới hạn dung lượng lưu trữ.

## 8. Tệp cần lưu trữ

Để tái kiểm tra số liệu, giữ:

- `ablation_summary.csv`;
- một cặp `cv_results.json` và `cv_results.csv` cho mỗi cấu hình;
- bản notebook đã chạy;
- mã nguồn ở nhánh có cơ chế lưu checkpoint an toàn;
- tệp nhãn gốc hoặc mã băm của tệp nhãn.

Để chạy suy luận, cần thêm các thư mục `fold-*/best/` của cấu hình được chọn. Các thư mục này không được tạo bởi ma trận ablation hiện tại; chúng phải được tạo bằng một lần chạy `sentiment-cv` riêng.

## 9. Tài liệu và mã thực thi liên quan

- Quy trình chạy notebook: [`notebooks/training-guide.md`](../notebooks/training-guide.md)
- Ghi chú notebook: [`notebooks/README.md`](../notebooks/README.md)
- Lệnh và giao diện thực thi: [`src/stf/cli.py`](../src/stf/cli.py)
- Chuẩn bị dữ liệu và khử trùng lặp: [`src/stf/sentiment/dataset.py`](../src/stf/sentiment/dataset.py)
- Xác thực chéo và ablation: [`src/stf/sentiment/experiments.py`](../src/stf/sentiment/experiments.py)
- Huấn luyện và lưu mô hình: [`src/stf/sentiment/model.py`](../src/stf/sentiment/model.py)
