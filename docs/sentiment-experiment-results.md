# Bản ghi thực nghiệm phân loại cảm xúc

> **Trạng thái:** Xác thực chéo theo tầng trên 1.306 nhãn đã hoàn thành. Ba điểm kiểm
> triển khai đã được tinh chỉnh (`merged-refit` 1.306 nhãn hồi cứu; `point-in-time`
> 1.049 nhãn đóng băng trước ngày kiểm thử đầu tiên; `size-matched-s7` 1.049 nhãn
> đối chứng cùng kích thước, hạt giống 7), kho tin đã được chấm lại và các phép đối
> chứng dự báo ghép cặp đã chạy xong. Lượt dự báo được báo cáo chính thức là
> `forecast_merged_sentiment_tft` (gồm cả nhánh LSTM lẫn TFT); lượt
> `forecast_merged_sentiment` cũ hơn chỉ còn giá trị lịch sử. Các phần về 306 mẫu
> bên dưới chỉ là bản ghi lịch sử của đường cơ sở và ablation.
>
> **Nguồn số liệu hiện tại:** [`../outputs/sentiment-cv-merged/cv_results.json`](../outputs/sentiment-cv-merged/cv_results.json),
> `outputs/merged-refit.zip`, `outputs/point-in-time.zip`, `outputs/size-matched-s7.zip`,
> `outputs/forecast_merged_sentiment_tft/{forecast_results,information_gain,information_gain_tft}.json`,
> `outputs/forecast_pit_sentiment/{forecast_results,information_gain,information_gain_tft}.json`,
> `outputs/forecast_sizematched_s7_sentiment/{forecast_results,information_gain}.json`,
> `outputs/forecast_excess_{merged,pit}_sentiment/{forecast_results,information_gain}.json`,
> `outputs/forecast_{h3,h5}_pit_sentiment/{forecast_results,information_gain}.json`,
> `outputs/forecast_pit_sentiment/backtest_*.json`,
> `outputs/signal_ic_{pit,merged}.json`.

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

Xác thực chéo chỉ đánh giá bộ phân loại; nó không tự chứng minh tín hiệu cảm xúc cải thiện
dự báo giá. Cấu hình này được khóa trước khi tinh chỉnh checkpoint triển khai.

## Tinh chỉnh điểm kiểm triển khai, chấm lại tin và dự báo ghép cặp

Ba điểm kiểm `full_data_refit` được tinh chỉnh với cùng cấu hình đã khóa (5 epoch cố
định, trọng số nghịch đảo tần suất, hạt giống 42):

| Điểm kiểm | Số nhãn huấn luyện | Phân bố lớp | Vai trò |
| --- | ---: | --- | --- |
| `models/sentiment/merged-refit/` | 1.306 | 194 NEGATIVE / 771 NEUTRAL / 341 POSITIVE | Hồi cứu: học từ toàn bộ tập nhãn |
| `models/sentiment/point-in-time/` | 1.049 | 167 NEGATIVE / 616 NEUTRAL / 266 POSITIVE | Ngoài mẫu: `label_cutoff` 2024-10-21 (phiên kiểm thử đầu tiên là 2024-10-22) |
| `models/sentiment/size-matched-s7/` | 1.049 | 167 NEGATIVE / 616 NEUTRAL / 266 POSITIVE | Đối chứng: mẫu ngẫu nhiên khớp lớp cùng kích thước, hạt giống 7, không giới hạn thời điểm |

Nguồn: `models/sentiment/{merged-refit,point-in-time,size-matched-s7}/manifest.json`
(`training_size`, `class_distribution`, `provenance.label_cutoff`,
`provenance.subset.sample_seed`). Manifest của `merged-refit` xác nhận mã băm nguồn
`18d896d8c6b0e9badc91828030e205f2c523704bf38ed4fcde8a52a895eb0b56` và môi trường
`torch 2.10.0+cu128`, `transformers 5.15.1`; hai điểm kiểm còn lại dùng
`torch 2.13.0+cu130`, `transformers 5.15.1`.

Các điểm kiểm đã chấm lại kho tin bằng `title_context`, `head_tail` và
`context-chars 400`, neo theo mốc 15:00. Trong cửa sổ 2020–2025 có 12.624 liên kết
tin--mã được ánh xạ, còn 11 liên kết neo sau mốc cuối cửa sổ bị loại
(`alignment_report_in_window` của `outputs/forecast_pit_sentiment/forecast_results.json`).
Độ phủ tin trên bảng mã--phiên là 6.948/14.990 dòng = 46,35%
(`news_coverage` trong cùng tệp). Nhánh cảm xúc và đối chứng đều dùng đúng cùng
parquet đã chấm; đối chứng chỉ thay xác suất mỗi bài thành
`(NEGATIVE=0, NEUTRAL=1, POSITIVE=0)`, nên giữ nguyên thời điểm, số lượng tin và
cờ có tin.

### Hiệu năng của lượt được báo cáo chính thức

Lượt `forecast_merged_sentiment_tft` đánh giá cuốn chiếu năm cửa sổ (train/val/test =
expanding, test 60 phiên mỗi cửa sổ, ba hạt giống 42–44, nên mỗi ô là trung bình
15 lần chạy). Bảng dưới là trung bình F1 vĩ mô / độ chính xác cân bằng / độ chính
xác, đọc từ `outputs/forecast_merged_sentiment_tft/forecast_results.json`
(khoá `summary`) và `outputs/forecast_merged_control_tft/forecast_results.json`
cho cột đối chứng trung tính:

| Nhánh | Chỉ giá | Hai nhánh trung tính | Hai nhánh cảm xúc thật |
| --- | ---: | ---: | ---: |
| LSTM — F1 vĩ mô | 0.3756 | 0.3790 | 0.3869 |
| LSTM — độ chính xác cân bằng | 0.3915 | 0.3936 | 0.3953 |
| LSTM — độ chính xác | 0.4341 | 0.4349 | 0.4244 |
| TFT — F1 vĩ mô | 0.3619 | 0.3654 | 0.3597 |
| TFT — độ chính xác cân bằng | 0.3776 | 0.3778 | 0.3733 |
| TFT — độ chính xác | 0.4123 | 0.4104 | 0.3936 |

Các mốc tham chiếu trong cùng lượt: lớp phổ biến nhất F1 vĩ mô 0.1453; dự đoán
ngẫu nhiên 0.3289 (mức ngẫu nhiên ba lớp là 0.3333 theo `chance_level`); hồi quy
logistic chỉ giá 0.3551 và hồi quy logistic giá--cảm xúc 0.3667. Ở lượt point-in-time
(`outputs/forecast_pit_sentiment/forecast_results.json`), nhánh LSTM giá--cảm xúc đạt
F1 vĩ mô 0.3883; các nhánh chỉ giá giữ nguyên vì chúng không phụ thuộc điểm kiểm
cảm xúc.

### Đóng góp thông tin tại ba điểm kiểm cảm xúc

Đóng góp thông tin được định nghĩa là hiệu số F1 vĩ mô giữa nhánh cảm xúc thật và
nhánh hai nhánh trung tính (cùng kiến trúc, cùng cờ có tin và khối lượng tin).
Ước lượng điểm là trung bình năm hiệu số theo cửa sổ; khoảng tin cậy 95% theo ngày
lấy mẫu lặp bên trong từng cửa sổ (300 khối, phân tầng theo cửa sổ). Nguồn: khoá
`effects.macro_f1` của `information_gain.json` trong từng thư mục lượt:

| Điểm kiểm cảm xúc | Lượt | Đóng góp | KTC 95% theo ngày |
| --- | --- | ---: | --- |
| `merged-refit` 1.306 nhãn (hồi cứu) | `forecast_merged_sentiment_tft` | +0.0080 | [-0.0025; +0.0186] |
| `point-in-time` 1.049 nhãn (ngoài mẫu) | `forecast_pit_sentiment` | +0.0094 | [-0.0009; +0.0196] |
| `size-matched-s7` 1.049 nhãn (đối chứng) | `forecast_sizematched_s7_sentiment` | +0.0028 | [-0.0074; +0.0133] |

Cả ba khoảng đều chứa 0. Hiệu ứng kiến trúc hai nhánh cộng hiện diện/khối lượng tin
(trung tính trừ chỉ giá) là +0.0034, chung cho ba lượt vì cùng parquet đặc trưng và
cùng nhánh chỉ giá. Hiệu số trực tiếp (cảm xúc trừ chỉ giá) ở lượt hồi cứu là
+0.0113; giá trị này trộn lẫn hiệu ứng kiến trúc nên không được báo cáo như đóng
góp thông tin.

Với nhánh TFT, đóng góp thông tin tương ứng là -0.0057 [-0.0161; +0.0047] ở lượt
hồi cứu và -0.0105 [-0.0209; +0.0013] ở lượt point-in-time
(`information_gain_tft.json` của `forecast_merged_sentiment_tft` và
`forecast_pit_sentiment`).

### Phân tích bổ sung

**Tương quan hạng Spearman của điểm cảm xúc (không qua mô hình, khung thăm dò).**
Nguồn: `outputs/signal_ic_pit.json` và `outputs/signal_ic_merged.json`. Trên tập
điểm kiểm point-in-time, tương quan hạng của `sent_pos_minus_neg` với lợi suất
vượt trội cùng phiên là +0.0233 [+0.0004; +0.0465], còn với lợi suất vượt trội
phiên kế tiếp là +0.0030 [-0.0207; +0.0269]. IC ở lượt hồi cứu có cùng hướng nhưng
cả hai khoảng đều chứa 0 (+0.0206 [-0.0015; +0.0432] và +0.0049 [-0.0185; +0.0281]).
IC gần 0 ở phiên kế tiếp không phải trần thông tin: một phép đo đơn điệu theo hạng
có thể bằng 0 ngay cả khi tín hiệu dự báo được (ví dụ quan hệ phi tuyến dạng
`y = x^2` có tương quan hạng bằng 0 nhưng dự báo được hoàn hảo), nên phần này chỉ
mang tính thăm dò.

**Nhãn lợi suất vượt trội.** Nguồn:
`outputs/forecast_excess_{merged,pit}_sentiment/{forecast_results,information_gain}.json`.
Khi nhãn là lợi suất trừ trung bình đồng hạng cùng phiên (`target_mode: excess`),
mọi nhánh cho F1 vĩ mô thấp hơn so với nhãn thô: LSTM chỉ giá 0.3399 so với 0.3756.
Đóng góp thông tin là -0.0005 [-0.0091; +0.0080] ở lượt point-in-time và
+0.0031 [-0.0065; +0.0124] ở lượt hồi cứu; cả hai khoảng chứa 0.

**Ablation chân trời.** Nguồn:
`outputs/forecast_{h3,h5}_pit_sentiment/{forecast_results,information_gain}.json`
(cùng điểm kiểm point-in-time, `horizon` 3 và 5; h = 1 là lượt `forecast_pit_sentiment`).
Đóng góp thông tin F1 vĩ mô lần lượt là +0.0094; +0.0134 [+0.0020; +0.0249] và
-0.0007 [-0.0120; +0.0099] cho h = 1, 3, 5 — không đơn điệu. Ở h > 1 các nhãn chồng
lấn nên khoảng lấy mẫu lặp theo ngày hẹp hơn mức bằng chứng cho phép; không kết
luận có hiệu ứng.

**Chẩn đoán kinh tế.** Nguồn: `outputs/forecast_pit_sentiment/backtest_*.json`.
Chiến lược vào lệnh tại chính giá đóng cửa mà tín hiệu vừa dùng nên chỉ là chẩn
đoán lý tưởng hoá, không giao dịch được. Nhánh LSTM giá--cảm xúc có Sharpe trước
phí +3.44, sau phí 20 điểm cơ bản -1.12, phí hòa vốn 15.1 điểm cơ bản, alpha so
với rổ +0.0025 (t = +3.39); null hoán vị của Sharpe trước phí có trung bình +0.01
và độ lệch chuẩn 0.88. Mua-và-nắm-giữ rổ đều đạt Sharpe +2.61; 23,2% số phiên
danh mục chỉ có một chiều. Nhánh LSTM chỉ giá tương ứng: Sharpe trước phí +2.49,
sau phí -1.05, phí hòa vốn 14.1 điểm cơ bản.

**Giới hạn thời điểm:** `labeled_merged.csv` có bài năm 2025, trong khi cửa sổ kiểm
thử đầu tiên bắt đầu ngày 2024-10-22. Vì vậy điểm kiểm `merged-refit` đã học từ văn
bản và nhãn tương lai so với một phần ngày đánh giá, và lượt hồi cứu chỉ là đo
lường nhất quán giữa xác suất thật và prior trung tính, không phải bằng chứng dự
báo ngoài mẫu. Bằng chứng ngoài mẫu là lượt `forecast_pit_sentiment` (điểm kiểm
đóng băng theo `label_cutoff` 2024-10-21) đi kèm đối chứng cùng kích thước
`forecast_sizematched_s7_sentiment` để tách ảnh hưởng của việc giảm tập huấn luyện.

### Lượt hồi cứu cũ (giữ làm bản ghi lịch sử)

Lượt `forecast_merged_sentiment`/`forecast_merged_control` (chỉ có nhánh LSTM, tiền
TFT) đã được thay bằng `forecast_merged_sentiment_tft`/`forecast_merged_control_tft`.
Số liệu cũ của nó — LSTM chỉ giá 0.3797, trung tính 0.3841, cảm xúc thật 0.3907,
đóng góp +0.0066 — và nhận xét về 6.950 dòng có tin chỉ còn giá trị lịch sử; không
được dùng trong báo cáo. Các artifact forecast của lượt này có trước lược đồ
sidecar phiên bản 2 nên không được diễn giải là đã kiểm tra mã băm
parquet/điểm kiểm bởi sidecar.

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

## 3. Ánh xạ artifact lịch sử

Ánh xạ dưới đây được giữ để truy nguyên các bảng CV/ablation 306 nhãn. Các tệp tương ứng
không nằm trong working tree hiện tại, vì vậy tên được ghi như bản kê lịch sử, không phải
liên kết tái lập trực tiếp.

Hai tệp không có hậu tố là kết quả chạy riêng cấu hình `title_context + head_tail`:
`outputs/cv_results.json` và `outputs/cv_results.csv`.

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

`cv_results.json` và `cv_results (8).json` là cùng một kết quả; hai CSV tương ứng cũng
là bản sao. Bảng xếp hạng lịch sử là `outputs/ablation_summary.csv`.

Artifact CV hiện có của lần chạy 1.306 nhãn là
`outputs/sentiment-cv-merged/cv_results.{json,csv}`; archive được giữ là
`outputs/merged__title_context__head_tail__inverse_frequency__e5.zip`.

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

## 6. Đánh giá trạng thái lịch sử trên 306 mẫu

### Đã xác nhận

- Notebook đã chạy đủ ma trận chín cấu hình.
- Mỗi cấu hình đã hoàn thành năm fold.
- Kết quả JSON và CSV có đầy đủ số liệu tổng hợp.
- Tất cả cấu hình dùng cùng hạt giống, siêu tham số và mã băm tệp nhãn.
- Chiến lược cắt token không được kích hoạt trên tệp nhãn này (mục 4.3), nên lần chạy
  không phân biệt được ba chiến lược.
- Cơ chế lưu trữ mới không làm ma trận ablation đầy đĩa.
- Lần chạy 5 epoch đã hoàn thành đủ chín cấu hình và lần chạy riêng cấu hình đại diện đã lưu các thư mục `fold-*/best/`.

### Hạn chế của lần chạy lịch sử

- Lần chạy 5 epoch lịch sử không phân loại cân bằng ba lớp: `NEGATIVE` có recall trung bình `0.0619` và bằng 0 ở ba trên năm fold.
- Không thể kết luận `context` tốt hơn `title`, vì số lượng mẫu sau chuẩn bị đầu vào khác nhau.
- Các giới hạn này được thay thế cho mục đích báo cáo bởi xác thực chéo theo tầng trên 1.306 nhãn ở đầu tài liệu.

## 7. Quyết định lịch sử đã được thay thế

Quyết định dưới đây chỉ giải thích cách nhánh dự báo lịch sử được tạo. Nó không phải quy
tắc lựa chọn cho bộ sinh mới. Tại thời điểm lần chạy lịch sử, tập dữ liệu có 306 mẫu,
gồm `31 NEGATIVE`, `204 NEUTRAL` và `71 POSITIVE`; lớp NEGATIVE quá nhỏ để tăng epoch
giải quyết được bất định.

### 7.1. Checkpoint lịch sử và giới hạn quy tắc chọn

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

Fold 1 được dùng trong nhánh dự báo lịch sử; quy tắc này vẫn tham chiếu tập kiểm thử
ngoài, nên không được tái sử dụng cho bộ sinh mới. Bộ sinh mới phải tinh chỉnh lại với
quy tắc validation-only hoặc tổ hợp toàn bộ năm checkpoint.

### 7.2. Khớp độ dài đầu vào giữa huấn luyện và suy luận

Checkpoint được huấn luyện trên `title` cộng `body_preview` bị chặn 400 ký tự, trong khi
`articles.parquet` giữ nội dung đầy đủ (trung vị 429 ký tự, phân vị 90 là 2218 ký tự). Nếu
suy luận trên nội dung đầy đủ, đầu vào sẽ dài hơn hẳn miền huấn luyện. Do đó `score-news`
được chạy với `--context-chars 400` để giữ độ dài ngữ cảnh trong đúng miền của checkpoint.

### 7.3. Giới hạn của đầu vào dự báo lịch sử

- Recall `NEGATIVE` của bộ sinh lịch sử là `0.0619`; tin xấu bị nhận diện rất yếu.
- Đặc trưng đưa vào nhánh dự báo là **phân phối xác suất mềm**, không phải nhãn cứng, nên
  nhiễu ở lớp thiểu số làm suy giảm tín hiệu thay vì tạo nhãn sai dứt khoát.
- Các kết luận của artifact 306 mẫu chỉ có giá trị lịch sử; chúng đã được thay bằng lượt
  chấm lại và đối chứng hồi cứu ở đầu tài liệu.

### 7.4. Hạng mục đánh giá ngoài mẫu đã hoàn thành

Hạng mục "đánh giá không rò rỉ thời điểm" đã được thực hiện theo phương án đóng
băng điểm kiểm: `point-in-time` được tinh chỉnh chỉ trên 1.049 nhãn trước phiên
kiểm thử đầu tiên (`label_cutoff` 2024-10-21), kho tin được chấm lại vào
`data/processed/news_sentiment_pit.parquet`, và lượt `forecast_pit_sentiment` được
chạy cùng đối chứng cùng kích thước `forecast_sizematched_s7_sentiment`. Kết quả
nằm ở bảng ba điểm kiểm ở đầu tài liệu: khoảng tin cậy theo ngày của cả hai lượt
đều chứa 0, nên chưa có bằng chứng đủ mạnh về đóng góp dương của phân cực cảm xúc.

## 8. Tệp cần lưu trữ

Để tái kiểm tra kết quả hiện tại, giữ:

- `outputs/sentiment-cv-merged/cv_results.json` và `cv_results.csv`;
- archive Kaggle chứa năm thư mục `fold-*/best/` cho tập 1.306 nhãn
  (`outputs/merged__title_context__head_tail__inverse_frequency__e5.zip`);
- ba điểm kiểm triển khai `models/sentiment/{merged-refit,point-in-time,size-matched-s7}/`
  (archive `outputs/merged-refit.zip`, `outputs/point-in-time.zip`,
  `outputs/size-matched-s7.zip`) kèm `manifest.json` của mỗi điểm kiểm;
- các parquet cảm xúc đã chấm `data/processed/news_sentiment_{merged,pit,sizematched_s7}.parquet`
  và manifest đi kèm;
- các thư mục lượt `outputs/forecast_*_{sentiment,control}*` cùng
  `forecast_results.json`, `information_gain*.json`, `backtest_*.json`;
- `outputs/signal_ic_{pit,merged}.json`;
- bản notebook đã chạy;
- mã nguồn ở nhánh có cơ chế lưu checkpoint an toàn;
- tệp nhãn gốc hoặc mã băm của tệp nhãn.

Các tệp ablation 306 mẫu là artifact lịch sử; chỉ giữ chúng khi cần tái tạo các bảng
so sánh lịch sử.

## 9. Tài liệu và mã thực thi liên quan

- Quy trình chạy notebook: [`notebooks/training-guide.md`](../notebooks/training-guide.md)
- Ghi chú notebook: [`notebooks/README.md`](../notebooks/README.md)
- Lệnh và giao diện thực thi: [`src/stf/cli.py`](../src/stf/cli.py)
- Chuẩn bị dữ liệu và khử trùng lặp: [`src/stf/sentiment/dataset.py`](../src/stf/sentiment/dataset.py)
- Xác thực chéo và ablation: [`src/stf/sentiment/experiments.py`](../src/stf/sentiment/experiments.py)
- Huấn luyện và lưu mô hình: [`src/stf/sentiment/model.py`](../src/stf/sentiment/model.py)
