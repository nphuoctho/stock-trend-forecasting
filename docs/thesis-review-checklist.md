# Checklist khi đọc lại và chỉnh sửa báo cáo

Ghi lại các ràng buộc diễn giải phải giữ nguyên khi biên tập. Đây không phải lỗi
cần sửa mà là những chỗ dễ bị hội đồng bắt nếu câu chữ bị làm mạnh lên trong lúc
sửa văn phong.

## Trạng thái đã xác minh

| Hạng mục | Trạng thái |
| --- | --- |
| Test | 177/177 xanh |
| `build/BaoCaoDATN.pdf` | 142 trang, build sạch, 0 tham chiếu/trích dẫn treo |
| Artifact trong `outputs/` | 14/14 lần chạy nhất quán: ước lượng điểm khớp khoảng tin cậy |
| Tài liệu tham khảo | 25 mục, mục nào cũng được trích ít nhất một lần, không có khoá treo |

Lệnh build (máy này không có `latexmk`):

```bash
cd docs/thesis-latex
mkdir -p build
xelatex -interaction=nonstopmode -output-directory=build BaoCaoDATN.tex
biber build/BaoCaoDATN
xelatex -interaction=nonstopmode -output-directory=build BaoCaoDATN.tex
xelatex -interaction=nonstopmode -output-directory=build BaoCaoDATN.tex
```

## 1. Giữ nguyên mức độ khẳng định

Kết luận của đồ án là **"chưa có bằng chứng đủ mạnh rằng cảm xúc cải thiện dự báo"**,
kèm vế **"dữ liệu cũng không loại trừ một cải thiện nhỏ"**.

Đó không phải cách nói giảm nhẹ của "cảm xúc không có tác dụng". Hai phát biểu khác
nhau về bản chất: một cái nói *chưa đo được*, cái kia nói *đã bác bỏ*. Đồ án chỉ có
căn cứ cho vế thứ nhất.

Khi rút gọn câu cho mượt, dễ vô tình biến vế thứ nhất thành vế thứ hai. Các từ cần
tránh trừ khi có căn cứ trực tiếp: *chứng minh*, *khẳng định*, *bác bỏ*, *trần*,
*vượt trội rõ rệt*, *có ý nghĩa thống kê*.

Ba ước lượng đóng góp thông tin đều **dương** (+0,0080 hồi cứu; +0,0094 ngoài mẫu;
+0,0028 đối chứng cùng kích thước) nhưng **mọi khoảng tin cậy đều chứa 0**, và đây là
nhiều phép so sánh trên cùng bộ dữ liệu. Không được viết thành một kết quả dương.

## 2. Ba con số phải luôn đi kèm giới hạn của nó

Tách con số khỏi giới hạn là cách nhanh nhất để bị phản biện.

| Con số | Giới hạn bắt buộc đi kèm |
| --- | --- |
| Tương quan hạng $\approx 0$ ở phiên kế tiếp | **Chưa phát hiện bằng chứng** về liên hệ **đơn điệu** của **một điểm số vô hướng cụ thể** ở chân trời một phiên --- không phải bác bỏ liên hệ đó, đúng theo quy tắc ở Mục 1. Cũng không phải cận trên của khả năng dự báo: phản ví dụ $y = x^2$ với $x$ đối xứng thì dự báo được hoàn hảo nhưng tương quan hạng bằng 0. Mục này là **thăm dò**, chưa đăng ký trước. |
| Khoảng $[+0{,}0020; +0{,}0249]$ ở $h = 3$ | **Không** phải bằng chứng về hiệu ứng khác 0. Nhãn ở $h > 1$ chồng lấn nên số quan sát độc lập hữu hiệu chỉ còn khoảng $n/h$, khoảng tin cậy hẹp hơn mức bằng chứng cho phép. Hình mẫu qua $h = 1, 3, 5$ cũng không đơn điệu. |
| Sharpe trước phí $+3{,}44$ | **Không giao dịch được**: sổ lệnh vào lệnh tại chính giá đóng cửa mà tín hiệu vừa tiêu thụ. Ngoài ra sổ **không trung hòa thị trường** (23,2% số phiên một chiều), và 300 phiên không đủ tách các mức Sharpe gần nhau (arm ngẫu nhiên đạt $-2{,}00$). |

## 3. Nhãn lợi suất vượt trội: chỉ nêu quan sát

Được phép viết: *"với nhãn lợi suất vượt trội, mọi nhánh học được đều đạt F1 vĩ mô
thấp hơn"*.

Không được quy hiệu số giữa hai cột thành tỷ lệ phần trăm. Đó là hiệu số giữa **hai
bài toán gán nhãn khác nhau, huấn luyện riêng biệt**, không phải phân rã phương sai
cũng không phải phân rã nhân quả; và mức ngẫu nhiên của F1 vĩ mô không đúng bằng
$1/3$ khi tập kiểm thử lệch lớp.

Mục này và mục tương quan hạng **dùng chung một bộ dữ liệu**, nên chúng không phải hai
lần lặp lại độc lập và không củng cố lẫn nhau theo nghĩa thống kê.

## 4. Phần vận hành

Sổ theo dõi trực tuyến **chưa có dòng dự báo hợp lệ nào**, và lý do là ràng buộc nguồn
dữ liệu chứ không phải lỗi vận hành: nhà cung cấp giá chỉ công bố giá đóng cửa của một
phiên vào ngày hôm sau. Phần vận hành chứng minh đường xử lý chạy được đầu--cuối và cơ
chế đóng dấu phát hành hoạt động đúng; nó **không** cung cấp bằng chứng hiệu năng trực
tuyến nào. Mọi con số hiệu năng đến từ đánh giá cuốn chiếu ngoài mẫu ở Chương 4.

## 5. Việc còn lại

Trang thông tin hội đồng trong `BaoCaoDATN.tex` còn 5 ô trống, đã đánh dấu `% TODO`:
số quyết định, ngày ký, tên Chủ tịch, Thư ký, Ủy viên. Chỉ người dùng điền được.
