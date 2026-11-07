# Dữ liệu PPE từ các nguồn được yêu cầu

## Kết quả lần tải hiện tại

Bản cuối có **1.681 ảnh**: train **1.100**, val **380**, test **201**.
Đóng góp: ilaydadeger 399, usckiki82 532, JoyKarmoker 717, Drive 33 ảnh.
Một nhóm ảnh trùng pixel nhưng khác nhãn đã bị cách ly; xem `annotation_conflicts.json`.
Repo dngsein đã tải đủ 7.063 ảnh và 7.063 nhãn nhưng **chưa đưa vào bộ gộp** vì
không có định nghĩa ID lớp. Drive chỉ gộp 33 ảnh gốc có nhãn đọc được, không phải
toàn bộ 6.606 tệp được chọn để tải. Downloader lần thử lại dừng sau 20 lỗi liên tiếp,
ghi nhận 37 lỗi trong các mục đã xử lý; các mục còn lại chưa tải xong.
`excluded_records` đếm cả dòng manifest và XML thiếu ảnh, không phải số ảnh duy nhất.

`merged_initial/` là bản trung gian được giữ để kiểm toán, **không dùng để train**
vì được tạo trước khi loại thư mục ảnh kết quả `result`. Chỉ dùng `merged/data.yaml`.

## Thư mục

- `raw/`: dữ liệu gốc, giữ tên nguồn, không sửa nhãn gốc.
- `metadata/`: cây GitHub tại commit tải về, README, cấu hình và danh sách Drive.
- `merged/`: dataset YOLO đã chuyển đổi, kèm `data.yaml`, `merge_report.json`,
  `provenance.json`, `excluded.json`, `duplicates.json`, `annotation_conflicts.json`.

**Đọc merge_report.json trước khi train. Đây là bộ gộp phục vụ rà soát nhãn, không
phải bộ đã được xác nhận gán nhãn đầy đủ cho tất cả các lớp.** Nguồn chỉ gán nhãn mũ
có thể có người/áo không được gán nhãn, gây false-negative supervision khi train chung.
Cần bổ sung nhãn hoặc huấn luyện theo tập con có cùng mức bao phủ nhãn.

## Quy tắc ánh xạ

| ID | Lớp đích | Nhãn nguồn |
|---|---|---|
| 0 | person | Person / person |
| 1 | helmet | Hardhat / hardhat / helmet |
| 2 | vest | Safety Vest |
| 3 | no_helmet | NO-Hardhat |
| 4 | no_vest | NO-Safety Vest |
| 5 | gloves | Gloves |
| 6 | mask | Mask |
| 7 | no_mask | NO-Mask |
| 8 | head | head (KHÔNG mặc định là không đội mũ) |
| 9 | person_with_helmet | person hardhat (box người, KHÔNG phải box mũ) |

Chỉ giữ các lớp người/PPE có định nghĩa rõ. Các phương tiện, máy móc, thang, cọc tiêu
trong JoyKarmoker không thuộc phạm vi PPE nên không chuyển sang dataset đích.
Không tự sinh no_helmet/no_vest/no_mask từ sự vắng mặt của PPE.

## Nguồn và giới hạn

1. https://github.com/ilaydadeger/proje24_yolo_hardhat
   - Chỉ dùng `dataset_small`; khai báo `names: ['head']`.
   - Config ghi nguồn Roboflow joseph-nelson/hard-hat-workers v14, Public Domain.
2. https://github.com/dngsein/Hardhat
   - Tải ảnh và nhãn về raw; repo không chứa định nghĩa ID lớp.
   - **Cách ly, chưa gộp vào train**. Không đoán ID dựa vào tên repo hay hình mẫu.
3. https://github.com/usckiki82/hardhat-dataset
   - VOC XML `hardhat`, `person hardhat`; ảnh không có XML không coi là background.
   - Chưa tìm thấy giấy phép dataset trong repo.
4. https://github.com/JoyKarmoker/Construction-Site-Safety-PPE-Detection
   - Chỉ lấy `Train/Construction_Site_Safety`, không lấy weights/video kết quả.
   - Config/README ghi Roboflow construction-site-safety v30, CC BY 4.0.
     Giữ metadata và nguồn để phục vụ attribution.
5. https://drive.google.com/drive/folders/10MeTaybDtEhC5mvnPkX35aD7S9rwcTmZ
   - Liệt kê được thư mục công khai. Chỉ lấy ảnh gốc/XML/manifest; bỏ thư mục `result`.
   - Việc tải từng file có thể bị chặn quyền/quota mặc dù xem được danh sách thư mục.
     Các file tải thất bại không được coi là đã gộp. Xem metadata/drive_download.json.
   - Chưa xác minh được giấy phép; không mặc định là dữ liệu tự do sử dụng.

Không cấp lại giấy phép chung cho bộ gộp. Kiểm tra quyền sử dụng từng nguồn trước
khi chia sẻ lại hoặc triển khai thương mại.

## Loại trùng và chia tập

Giải mã ảnh RGB và hash cả kích thước + pixel để phát hiện ảnh trùng kể cả khác metadata.
Ảnh trùng có nhãn khác nhau được **cách ly**, không ghép box một cách tùy tiện.
Các bản augmentation có cùng tên trước `.rf.` và các frame có mẫu tên video nhận diện
được cùng group; hash group với seed 42 chia khoảng 70/20/10. Không cam kết đúng tỷ lệ
tuyệt đối, không phân tầng lớp. Kiểm tra số lượng từng lớp ở từng split trước khi train.
Chưa phát hiện toàn bộ ảnh gần giống hay mọi frame cùng video không thể nhận biết qua tên.
Test mới không còn tương đương benchmark split gốc.

## Tái tạo

Cài `requirements-data.txt`, rồi chạy các script tại thư mục gốc dự án theo thứ tự:
`inspect_sources.py`, `download_datasets.py`, `download_drive_data.py`, `merge_datasets.py`.
Drive cần `metadata/drive_listing.json`; danh sách này đã được kiểm kê trong lần chạy đầu.
Downloader không thực thi code tải về. GitHub được ghim commit và kiểm tra Git blob SHA1.
Dataset output phải là thư mục mới; có thể dùng `merge_datasets.py --output <đường-dẫn-mới>`.
Sau khi copy bộ gộp lên Colab, sửa `path` trong data.yaml theo vị trí mới, không sửa ID lớp.