# Huấn luyện YOLO26 phát hiện đồ bảo hộ tại cổng kiểm soát

**Quy trình chính: GitHub (mã) → clone trên Colab → dataset ZIP từ Drive → train GPU → lưu kết quả vào Drive.**
Hướng dẫn đầy đủ nằm trong [docs/COLAB.md](docs/COLAB.md).
Notebook chạy trực tiếp: [train_ppe_colab.ipynb](train_ppe_colab.ipynb).

## Dataset đã tải và chuẩn hóa cho dự án

Xem `/home/hieupc/Documents/btlDEEP/datasets/README.md` để biết nguồn, ánh xạ lớp,
giấy phép, phần dữ liệu bị cách ly và giới hạn nhãn. Dataset cuối nằm trong
`/home/hieupc/Documents/btlDEEP/datasets/merged/`, cấu hình là `data.yaml` bên trong.
Không dùng `data.example.yaml` để thay thế names của bộ gộp (bộ gộp có 10 lớp).
Các repo có mức bao phủ nhãn khác nhau: cần rà soát/bổ sung nhãn trước khi train
model đa lớp phục vụ cảnh báo thực tế. Drive chỉ tải được một phần do lỗi truy xuất.

Mặc định fine-tune **YOLO26s** pretrained (`yolo26s.pt`) trên Google Colab GPU.
Tài liệu Ultralytics hiện khuyến nghị YOLO26 cho dự án mới; YOLO27 chưa phát hành weights.
Nguồn: https://docs.ultralytics.com/models/ và https://docs.ultralytics.com/modes/train/

## 1. Dữ liệu cần chuẩn bị

```text
ppe_dataset/
  data.yaml
  images/train/...
  images/val/...
  images/test/...     # tùy chọn
  labels/train/...
  labels/val/...
  labels/test/...     # tùy chọn
```

Mỗi ảnh `images/train/abc.jpg` ứng với `labels/train/abc.txt`. Một dòng nhãn:
`class_id x_center y_center width height`, tọa độ chuẩn hóa [0, 1].
Chỉ hỗ trợ bounding box detection, không phải nhãn segmentation.
Copy nội dung `data.example.yaml` thành `data.yaml`, sửa `path` và **đúng thứ tự names
trong bộ dữ liệu của bạn**. Không đổi tên lớp đơn thuần để biến nhãn hiện có thành nhãn mới.
Script nhận mỗi split là một thư mục ảnh, không nhận danh sách hay file manifest txt.

Các lớp mẫu:

| ID | Lớp | Quy ước gợi ý |
|---|---|---|
| 0 | person | Toàn người |
| 1 | helmet | Mũ bảo hộ |
| 2 | vest | Áo bảo hộ/phản quang |
| 3 | no_helmet | Vùng đầu không đội mũ |
| 4 | no_vest | Vùng thân không mặc áo bảo hộ |

Đây là **giả định minh họa**, không bắt buộc phải dùng 5 lớp. Nếu cần găng tay,
giày, kính… phải có dữ liệu và lớp tương ứng. Quy ước box cho các lớp vi phạm phải
nhất quán trong toàn bộ train/val/test. Không gán `no_helmet` khi đầu bị che khuất.
Nếu dữ liệu chỉ có person/helmet/vest thì không có metrics trực tiếp cho lớp thiếu PPE;
phải bổ sung logic ghép PPE với từng người và kiểm chứng bằng dữ liệu vi phạm.

Chia khoảng 70/20/10 theo **video, camera, ca làm/ngày quay**, không chia ngẫu nhiên
các frame liền kề. Script chặn ảnh trùng byte giữa các split nhưng không phát hiện
frame gần giống. Đảm bảo có vi phạm, đủ đồ, che khuất, đông người, sáng/tối, nhiều góc camera.
Ảnh không có nhãn được xem là background và đếm trong báo cáo: kiểm tra để tránh bỏ sót nhãn.
Các lớp không xuất hiện ở val không có đánh giá đáng tin cậy; kiểm tra `objects_per_class`.

## 2. Chạy Colab

1. Mở `train_ppe_colab.ipynb` bằng Google Colab, chọn **Runtime → Change runtime type → GPU**.
2. Điền `REPO_URL` và `REF` trong notebook. Notebook clone toàn bộ mã từ GitHub vào `/content/ppe-project`.
3. Upload ZIP dataset và checksum lên Google Drive theo `docs/COLAB.md`. Notebook kiểm tra checksum,
   giải nén sang `/content/ppe_dataset` và tự sửa đường dẫn `path` trong YAML, giữ nguyên ID lớp.
4. Chạy lần lượt các cell. Kết quả được lưu trực tiếp vào
   `/content/drive/MyDrive/PPE/runs/<tên_run>/` để tránh mất toàn bộ khi Colab ngắt kết nối.

Script độc lập sau khi cài requirements:

```bash
python /content/ppe-project/train_ppe.py \
  --data /content/ppe_dataset/data.yaml \
  --model yolo26s.pt --epochs 200 --patience 35 \
  --imgsz 960 --batch 4 --device 0 \
  --project /content/drive/MyDrive/PPE/runs --name ppe_experiment_01
```

dùng `yolo26n.pt`. Thử imgsz 960 nếu mũ quá nhỏ và GPU đủ bộ nhớ.
Tên run phải mới để không ghi đè kết quả. Nếu hết VRAM, giảm batch xuống 2 hoặc dùng
`yolo26n.pt`. `imgsz=960` thường tốt hơn cho mũ nhỏ nhưng chậm và tốn VRAM hơn.
Local CPU có thể dùng `--device cpu --workers 0`, nhưng train sẽ chậm.
`requirements.txt` cho phép nhận bản cập nhật YOLO26; mỗi run lưu `requirements.lock.txt`
để biết chính xác phiên bản đã sử dụng. Không tự thay PyTorch/CUDA có sẵn trên Colab.

### Chọn model tốt nhất bằng validation

Không chọn model bằng test. Để so sánh YOLO26n/YOLO26s và nhiều kích thước ảnh, chạy
sweep trên GPU Colab:

```bash
python /content/ppe-project/train_sweep.py \
  --data /content/ppe_dataset/data.yaml \
  --models yolo26n.pt,yolo26s.pt --imgszs 640,960 \
  --epochs 200 --patience 35 --batch 4 --device 0 \
  --project /content/drive/MyDrive/PPE/sweeps
```

Sweep chạy tuần tự từng candidate, chọn `champion/weights/best.pt` theo
`metrics/mAP50-95(B)` trên **validation**. Chỉ sau khi chọn xong, champion mới được
đánh giá trên test và kết quả nằm trong `champion/selection.json`. Nếu dữ liệu đủ
VRAM, có thể thêm `--imgszs 1280`; không tăng kích thước chỉ vì metric train cao hơn.

Mặc định train ghi cảnh báo nếu lớp không xuất hiện trong validation. Dùng
`--require-all-classes-in-val` để dừng hẳn thay vì tiếp tục; đây là lựa chọn nên dùng
khi báo cáo kết quả đa lớp. Không dùng `data.example.yaml` cho dataset 10 lớp.

## 3. Thuật toán và early stopping

1. Kiểm tra cấu trúc, ID lớp, tọa độ box, ảnh trùng giữa các split; in thống kê.
2. Load pretrained weights; train AdamW, LR ban đầu 0.001, cosine decay,
   weight decay 0.0005; seed 42, deterministic bật, AMP trên GPU.
3. Sau mỗi epoch, Ultralytics chọn `best.pt` theo validation; patience mặc định là 35.
   Pipeline lưu thêm metadata và đánh giá lại checkpoint độc lập.
4. `train_sweep.py` chọn candidate theo **mAP50–95(B) cao nhất trên val**, sau đó mới
   đánh giá champion trên test. Test không tham gia chọn model.
5. Kết quả được lưu trong `summary.json`, `metrics_val.json`, `metrics_test.json`,
   `dataset_report.json`, `environment.json` và `requested_config.json`.

Native checkpoint có thể ghi đè `best.pt` khi mAP bằng mức tốt nhất. Do đó summary
ghi cả epoch tốt nhất đầu tiên của monitor và epoch checkpoint tốt nhất cuối cùng
khi hòa. Kết quả validation đánh giá lại có thể chênh rất nhỏ do số học.
Seed/deterministic giúp lặp lại nhưng không bảo đảm giống tuyệt đối trên GPU/phiên bản khác.

## 4. Các tệp kết quả

| Tệp | Nội dung |
|---|---|
| `weights/best_val.pt` | Checkpoint epoch tốt nhất theo val mAP50–95; run đơn lẻ nên dùng file này |
| `weights/last.pt` | Checkpoint cuối |
| `weights/best.pt` | Checkpoint native của Ultralytics; chỉ là fallback nếu không có epoch checkpoint |
| `weights/epoch*.pt` | Checkpoint mỗi epoch để chọn đúng metric |
| `results.csv` | Loss train/val, precision, recall, mAP50, mAP50–95, LR từng epoch |
| `args.yaml`, `requested_config.json` | Tham số thực tế và tham số yêu cầu |
| `environment.json`, `requirements.lock.txt` | Python, GPU, CUDA, torch, Ultralytics và thư viện |
| `dataset_report.json`, `data.original.yaml`, `dataset.resolved.yaml` | Thống kê và cấu hình dữ liệu |
| `early_stopping.json` | Best score/epoch, số epoch không tăng, trạng thái dừng |
| `metrics_val.json`, `metrics_test.json` | Chỉ số tổng, từng lớp có GT, thời gian inference |
| `summary.json` | Tóm tắt, epoch tốt nhất, đường dẫn checkpoint |
| `results.png`, `best_val/`, `best_test/` | Đồ thị, confusion matrix, PR/F1 curves theo phiên bản |
| `failure.json` | Lỗi hoặc ngắt thủ công trong giai đoạn train/đánh giá |

Thông số được in ra đầu run, mỗi epoch và cuối run. Trong Colab, stdout/stderr còn
được lưu vào `console_<timestamp>.log`. Checkpoint ghi vào Drive theo mỗi epoch,
nhưng không thể bảo đảm hoàn tất file đang ghi nếu runtime bị ngắt đột ngột.
Checkpoint định kỳ giữ optimizer để hỗ trợ resume bằng Ultralytics; `best.pt/last.pt`
sau khi train hoàn tất có thể đã bị strip optimizer. Script này **không tự resume**
monitor tùy biến; chạy lại là một thí nghiệm mới, không coi là tiếp nối tương đương.

## 5. Từ model sang cảnh báo cổng

Pipeline triển khai đề xuất (chưa phải chức năng được cài trong bộ train):

`Camera → YOLO → tracking person → vùng cổng/đường cắt → ghép PPE/vi phạm với người
→ xác nhận nhiều frame → cảnh báo + ảnh + thời gian + track_id → cooldown`.

Không kết luận thiếu đồ chỉ vì một frame không detect được mũ/áo; có thể bị che,
motion blur hoặc confidence thấp. Chỉ xác nhận khi vùng đầu/thân quan sát được,
người nằm trong vùng kiểm soát và vi phạm lặp lại; hạn chế cảnh báo trùng bằng track ID.
Tune confidence/cooldown trên val/video riêng, sau đó báo cáo recall vi phạm,
tỷ lệ báo động sai và độ trễ trên test thực tế. mAP không phải tỷ lệ cảnh báo đúng
và không thay thế kiểm định an toàn lao động.

## 6. Kiểm thử

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile train_ppe.py
```

Không kèm dataset hay weights đã train. Cần internet để cài thư viện và tải pretrained.
Kiểm tra giấy phép Ultralytics (AGPL-3.0/Enterprise) trước khi triển khai thương mại.