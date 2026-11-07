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

## 2. Đưa code và dataset lên Google Colab

Quy trình của dự án là:

```text
Máy cá nhân                         GitHub                         Google Colab
datasets/merged → ZIP + SHA256      code + notebook        →       clone code
       │                                  │                           │
       └──────── upload Google Drive ─────┴──────────────→           GPU train
                                                                     │
                                                checkpoint/log/metric → Google Drive
```

**Không upload ảnh dataset, ZIP hoặc weights lên GitHub.** Repository chỉ chứa code,
notebook, test, tài liệu và metadata nguồn. Dataset được đưa lên Google Drive vì kích
thước lớn và đã được `.gitignore` loại khỏi Git.

### Bước 1 — Đóng gói dataset trên máy cá nhân

Chạy trên máy đang có thư mục `/home/hieupc/Documents/btlDEEP`:

```bash
cd /home/hieupc/Documents/btlDEEP

# Nếu chưa có dependency dữ liệu:
python3 -m pip install -r requirements-data.txt

# Tạo ZIP từ dataset đã merge và tạo checksum đi kèm.
python3 colab_utils.py pack \
  --source /home/hieupc/Documents/btlDEEP/datasets/merged \
  --archive /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip

# Kiểm tra checksum trước khi upload.
sha256sum /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip
cat /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip.sha256
```

Nếu máy không cài được package toàn cục, dùng dependency có sẵn của dự án:

```bash
cd /home/hieupc/Documents/btlDEEP
PYTHONPATH=/home/hieupc/Documents/btlDEEP/.deps python3 colab_utils.py pack \
  --source /home/hieupc/Documents/btlDEEP/datasets/merged \
  --archive /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip
```

Lệnh `pack` tạo cấu trúc ZIP trực tiếp ở gốc:

```text
data.yaml
images/train/...
images/val/...
images/test/...
labels/train/...
labels/val/...
labels/test/...
merge_report.json
provenance.json
...
```

Nếu ZIP đã tồn tại, script không ghi đè. Hãy đổi tên archive, ví dụ
`ppe_dataset_v2.zip`, và upload cả file checksum tương ứng.

### Bước 2 — Upload dataset lên Google Drive

Tạo thư mục sau trong Google Drive cá nhân:

```text
My Drive/
└── PPE/
    ├── datasets/
    │   ├── ppe_dataset.zip
    │   └── ppe_dataset.zip.sha256
    ├── runs/
    └── sweeps/
```

Upload **cả hai file** sau vào `My Drive/PPE/datasets/`:

```text
/home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip
/home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset.zip.sha256
```

Không cần bật chia sẻ công khai. Colab sẽ đọc file sau khi mount đúng tài khoản Google.

### Bước 3 — Mở Colab và clone code từ GitHub

Mở https://colab.research.google.com, tạo notebook mới, chọn:

```text
Runtime → Change runtime type → T4 GPU hoặc GPU đang được cấp
```

Chạy cell đầu tiên:

```python
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/phamminhhieupc/DEEPLERNING.git"
REF = "main"  # Có thể thay bằng commit SHA để tái lập chính xác.
REPO_DIR = Path("/content/ppe-project")

if REPO_DIR.exists():
    raise FileExistsError(
        f"{REPO_DIR} đã tồn tại. Dùng Runtime → Restart session hoặc xóa thư mục trước khi clone lại."
    )

subprocess.run(["git", "clone", "--branch", REF, REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "requirements-dev.txt")
], check=True)
sys.path.insert(0, str(REPO_DIR))
```

Nếu repository là private, không chèn token vào `REPO_URL`. Hãy clone bằng credential
an toàn của bạn hoặc dùng repository public.

### Bước 4 — Mount Google Drive và kiểm tra GPU

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import subprocess
import torch
import ultralytics

if not torch.cuda.is_available():
    raise RuntimeError("Chưa có GPU. Chọn Runtime → Change runtime type → GPU rồi chạy lại.")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
subprocess.run(["nvidia-smi"], check=True)
```

### Bước 5 — Kiểm tra checksum và giải nén dataset

```python
from pathlib import Path
import json
from colab_utils import prepare_dataset

DATASET_ZIP = Path("/content/drive/MyDrive/PPE/datasets/ppe_dataset.zip")
CHECKSUM_FILE = Path("/content/drive/MyDrive/PPE/datasets/ppe_dataset.zip.sha256")
LOCAL_DATASET = Path("/content/ppe_dataset")

if not DATASET_ZIP.is_file():
    raise FileNotFoundError(f"Không tìm thấy: {DATASET_ZIP}")
if not CHECKSUM_FILE.is_file():
    raise FileNotFoundError(f"Không tìm thấy: {CHECKSUM_FILE}")

expected_sha256 = CHECKSUM_FILE.read_text(encoding="utf-8").split()[0]
prepared = prepare_dataset(DATASET_ZIP, LOCAL_DATASET, expected_sha256)
DATA_YAML = Path(prepared["data"])
print(json.dumps(prepared, indent=2, ensure_ascii=False))
```

Sau cell này, dataset phải có:

```text
/content/ppe_dataset/data.yaml
/content/ppe_dataset/images/train/
/content/ppe_dataset/images/val/
/content/ppe_dataset/labels/train/
/content/ppe_dataset/labels/val/
```

Nếu `/content/ppe_dataset` đã tồn tại, không chạy lại cell giải nén. Hãy dùng runtime
mới hoặc đổi `LOCAL_DATASET` sang thư mục mới sau khi kiểm tra dữ liệu cũ.

### Bước 6 — Chạy smoke test trước

Smoke test giúp kiểm tra model download, GPU, YAML, label và quyền ghi Drive trước khi
chạy nhiều giờ:

```bash
%cd /content/ppe-project
!python train_ppe.py \
  --data /content/ppe_dataset/data.yaml \
  --model yolo26n.pt \
  --epochs 1 --patience 1 \
  --imgsz 640 --batch 2 --workers 0 --device 0 \
  --project /content/drive/MyDrive/PPE/runs \
  --name smoke_01
```

Smoke test thành công khi run có `summary.json` hoặc có `failure.json` chứa lỗi rõ ràng.
Kiểm tra log tại:

```text
/content/drive/MyDrive/PPE/runs/smoke_01/
```

### Bước 7 — Train một model

Sau khi smoke test thành công, chạy model đơn lẻ:

```bash
%cd /content/ppe-project
!python train_ppe.py \
  --data /content/ppe_dataset/data.yaml \
  --model yolo26s.pt \
  --epochs 200 --patience 35 \
  --imgsz 960 --batch 4 --workers 2 --device 0 \
  --project /content/drive/MyDrive/PPE/runs \
  --name ppe_yolo26s_960_v1
```

Nếu bị hết VRAM, dùng `--batch 2`, `--imgsz 640` hoặc đổi model thành `yolo26n.pt`.
Tên run phải khác mỗi lần chạy; script không ghi đè run cũ.

### Bước 8 — Chạy sweep để chọn model tốt nhất

Để so sánh YOLO26n/YOLO26s và kích thước ảnh 640/960:

```bash
%cd /content/ppe-project
!python train_sweep.py \
  --data /content/ppe_dataset/data.yaml \
  --models yolo26n.pt,yolo26s.pt \
  --imgszs 640,960 \
  --epochs 200 --patience 35 \
  --batch 4 --workers 2 --device 0 \
  --project /content/drive/MyDrive/PPE/sweeps
```

Sweep chạy tuần tự từng candidate và chọn theo `metrics/mAP50-95(B)` trên validation.
Test chỉ được đánh giá sau khi chọn xong. Model champion nằm ở:

```text
/content/drive/MyDrive/PPE/sweeps/sweep_<timestamp>/champion/weights/best.pt
```

Thông tin chọn model nằm ở:

```text
/content/drive/MyDrive/PPE/sweeps/sweep_<timestamp>/champion/selection.json
```

Nếu GPU yếu, bắt đầu bằng:

```bash
!python train_sweep.py \
  --data /content/ppe_dataset/data.yaml \
  --models yolo26n.pt --imgszs 640 \
  --epochs 100 --patience 20 --batch 2 --workers 0 --device 0 \
  --project /content/drive/MyDrive/PPE/sweeps
```

### Bước 9 — Tải model về máy

Sau khi train xong, tải checkpoint champion hoặc copy sang thư mục Drive khác:

```python
from google.colab import files

CHAMPION = "/content/drive/MyDrive/PPE/sweeps/sweep_<timestamp>/champion/weights/best.pt"
files.download(CHAMPION)
```

File lớn có thể tải chậm; nên giữ bản gốc trong Google Drive. Đồng thời lưu cùng
`selection.json`, `summary.json`, `metrics_val.json`, `metrics_test.json` và
`environment.json` để biết model được train bằng dataset/cấu hình nào.

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