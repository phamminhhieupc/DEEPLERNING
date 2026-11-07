# Hướng dẫn đưa dự án lên GitHub và huấn luyện trên Google Colab

## 1. Kiến trúc triển khai

```text
Máy cá nhân                 GitHub                   Google Colab
mã + notebook  ── push ──→  repo mã  ── git clone ──→ /content/ppe-project
dataset merged ── ZIP ──→   Google Drive ─ giải nén → /content/ppe_dataset
                                                      │ GPU train YOLO26
Google Drive/PPE/runs  ←── best.pt, CSV, logs, metrics ──┘
```

Không cần build Docker/server. Cài dependency và chạy Python trong runtime GPU Colab.
Repo chỉ chứa mã, notebook, tài liệu và metadata. Không commit ảnh dataset, ZIP,
weights, môi trường Python hay kết quả train. `.gitignore` đã loại chúng.
`datasets/metadata` có thể giữ trên GitHub để truy vết nguồn.

Dataset hiện tại là bản gộp 10 lớp; không thay bằng `data.example.yaml` có 5 lớp.
Đọc `/home/hieupc/Documents/btlDEEP/datasets/README.md` trước khi sử dụng:
nhãn giữa các nguồn chưa đầy đủ đồng nhất, Drive chỉ tải được một phần,
và dngsein chưa gộp vì thiếu định nghĩa ID. Đưa lên Colab không khắc phục các hạn chế nhãn này.

## 2. Đóng gói dataset trên máy cá nhân (một lần)

Tại máy đang có bộ dữ liệu, chạy:

```bash
python3 -m pip install -r /home/hieupc/Documents/btlDEEP/requirements-data.txt
python3 /home/hieupc/Documents/btlDEEP/colab_utils.py pack \
  --source /home/hieupc/Documents/btlDEEP/datasets/merged \
  --archive /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset_v2.zip
```

Nếu hệ thống chặn pip toàn cục, dùng virtualenv. Tại môi trường hiện tại đã có dependency
trong `.deps`, có thể thay dòng cài pip bằng:

```bash
PYTHONPATH=/home/hieupc/Documents/btlDEEP/.deps python3 \
  /home/hieupc/Documents/btlDEEP/colab_utils.py pack \
  --source /home/hieupc/Documents/btlDEEP/datasets/merged \
  --archive /home/hieupc/Documents/btlDEEP/artifacts/ppe_dataset_v2.zip
```

Lệnh tạo ZIP và file `ppe_dataset_v2.zip.sha256`. ZIP có cấu trúc trực tiếp ở gốc:

Trong lần chuẩn bị hiện tại, hai tệp đã được tạo sẵn tại
`/home/hieupc/Documents/btlDEEP/artifacts/` (ZIP khoảng 130 MiB).
Bạn có thể upload ngay, không cần chạy lại lệnh pack nếu dataset chưa thay đổi.

```text
data.yaml
images/train/...
images/val/...
images/test/...
labels/train/...
labels/val/...
labels/test/...
merge_report.json, provenance.json, ...
```

`path` trong ZIP chuyển thành `.` để di chuyển được giữa máy và Colab. Class names/IDs
và nhãn không thay đổi. Script không ghi đè ZIP đã tồn tại; đổi tên nếu đóng gói bản mới.
Upload **cả hai tệp** lên Drive:

```text
MyDrive/PPE/datasets/ppe_dataset_v2.zip
MyDrive/PPE/datasets/ppe_dataset_v2.zip.sha256
```

Không cần chia sẻ Drive công khai: Colab mount Drive cá nhân qua tài khoản của bạn.

## 3. Đẩy mã lên GitHub

Code đã được đẩy lên repository:

```text
https://github.com/phamminhhieupc/DEEPLERNING.git
```

Khi cần cập nhật code từ máy cá nhân, dùng remote/branch hiện tại, không chạy lại
`git init` hoặc thêm remote trùng:

```bash
git -C /home/hieupc/Documents/btlDEEP init
git -C /home/hieupc/Documents/btlDEEP branch -M main
git -C /home/hieupc/Documents/btlDEEP add .
git -C /home/hieupc/Documents/btlDEEP status --short
# Kiểm tra KHÔNG có raw/, merged/, artifacts/, .deps/, .venv/, ZIP hay weights.
git -C /home/hieupc/Documents/btlDEEP diff --cached --stat
git -C /home/hieupc/Documents/btlDEEP commit -m "Add YOLO26 PPE Colab training pipeline"
```

Remote của dự án:

```bash
git -C /home/hieupc/Documents/btlDEEP remote -v
git -C /home/hieupc/Documents/btlDEEP push -u origin main
```

Xác thực bằng GitHub CLI/credential manager/PAT trên máy cá nhân. Không đưa token vào notebook,
URL repo, tài liệu hoặc log. `.gitignore` không loại được file đã được Git theo dõi từ trước;
nếu đã commit dataset, phải gỡ khỏi index/lịch sử trước khi push dữ liệu lớn.

Các tệp cần có trên repo:

- `/home/hieupc/Documents/btlDEEP/train_ppe_colab.ipynb`
- `/home/hieupc/Documents/btlDEEP/train_ppe.py`
- `/home/hieupc/Documents/btlDEEP/colab_utils.py`
- `/home/hieupc/Documents/btlDEEP/requirements.txt`
- `/home/hieupc/Documents/btlDEEP/docs/COLAB.md`
- tests và các script chuẩn bị dữ liệu nếu muốn tái tạo dữ liệu.

## 4. Mở notebook trong Colab

1. Truy cập https://colab.research.google.com/ → **File → Open notebook → GitHub**.
2. Dán URL repo vừa push, chọn `train_ppe_colab.ipynb`.
3. **Runtime → Change runtime type → GPU** (ví dụ T4 nếu tài khoản được cấp).
4. Repository đã cấu hình sẵn URL `https://github.com/phamminhhieupc/DEEPLERNING.git` trong notebook.
5. `REF = "main"` dùng nhánh mới nhất, hoặc đặt commit SHA để tái lập chính xác phiên bản.
6. Chạy lần lượt các cell từ trên xuống.

Cell đầu clone vào `/content/ppe-project`, checkout REF và cài `requirements.txt`.
Cell thứ hai mount Drive và kiểm tra GPU. Cell chuẩn bị dữ liệu đọc ZIP từ Drive,
kiểm tra SHA256, giải nén vào `/content/ppe_dataset`, cập nhật YAML và kiểm tra nhãn/split.
Như vậy không phải upload riêng script lên Colab.

Notebook mặc định hỗ trợ clone **repo public** không cần token. Với repo private,
clone bằng phương thức xác thực an toàn của bạn trước, rồi chỉ chạy phần cài dependency
và các cell tiếp theo; không chèn token vào `REPO_URL` hoặc lưu nó trong notebook.

## 5. Cấu hình train

| Biến notebook | Mặc định | Ý nghĩa |
|---|---|---|
| MODEL | yolo26s.pt | Pretrained; `yolo26n.pt` cho screening nhanh |
| EPOCHS | 200 | Train cuối; screening dùng 80 |
| PATIENCE | 35 | Train cuối; screening dùng 15 |
| BATCH | 4 | Train cuối; screening T4 có thể dùng 8, giảm khi OOM |
| IMGSZ | 960 | Train cuối; screening dùng 640 để nhanh hơn |
| WORKERS | 4 | Tăng tốc cấp dữ liệu cho T4; nếu DataLoader lỗi, đặt 0 |
| CACHE | ram | Cache ảnh trong RAM; đổi `disk`/`False` nếu thiếu RAM |
Lần đầu nên đặt `EPOCHS = 1`, `BATCH = 2`, `CACHE = "False"` để kiểm tra môi trường, pretrained download,
Lần đầu nên đặt `EPOCHS = 1`, `BATCH = 2`, `CACHE = "False"` để kiểm tra môi trường, pretrained download,
nhãn và đầu ra. Sau khi thành công, đổi về giá trị train thật và chạy lại **cell train**:

### Profile screening nhanh và train cuối

Để giảm thời gian mỗi epoch trên T4, dataset phải được giải nén ở `/content/ppe_dataset`,
không đọc ảnh trực tiếp từ Google Drive. Screening nhanh ghi kết quả tạm vào `/content`:

```bash
%cd /content/ppe-project
!python train_ppe.py \
  --data /content/ppe_dataset/data.yaml \
  --model yolo26n.pt --epochs 80 --patience 15 \
  --imgsz 640 --batch 8 --workers 4 --cache ram \
  --save-period 5 --device 0 \
  --project /content/runs_fast --name ppe_yolo26n_640_screening
```

Copy kết quả screening lên Drive sau khi hoàn tất:

```bash
!cp -r /content/runs_fast/ppe_yolo26n_640_screening \
  /content/drive/MyDrive/PPE/runs/
```

Train cuối dùng model lớn hơn và ảnh lớn hơn:

```bash
%cd /content/ppe-project
!python train_ppe.py \
  --data /content/ppe_dataset/data.yaml \
  --model yolo26s.pt --epochs 200 --patience 35 \
  --imgsz 960 --batch 4 --workers 4 --cache ram \
  --save-period 1 --device 0 \
  --project /content/drive/MyDrive/PPE/runs --name ppe_yolo26s_960_final
```

Nếu bị OOM, giảm `batch` 4 → 2, đổi `yolo26s` thành `yolo26n`, rồi mới giảm `imgsz`.
Nếu RAM không đủ, dùng `--cache disk`; nếu disk cache cũng không phù hợp, dùng
`--cache False`. Theo dõi T4 bằng `!nvidia-smi` và `!nvidia-smi dmon -s pucm`.
tên run có timestamp mới, không ghi đè. Không cần giải nén lại dataset.

Script in tham số trước khi train, dùng AMP trên GPU, workers=4 và cache RAM để tận dụng T4,
đồng thời lưu checkpoint theo validation mAP50–95, dừng sớm
khi không cải thiện nghiêm ngặt (điểm bằng nhau không reset patience), đánh giá lại
best model trên val/test nếu test tồn tại. Test không chọn model.

`requirements.txt` cho phép phiên bản Ultralytics trong dải hỗ trợ; môi trường và
`pip freeze` được ghi trong từng run, cùng Git commit trong `environment.json`.
Không tự cài lại PyTorch/CUDA trong notebook; tận dụng bản có sẵn trên Colab.

## 6. Kết quả lưu ở đâu?

```text
/content/drive/MyDrive/PPE/runs/
  console_ppe_<timestamp>.log
  ppe_<timestamp>/
    weights/best_val.pt
    weights/last.pt
    weights/epoch*.pt
    results.csv
    epoch_metrics.jsonl
    epoch_metrics.csv
    results.png
    args.yaml
    requested_config.json
    environment.json
    requirements.lock.txt
    early_stopping.json
    summary.json
    metrics_val.json
    metrics_test.json        # khi có test
    best_val/, best_test/    # đồ thị đánh giá
```

Cell cuối đọc summary và hiển thị đồ thị. Với run đơn lẻ, model triển khai là
`weights/best_val.pt` (hoặc đường dẫn `best_model` trong `summary.json`). Với sweep,
dùng `champion/weights/best.pt`.
Không dùng đường dẫn `/home/hieupc/...` trên Colab; mọi đường dẫn runtime là `/content/...`.

## 7. Ngắt runtime, chạy lại và khắc phục lỗi

- **Không thấy GPU**: chọn GPU runtime và kiểm tra hạn mức Colab. Không tự chạy CPU âm thầm.
- **ZIP/SHA256 không tìm thấy**: kiểm tra tài khoản Drive, tên file và `DATASET_ZIP`.
- **Checksum mismatch**: upload lại đủ ZIP và checksum tương ứng, không bỏ qua lỗi.
- **Dataset destination already exists**: nếu dataset đã chuẩn bị thành công, bỏ qua cell đó;
  sau khi khởi động lại kernel, đặt `DATA_YAML = Path("/content/ppe_dataset/data.yaml")`.
  Nếu giải nén dang dở, chọn thư mục mới cho `LOCAL_DATASET` hoặc xóa thủ công sau khi kiểm tra.
- **Repo đã tồn tại**: không chạy lại cell clone; dùng các cell tiếp theo. Muốn lấy mã mới,
  chủ động `git pull`/checkout và restart session trước khi import lại module.
- **CUDA out of memory**: giảm BATCH, IMGSZ hoặc dùng YOLO26n; restart runtime nếu GPU vẫn giữ bộ nhớ.
- **DataLoader worker lỗi**: WORKERS=0.
- **Run already exists**: đổi RUN_NAME; notebook tự tạo timestamp mới mỗi lần chạy cell train.
- **Colab ngắt kết nối**: `/content` có thể mất, nhưng checkpoint đã ghi vào Drive được giữ lại.
  File đang ghi lúc ngắt không được bảo đảm hoàn chỉnh. Clone/giải nén lại khi có runtime mới.
- **Dừng thủ công**: wrapper dừng subprocess, lưu console log; không để tiến trình train chạy nền ngoài ý muốn.

Hiện **không có resume chính xác optimizer + trạng thái monitor tùy biến**. Chạy lại notebook
là thí nghiệm mới. Có thể dùng đường dẫn checkpoint tin cậy làm MODEL để fine-tune tiếp,
nhưng đây không tương đương resume: optimizer, scheduler và patience bắt đầu lại.

## 8. Chọn model tốt nhất và kiểm thử mã

Để chọn model tốt nhất có cơ sở, chạy sweep trên validation trước:

```bash
python /content/ppe-project/train_sweep.py \
  --data /content/ppe_dataset/data.yaml \
  --models yolo26n.pt,yolo26s.pt --imgszs 640,960 \
  --epochs 200 --patience 35 --batch 4 --workers 4 --cache ram --device 0 \
  --project /content/drive/MyDrive/PPE/sweeps
```

Model triển khai nằm tại `sweep_<timestamp>/champion/weights/best.pt`. Candidate được
xếp hạng bằng validation mAP50-95; test chỉ được chạy sau khi champion đã được chọn.
Nếu hết VRAM, dùng `--models yolo26n.pt --imgszs 640 --batch 2`.

## 9. Kiểm thử mã

Cài `requirements-dev.txt` để chạy đầy đủ cả tests cho công cụ merge và training. Trên Colab:

```bash
python -m pip install -r /content/ppe-project/requirements-dev.txt
python -m unittest discover -s /content/ppe-project/tests -v
```

Notebook được kiểm tra cú pháp cell và công cụ pack/prepare được kiểm thử tự động.
Việc train GPU thực tế cần chạy smoke test trên runtime Colab của bạn; không có cam kết
về hạn mức GPU, thời lượng phiên hay độ chính xác model khi chưa train/đánh giá thật.