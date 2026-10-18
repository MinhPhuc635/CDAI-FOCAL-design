# Vehicle Acoustic–Seismic Classifier

Dashboard ba khối cho nhận diện phương tiện từ dataset MOD_vehicle bằng pipeline FOCAL upstream:

1. Tải tín hiệu thô `.wav` hoặc `.csv`.
2. Bấm **Chạy dự đoán**. Nếu chưa có checkpoint MOD_vehicle, ứng dụng tự tiền xử lý dataset, pretrain FOCAL và fine-tune classifier một lần.
3. Hiển thị lớp phương tiện, độ tin cậy trên mẫu hiện tại và test accuracy đo trên tập giữ lại.

## Định dạng đầu vào

Một lượt ghi cần có tín hiệu âm thanh và địa chấn đồng bộ:

- Âm thanh: WAV PCM mono 16 kHz (`.wav`), hoặc CSV tên `aud.csv` / `aud16000.csv`.
- Địa chấn: `ehz.csv` (thành phần địa chấn dọc, 100 Hz).

Dataset train mặc định nằm tại `F:\MOD_vehicle\raw_data\<class>\<run>\`. Các thư mục lớp có hậu tố số như `bicycle`, `bicycle2` được gộp về cùng nhãn `bicycle`; tên thư mục lượt ghi được giữ riêng để chia tập theo recording. FOCAL dùng schema upstream `MOD.yaml`, còn tín hiệu được đọc từ MOD_vehicle.

Ứng dụng resample audio 16 kHz → 8 kHz, chia tín hiệu thành cửa sổ 2 giây (10 interval) và dùng cùng biểu diễn FFT như FOCAL. WAV là lựa chọn gọn hơn cho âm thanh.

## Chạy trên máy phát triển

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MOD_VEHICLE_DATA_DIR='F:\MOD_vehicle\raw_data'
streamlit run app.py
```

Lần bấm dự đoán đầu tiên khi chưa có checkpoint mang metadata MOD_vehicle sẽ tự chuẩn bị dữ liệu cửa sổ, chạy FOCAL pretraining và fine-tuning, đánh giá test set, rồi lưu checkpoint/metadata trong `models/`. Checkpoint cũ được train từ MOD sẽ không được dùng nhầm; những lần sau dùng checkpoint MOD_vehicle đã lưu. Có thể train lại bằng `python train_focal.py --force`.

## Độ chính xác và giới hạn dataset

`Độ tin cậy dự đoán` là xác suất trung bình của các cửa sổ trong file đang phân loại. `Test accuracy` được tính trên recording giữ riêng; các thư mục lớp có hậu tố số được gộp về lớp gốc, nhưng recording vẫn được giữ tách biệt khi chia train/validation/test. Accuracy thực tế sau khi train MOD_vehicle được ghi trong `models/model_metrics.json` và log.

## Deploy để máy khác dùng mà không train lại

Chạy training một lần trên máy có dataset `F:\MOD_vehicle\raw_data`, sau đó đóng gói các file `models/vehicle_focal.pt`, `models/labels.json`, `models/model_metrics.json` cùng ứng dụng. Máy chủ deploy cần giữ checkpoint; máy khách chỉ mở URL và tải cặp audio/seismic lên. Không cần chép dataset hay train trên máy khách. Nếu checkpoint không được deploy, server chỉ có thể tự train khi `MOD_VEHICLE_DATA_DIR` trỏ tới dataset.

```powershell
docker build -t vehicle-focal .
docker run --rm -p 8501:8501 `
  -e MOD_VEHICLE_DATA_DIR=/data/MOD_vehicle/raw_data `
  -v F:\MOD_vehicle:/data/MOD_vehicle `
  -v ${PWD}\models:/app/models `
  vehicle-focal
```

Mở `http://localhost:8501`. Container cũng nhận biến môi trường `PORT` để tương thích với nền tảng cloud. Khi deploy với checkpoint sẵn có, server không cần mount dataset MOD_vehicle. Để huấn luyện trong container, mount thư mục dataset tại `/data/MOD_vehicle` và đặt `MOD_VEHICLE_DATA_DIR=/data/MOD_vehicle/raw_data`.

## Nguồn

Mã nguồn model và preprocessing: [tomoyoshki/focal](https://github.com/tomoyoshki/focal) (MIT). Source upstream được giữ trong `focal-main/`.
