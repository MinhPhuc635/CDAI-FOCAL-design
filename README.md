# Vehicle Acoustic–Seismic Classifier

Ứng dụng web phân loại phương tiện từ tín hiệu âm thanh và địa chấn bằng mô hình FOCAL. Giao diện nhận cặp tín hiệu, chạy suy luận và hiển thị lớp phương tiện dự đoán, độ tin cậy, cùng xác suất của một số lớp tiếp theo.

## Mở ứng dụng

Dùng ứng dụng đã deploy tại:

**[https://cdai-focal-1.onrender.com/](https://cdai-focal-1.onrender.com/)**

Không cần cài đặt Python hay huấn luyện mô hình trên máy người dùng. Server deploy cần có checkpoint FOCAL và các file cấu hình nhãn/metrics đi kèm. Lần đầu mở dịch vụ miễn phí có thể cần chờ server khởi động.

## Chuẩn bị dữ liệu

Dataset tham khảo và tải về từ Kaggle:

**[FOCAL Dataset trên Kaggle](https://www.kaggle.com/datasets/nguyenmp/focal-dataset)**

Mỗi lượt ghi cần có đủ hai thành phần tín hiệu đồng bộ:

- **Âm thanh:** file `.wav` hoặc CSV theo cấu trúc MOD, thường có tên bắt đầu bằng `aud` hoặc `audio` (ví dụ `aud.csv`, `aud16000.csv`).
- **Địa chấn:** CSV thành phần dọc EHZ, thường có tên `ehz.csv`.

Ứng dụng nhận file `.csv` và `.wav`. WAV cần đọc được; với dữ liệu MOD, hãy giữ các file âm thanh và EHZ thuộc cùng lượt ghi để chúng tương ứng với nhau. Chỉ tải một file âm thanh hoặc một file địa chấn thì chưa đủ để dự đoán.

> Lưu ý: Kaggle cung cấp dataset để tải về và nghiên cứu. Ứng dụng trên Render nhận các file tín hiệu bạn tải lên; nó không tự truy cập ổ đĩa `F:` trên máy cá nhân của bạn.

## Cách sử dụng

1. Mở [ứng dụng Vehicle Acoustic–Seismic Classifier](https://cdai-focal-1.onrender.com/).
2. Ở khối **Dữ liệu đầu vào**, tải lên file âm thanh `.wav` (hoặc `aud*.csv`) và file địa chấn EHZ `.csv` thuộc cùng lượt ghi. Có thể chọn nhiều file cùng lúc.
3. Kiểm tra danh sách file. Nếu app không tự chọn được tín hiệu, chọn thủ công file âm thanh và CSV EHZ thích hợp.
4. Nhấn **Chạy dự đoán**.
5. Đọc kết quả ở khối **Kết quả**: phương tiện nhận diện, độ tin cậy dự đoán và xác suất lớp khác. Nếu tín hiệu yếu, không hợp lệ hoặc dự đoán không đủ rõ, app có thể báo **Không xác định / có thể ngoài tập dữ liệu** thay vì gán một nhãn chắc chắn.

Ứng dụng chia tín hiệu thành các cửa sổ 2 giây để suy luận bằng FOCAL. Giới hạn hiện tại trên server Render là tối đa **16 MB mỗi file** và **24 MB tổng dung lượng mỗi lượt tải lên**.

## Hiểu kết quả

- **Độ tin cậy dự đoán** là điểm xác suất của lớp đứng đầu, tổng hợp từ các cửa sổ tín hiệu trong file đã tải lên. Đây không phải là cam kết rằng dự đoán đúng với tỷ lệ tương ứng.
- **Test accuracy** (nếu hiển thị) là chỉ số đánh giá mô hình trên tập kiểm tra của dataset, không phải độ chính xác đã được xác nhận cho file bạn vừa tải lên.
- Nhãn `Không xác định` có nghĩa tín hiệu nằm ngoài điều kiện mà mô hình nhận diện đáng tin cậy, hoặc các lớp dự đoán quá gần nhau. Hãy kiểm tra lại cặp file, chất lượng tín hiệu và tính đồng bộ.

## Chạy ứng dụng trên máy cá nhân

Yêu cầu Python theo `requirements.txt`. Trong PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Mở địa chỉ local được Streamlit in ra, thường là `http://localhost:8501`.

Nếu muốn huấn luyện lại từ dataset đã tải về, cấu hình biến môi trường tới thư mục `raw_data` có cấu trúc lớp/lượt ghi của dataset rồi chạy `train_focal.py`. Ví dụ:

```powershell
$env:MOD_VEHICLE_DATA_DIR='F:\MOD_vehicle\raw_data'
python train_focal.py
```

Huấn luyện cần dataset và tài nguyên máy phù hợp. Người chỉ sử dụng ứng dụng đã deploy không cần thực hiện bước này.

## Công nghệ và nguồn

- Giao diện: Streamlit
- Mô hình và pipeline: FOCAL, dựa trên mã nguồn [tomoyoshki/focal](https://github.com/tomoyoshki/focal)
- Dataset: [nguyenmp/focal-dataset trên Kaggle](https://www.kaggle.com/datasets/nguyenmp/focal-dataset)
