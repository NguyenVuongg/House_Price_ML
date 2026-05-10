# House Price Prediction

Dự án Machine Learning dự đoán giá nhà dựa trên các đặc trưng như diện tích, số phòng ngủ, số phòng tắm, số tầng, vị trí đường chính, tầng hầm, điều hòa, chỗ đậu xe và tình trạng nội thất.

## Tổng quan

Repo này đang được tổ chức theo hướng một project ML cơ bản:

- Lưu dữ liệu thô trong `data/raw/`
- Lưu dữ liệu đã xử lý trong `data/processed/`
- Viết notebook EDA trong `notebook/`
- Đặt code xử lý, train, evaluate và predict trong `src/`
- Lưu model đã huấn luyện trong `models/`
- Lưu báo cáo, biểu đồ và output trong `output/` hoặc `plot/`

Hiện tại project có dataset `Housing.csv` và các file Python trong `src/` đang là khung để tiếp tục triển khai pipeline.

## Cấu trúc thư mục

```text
EX_ML/
├── data/
│   ├── raw/
│   │   └── Housing.csv
│   └── processed/
├── models/
│   └── house_price_model.pkl
├── notebook/
│   └── eda.ipynb
├── output/
│   ├── plots/
│   └── reports/
├── plot/
│   └── report/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── main.py
├── requirements.txt
├── note.txt
└── README.md
```

## Dữ liệu

File dữ liệu chính:

```text
data/raw/Housing.csv
```

Dataset có 545 bản ghi và 13 cột:

| Cột | Ý nghĩa |
| --- | --- |
| `price` | Giá nhà, biến mục tiêu cần dự đoán |
| `area` | Diện tích nhà |
| `bedrooms` | Số phòng ngủ |
| `bathrooms` | Số phòng tắm |
| `stories` | Số tầng |
| `mainroad` | Nhà có nằm trên đường chính hay không |
| `guestroom` | Có phòng khách/phòng cho khách hay không |
| `basement` | Có tầng hầm hay không |
| `hotwaterheating` | Có hệ thống nước nóng hay không |
| `airconditioning` | Có điều hòa hay không |
| `parking` | Số chỗ đậu xe |
| `prefarea` | Có nằm trong khu vực ưu tiên hay không |
| `furnishingstatus` | Tình trạng nội thất: `furnished`, `semi-furnished`, `unfurnished` |

Một số khoảng giá trị trong dữ liệu:

- `price`: từ 1,750,000 đến 13,300,000
- `area`: từ 1,650 đến 16,200
- `bedrooms`: từ 1 đến 6
- `bathrooms`: từ 1 đến 4
- `stories`: từ 1 đến 4
- `parking`: từ 0 đến 3

## Ý tưởng pipeline

Các file trong `src/` nên được dùng theo vai trò sau:

- `preprocess.py`: đọc dữ liệu, xử lý missing values nếu có, mã hóa biến phân loại và tách train/test
- `train.py`: huấn luyện model dự đoán giá nhà
- `evaluate.py`: đánh giá model bằng các chỉ số như MAE, MSE, RMSE hoặc R2
- `predict.py`: tải model đã train và dự đoán giá cho dữ liệu mới
- `utils.py`: chứa các hàm dùng chung cho nhiều bước

## Cài đặt

Yêu cầu Python 3.10+.

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Các thư viện hiện có trong `requirements.txt`:

## Cách sử dụng

Do các file xử lý chính trong `src/` hiện chưa được triển khai đầy đủ, bước chạy pipeline cần được hoàn thiện trước. Một luồng chạy đề xuất:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

File `main.py` hiện chỉ là entry point mẫu và chưa điều phối pipeline ML.

## Hướng phát triển tiếp theo

- Hoàn thiện EDA trong `notebook/eda.ipynb`
- Triển khai xử lý dữ liệu trong `src/preprocess.py`
- Huấn luyện model baseline bằng Linear Regression, Random Forest hoặc Gradient Boosting
- Lưu model bằng `pickle` hoặc `joblib` vào `models/`
- Sinh báo cáo đánh giá trong `output/reports/`
- Sinh biểu đồ EDA hoặc kết quả model trong `output/plots/`
- Bổ sung ví dụ input cho bước dự đoán trong `src/predict.py`

## Trạng thái hiện tại

- Dataset đã có: `data/raw/Housing.csv`
- Cấu trúc thư mục ML đã được tạo
- `requirements.txt` đã có các thư viện cơ bản
- Các file trong `src/`, notebook và model đang là placeholder, cần triển khai thêm để chạy được pipeline hoàn chỉnh
