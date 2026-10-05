# Low-Light Enhancement

## 1. Lý do chọn đề tài

Nhiều hình ảnh chụp trong điều kiện ánh sáng yếu thường gặp phải hiện tượng nhiễu, mất chi tiết hoặc màu sắc không chính xác. Điều này gây khó khăn cho các hệ thống thị giác máy tính trong việc nhận diện, phân tích hình ảnh, cũng như giảm trải nghiệm của người dùng. Đề tài "Low-Light Enhancement" được triển khai nhằm xây dựng các mô hình nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng thấp, hỗ trợ các ứng dụng về camera số, bảo mật, y tế và các lĩnh vực khác có yêu cầu với hình ảnh chất lượng cao từ môi trường ánh sáng kém.

### **dataset/**  
Chỉ chứa các ảnh sáng đủ từ bộ dữ liệu BDD100K. Ảnh low-light sẽ được tạo ra động trong quá trình training.

## 2. Thông tin về Dataset

- **Nguồn:** [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- **Số lượng ảnh:** Khoảng 10,000 ảnh đầy đủ ánh sáng được chụp từ nhiều bối cảnh giao thông khác nhau.
- **Tải dataset:**  
  - Truy cập: https://bdd-data.berkeley.edu/
  - Link trực tiếp: [Download Images](https://bdd-data.berkeley.edu/view/bdd100k)
- **Đặc điểm:** Dataset sơ cấp chỉ gồm các ảnh đủ sáng (daylight và clear conditions).

### Pipeline tiền xử lý  
- Ảnh low-light không thu thập riêng mà được tạo động từ ảnh sáng đủ.
- Khi lấy từng index trong dataset, sẽ tạo ảnh low-light bằng công thức:
  
  ```
  low_img = k * (light_img ** alpha)
  ```
  
  - **k**: số thực ngẫu nhiên trong đoạn [0.3, 0.7]
  - **alpha**: số thực ngẫu nhiên trong đoạn [1.5, 2.5]
- Quá trình transform này giúp dữ liệu đa dạng hóa tình trạng ánh sáng yếu cho quá trình training.
- Toàn bộ pipeline này được xử lý tại file `utils/dataset.py`.
## 3. Input và Output

- **Input**: Một bức ảnh chụp trong điều kiện ánh sáng yếu (low-light image).
- **Output**: Một bức ảnh đã được tăng cường, sáng hơn, giữ lại nhiều chi tiết và màu sắc tự nhiên hơn.

## 4. Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: chỉ số đo lường độ sai lệch giữa các giá trị pixel của ảnh sau khi tăng cường so với ảnh gốc
- **SSIM (Structural Similarity Index)**: Chỉ số đo lường độ tương đồng về cấu trúc.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: đánh giá chất lượng cảm quan (perceptual quality) – tức là mắt người nhìn thấy ảnh có "thật" hay không.

## 4. Sơ đồ cấu trúc folder

```
Low-Light-Enhancement/
├── utils/
│   └── Chứa các hàm xử lý tái sử dụng được, ví dụ: tiền xử lý dữ liệu, tách/ghép ảnh, làm sạch dataset,...
│       Đặc biệt: pipeline tiền xử lý dataset nằm tại utils/dataset.py
├── models/
│   └── Chứa các logic và kiến trúc cho các models nâng cao ảnh sáng yếu.
│       Bao gồm các module gan, mirnetv2, retinex
├── dataset/
│   └── Chứa các ảnh sáng đủ từ dataset BDD100K được chia thành 3 folder train (7000), validation (2000) và test (1000)
├── checkpoint/
│   └── Lưu trữ các checkpoint hoặc trọng số của các mô hình sau mỗi lần train.
├── best_model_state/
│    └── Chứa các kết quả tốt nhất của từng model
├── notebook/
│   └── Chứa các file chạy bằng ipynb
├── requirements.txt
│   └── Danh sách các package cần cài đặt cho project.
├── README.md
│   └── File giới thiệu dự án.
├── Gan.py
│   └── File chạy training model GAN
├── mirnetv2_main.py
│   └── File chạy training model Mirnetv2
├── Retinex_main.py
│   └── File chạy training model Retinex
└── streamlit.py
   └── File chạy demo
```

### **utils/**  
Chứa các hàm tiện ích,
- pipeline tiền xử lý dataset trong file `utils/dataset.py`. Khi lấy từng index trong dataset, sẽ tạo ra ảnh low-light tương ứng để phục vụ cho quá trình training.
- hàm đánh giá metrics `utils/evaluation.py` chứa các logic đánh giá model

### **models/**  
Chứa logic, kiến trúc các mô hình Deep Learning cho bài toán nâng cao ảnh sáng yếu. Bao gồm các module chính:
- Gan
- mirnetv2_model
- Retinex

## 5. Hướng dẫn sử dụng

### 5.1. Cài đặt môi trường

```bash
git clone https://github.com/tuananhpa/Low-Light-Enhancement.git
cd Low-Light-Enhancement
pip install -r requirements.txt
```

### 5.2. Dataset

- Đặt ảnh daylight vào folder `dataset/`.
- Việc sinh ảnh low-light sẽ tự động xảy ra khi train model (theo pipeline đã mô tả).

### 5.3. Chạy training model
GANs:
```bash
python Gan.py
```

MIRnetv2: 
```bash
python mirnetv2_main.py
```

Retinex:
```bash
python Retinex_main.py
```

Kết quả sẽ được lưu checkpoint vào folder `checkpoints/ tên model`.

## 6. Demo 
Chạy streamlit để mở web demo:
```bash
streamlit run streamlit.py
```



