# Phân loại melanoma bằng học sâu kết hợp XAI

Đồ án cơ sở: xây dựng và so sánh các mô hình học sâu (transfer learning) phân loại ảnh tổn thương da lành tính / ác tính (melanoma) trên bộ dữ liệu SIIM-ISIC, kết hợp các phương pháp giải thích mô hình (XAI) để kiểm tra mô hình đang "nhìn" vào đâu.


## Mục tiêu

1. Benchmark các mô hình pretrained (`resnet18`, `mobilenet_v2`, `efficientnet_b0`) bằng transfer learning, chỉ huấn luyện phần classifier head.
2. Chọn mô hình tốt nhất theo tập validation và thực hiện các thử nghiệm ablation trên mô hình đó.
3. Giải thích dự đoán bằng Grad-CAM; bổ sung LIME nếu còn thời gian.
4. Áp dụng ablation cho các mô hình pretrained còn lại.

Định hướng: không tìm research gap mới, mà tổng hợp, tái hiện và hoàn thiện các hướng làm hiện có, trình bày theo cấu trúc luận văn.

## Dữ liệu

- Bộ dữ liệu: [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) (Kaggle).
- Cần tham gia cuộc thi và chấp nhận điều khoản sử dụng trước khi tải dữ liệu. Vui lòng đọc kỹ điều khoản trước khi dùng và công bố kết quả.
- Đặc điểm chính: dữ liệu rất mất cân bằng (melanoma chiếm khoảng 1.76%), một bệnh nhân có nhiều ảnh
- 
## Cài đặt

```bash
git clone <repo-url>
cd <thư-mục-project>
uv sync
```

Thư viện dự kiến sử dụng (thêm bằng `uv add`):

```bash
uv add torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow opencv-python tqdm
uv add grad-cam lime scikit-image
```

## Cấu trúc project (dự kiến)

```
.
├── pyproject.toml
├── README.md
├── configs/            
├── data/
│   ├── raw/            
│   └── splits/        
├── notebooks/          
├── results/           
├── reports/           
└── src/
    └── <isic_melanoma>/
        ├── data/       # dataset, sampling, split theo bệnh nhân
        ├── models/     # backbone + classifier head
        ├── train.py
        ├── evaluate.py
        └── xai/        # Grad-CAM, LIME, đánh giá XAI
```

## Quy trình thực nghiệm

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0 | EDA, sampling, chia tập theo bệnh nhân, dataloader | ☐ |
| 1 | Benchmark 3 backbone (chỉ train head), nhiều seed | ☐ |
| 2 | Ablation trên backbone tốt nhất | ☐ |
| 3 | XAI: Grad-CAM (+ LIME) và đánh giá | ☐ |
| 4 | Mở rộng ablation cho backbone khác | ☐ |

### Nguyên tắc để kết quả đáng tin cậy

- Cùng split, cùng seed, cùng tiền xử lý và cùng cấu hình huấn luyện cho mọi backbone.
- Chọn mô hình và cấu hình bằng **validation**; chỉ chạy **test một lần** cho cấu hình cuối.
- Báo cáo trung bình ± độ lệch chuẩn qua nhiều seed (hoặc k-fold theo bệnh nhân).
- Với head-only, các lớp BatchNorm của backbone giữ ở chế độ `eval`.

### Metric

ROC-AUC (chọn mô hình), PR-AUC, sensitivity tại mức specificity cố định, confusion matrix tại ngưỡng chọn trên validation; ghi thêm số tham số và thời gian suy luận.

## Cách chạy

Sẽ bổ sung khi các script được hoàn thiện (huấn luyện, đánh giá, XAI).

## Tái lập kết quả

- Cố định seed cho `random`, `numpy` và `torch`.
- Lưu file split trong `data/splits/`.
- Mỗi lần chạy ghi kết quả vào CSV (backbone, cấu hình, seed, metric) trong `results/`.

## Nguồn tham khảo và ghi nhận

- Cuộc thi và dữ liệu: SIIM-ISIC Melanoma Classification (Kaggle).
- Notebook nền tảng dùng để tham khảo và phát triển: *SIIM-ISIC Melanoma Classification: CNN + LIME* (repo `Pisit-Janthawee/dl-binaryclassification-siim-isic-melanoma-cnn-lime`). Phần EDA được giữ ý tưởng và implement lại; các phần còn lại được thiết kế lại cho đồ án này.
- Danh sách tài liệu tham khảo đầy đủ: xem trong báo cáo (`reports/`).

## Giấy phép

Chưa xác định.