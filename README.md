# Speech Emotion Recognition (SER) — PyTorch Implementation

Hệ thống nhận dạng cảm xúc giọng nói (SER) sử dụng Log Spectrogram làm đặc trưng đầu vào và các mạng CNN (Transfer Learning) để phân loại cảm xúc. Được thực hiện và mở rộng trên bộ dữ liệu **IEMOCAP** (5 sessions, 10 speakers).

---

## Kết quả thực nghiệm (Single Fold: val=1F, test=1M)

| # | Mô hình | Loss Function | Data Augmentation | Params | Test Loss | WA (%) | UA (%) |
|---|---------|--------------|-------------------|--------|-----------|--------|--------|
| 1 | AlexNet (Baseline) | Cross Entropy | Không | 57.02M | 1.2138 | 50.73 | 25.00 |
| 2 | AlexNet GAP | Cross Entropy | Không | 2.47M | 0.9859 | 78.54 | 64.89 |
| 3 | ResNet-18 | Focal Loss | EMix-S | 11.18M | 0.4541 | 76.10 | 64.37 |
| 4 | DenseNet-121 | Focal Loss | EMix-S | 6.96M | 0.2927 | 68.78 | 66.48 |
| 5 | **EfficientNet-B0** ★ | **Focal Loss** | **EMix-S** | **4.01M** | **0.2786** | **80.49** | **73.25** |
| 6 | MobileNet-V2 | Focal Loss | EMix-S | 2.23M | 0.3549 | 76.59 | 71.15 |

> ★ **EfficientNet-B0 + Focal Loss + EMix-S** là mô hình tốt nhất, đạt WA=80.49% và UA=73.25%.

---

## Cấu trúc thư mục

```
speech_emo_recognition/
├── IEMOCAP_logspec200.pkl      # Dataset đặc trưng Log Spectrogram
├── train_ser.py                # Script huấn luyện chính
├── model.py                    # Định nghĩa tất cả kiến trúc mạng
├── data_utils.py               # Xử lý và tải dữ liệu
├── run_experiments.py          # Chạy tự động tất cả thực nghiệm
├── crossval_efficientnet.py    # 5-Fold Cross-Validation cho EfficientNet-B0
├── ablation_study.py           # Ablation study: CE vs Focal, EMix variants
├── plot_comparison.py          # Vẽ biểu đồ so sánh tổng hợp
├── test_demo.py                # Demo nhận dạng cảm xúc từ file .wav
├── features_extraction/        # Scripts trích xuất đặc trưng
└── *.pth                       # Trọng số các mô hình đã huấn luyện
```

---

## Môi trường yêu cầu

```bash
# Môi trường Python (Windows, Miniforge)
D:\miniforge3\envs\pt310\python.exe

# Cài đặt dependencies
pip install -r requirements.txt
```

**requirements.txt** bao gồm: `torch`, `torchvision`, `numpy`, `scikit-learn`, `imbalanced-learn`, `pandas`, `matplotlib`, `librosa`, `pillow`.

---

## Hướng dẫn sử dụng

### 1. Huấn luyện một mô hình đơn lẻ

```bash
python train_ser.py IEMOCAP_logspec200.pkl \
    --ser_model efficientnet \
    --val_id 1F --test_id 1M \
    --num_epochs 60 --batch_size 64 --lr 0.0001 \
    --seed 100 --gpu 1 \
    --save_label efficientnet_focal_loss \
    --loss_type focal \
    --emix emix-s \
    --pretrained
```

**Tham số `--ser_model`:**

| Giá trị | Mô hình |
|---------|---------|
| `alexnet` | AlexNet Finetuning (57M params) |
| `alexnet_gap` | AlexNet + Global Average Pooling (2.5M params) |
| `resnet` | ResNet-18 với Skip Connection (11.2M params) |
| `densenet` | DenseNet-121 với Dense Block (7.0M params) |
| `efficientnet` | EfficientNet-B0 — **Model tốt nhất** (4.0M params) |
| `mobilenet` | MobileNet-V2 — Siêu nhẹ (2.2M params) |

**Tham số `--loss_type`:**
- `ce` — Cross Entropy Loss (mặc định)
- `focal` — Focal Loss với γ=2.0, alpha được tính tự động theo phân bố dữ liệu

**Tham số `--emix`:**
- `emix-s` — EMix-Same: Trộn mẫu cùng nhãn, λ~U(0,1)
- `emix-n` — EMix-Neutral: Trộn với mẫu Neutral, λ~U(0.5,1)
- `emix-ns` — EMix-NS: Kết hợp cả hai

**Phân chia Speaker-Independent:**

| Speaker ID | Session | Giới tính |
|------------|---------|-----------|
| 1F, 1M | Session 1 | Female, Male |
| 2F, 2M | Session 2 | Female, Male |
| 3F, 3M | Session 3 | Female, Male |
| 4F, 4M | Session 4 | Female, Male |
| 5F, 5M | Session 5 | Female, Male |

### 2. Chạy tất cả thực nghiệm tự động

```bash
python run_experiments.py
```

Tự động huấn luyện 6 mô hình (bỏ qua nếu đã có `.pth`), vẽ biểu đồ, và cập nhật bảng báo cáo.

### 3. 5-Fold Cross-Validation (EfficientNet-B0)

```bash
python crossval_efficientnet.py
```

Chạy 5 fold tuần tự (mỗi fold 60 epochs). Kết quả lưu vào `crossval_efficientnet_results.json`.

### 4. Ablation Study

```bash
python ablation_study.py
```

So sánh 4 cấu hình: CE only, Focal only, Focal+EMix-S, Focal+EMix-NS.

### 5. Vẽ biểu đồ so sánh

```bash
python -X utf8 plot_comparison.py
```

Tạo các biểu đồ:
- `comparison_wa_ua.png` — WA/UA grouped bar chart
- `comparison_per_class.png` — Per-class accuracy heatmap
- `comparison_params_vs_wa.png` — Params vs WA scatter
- `summary_figure.png` — Publication-quality figure
- `crossval_results.png` — 5-Fold CV results (nếu có JSON)

### 6. Demo nhận dạng cảm xúc

```bash
python test_demo.py --model efficientnet --pth efficientnet_focal_loss.pth --wav <path_to_wav>
```

---

## Phương pháp

### Đặc trưng đầu vào
- **Log Spectrogram** 200 tần số, cắt thành segment 200ms
- Chuẩn hóa MinMax về [0, 1]
- Chuyển sang ảnh 3-kênh (gray → RGB-like) với AlexNet preprocessing (Resize 256)

### Cải tiến chính
1. **Focal Loss** (γ=2.0): Tập trung học các mẫu khó, giải quyết mất cân bằng dữ liệu
2. **EMix Augmentation**: Tăng cường dữ liệu theo kỹ thuật trộn đặc trưng cảm xúc
3. **Transfer Learning**: Sử dụng ImageNet pre-trained weights

### Phân chia dữ liệu (Speaker-Independent)
- **Train**: 8 speakers từ 4 session
- **Validation**: 1 speaker từ session còn lại
- **Test**: 1 speaker còn lại từ session đó

---

## Tài liệu tham khảo

- IEMOCAP Dataset: [Busso et al., 2008](https://sail.usc.edu/iemocap/)
- Focal Loss: [Lin et al., 2017 — RetinaNet](https://arxiv.org/abs/1708.02002)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- EMix Augmentation: Dựa trên MixUp [Zhang et al., 2018](https://arxiv.org/abs/1710.09412)
- AlexNet: [Krizhevsky et al., 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
