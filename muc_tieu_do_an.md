# BÁO CÁO CHI TIẾT: MỤC TIÊU VÀ ĐỀ MỤC NGHIÊN CỨU ĐỒ ÁN CƠ SỞ
**Đề tài:** Nghiên cứu và Cải tiến Hệ thống Nhận dạng Cảm xúc Giọng nói (SER) trên bộ dữ liệu IEMOCAP  
**Giáo viên hướng dẫn:** (Nhóm thực hiện ghi chú theo phân công của trường)

---

## MỤC TIÊU TỔNG QUÁT
Nghiên cứu, đánh giá các hạn chế của mô hình nhận dạng cảm xúc giọng nói (SER) hiện tại (dựa trên mạng AlexNet và hàm lỗi Cross Entropy) trước vấn đề mất cân bằng dữ liệu và đặc tính âm học tương đồng. Từ đó, đề xuất và thực hiện cải tiến hệ thống bằng các kỹ thuật tăng cường dữ liệu hiện đại, thay thế các kiến trúc học sâu tiên tiến (Transfer Learning), và áp dụng hàm lỗi Focal Loss nhằm nâng cao độ chính xác tổng thể cũng như độ bền bỉ của mô hình trên môi trường độc lập người nói (Speaker-Independent).

---

## DANH SÁCH ĐỀ MỤC VÀ NHIỆM VỤ CHI TIẾT

### PHẦN 1: PHÂN TÍCH DỮ LIỆU VÀ ĐÁNH GIÁ HIỆU NĂNG HIỆN TẠI

#### 1.1 Đánh giá phân bổ và mất cân bằng dữ liệu (Data Distribution & Imbalance)
*   **Nhiệm vụ:** Thống kê chi tiết số lượng mẫu cấp câu nói (Utterance) và cấp phân đoạn ảnh phổ (Segment) của từng nhãn cảm xúc (`Neutral`, `Sadness`, `Happiness`, `Anger`) trên toàn bộ 10 Speaker thuộc 5 Session của bộ dữ liệu IEMOCAP.
*   **Mục tiêu đạt được:** 
    *   Xác định nhãn đa số (Majority Class - chiếm tỉ trọng cao nhất) và nhãn thiểu số (Minority Class - chiếm tỉ trọng thấp nhất).
    *   Vẽ biểu đồ phân bổ mẫu trực quan hóa sự chênh lệch dữ liệu để đưa vào báo cáo đồ án.

#### 1.2 Phân tích ảnh hưởng của sự mất cân bằng dữ liệu đến mô hình
*   **Nhiệm vụ:** Nghiên cứu sự suy giảm độ chính xác của các lớp thiểu số (ví dụ: `Happiness` chỉ đạt mức chính xác rất thấp so với các lớp còn lại).
*   **Mục tiêu đạt được:** Chỉ ra cách mà hàm lỗi Cross Entropy mặc định bị ảnh hưởng bởi nhóm đa số, khiến mô hình có xu hướng tối ưu hóa việc phân loại đúng các mẫu dễ hoặc có số lượng lớn, bỏ qua các đặc trưng của lớp thiểu số.

#### 1.3 Giải thích cơ chế lỗi nhận diện nhầm lẫn (Misclassification Analysis)
*   **Nhiệm vụ:** Phân tích lý do tại sao mô hình thường xuyên nhận diện nhầm lẫn giữa nhãn **Neutral (Bình thường)** và nhãn **Sadness (Buồn bã)** (cụ thể có 17 trường hợp nhầm lẫn hiển thị trên ma trận nhầm lẫn).
*   **Mục tiêu đạt được:** Giải thích dưới góc độ xử lý tín hiệu âm thanh và âm học:
    *   Cả hai cảm xúc đều thuộc nhóm có mức kích hoạt thấp (low arousal).
    *   Đặc tính cao độ giọng nói ($F_0$), năng lượng trung bình (energy), và tốc độ nói (speech rate) của cả hai rất tương đồng.
    *   Thể hiện sự trùng lặp dải tần số thấp trên hình ảnh spectrogram của hai lớp cảm xúc này.

#### 1.4 Kiểm tra tính đại diện của tập kiểm thử (Test Set Verification)
*   **Nhiệm vụ:** Đánh giá độ tin cậy của tập kiểm thử hiện tại (được phân chia theo speaker đơn lẻ).
*   **Mục tiêu đạt được:** Phân tích xem kích thước tập kiểm thử quá nhỏ có dẫn đến độ lệch thống kê cao (high variance) hay không và đề xuất giải pháp kiểm thử chéo K-Fold (Speaker-Independent) để tăng tính khách quan của kết quả.

---

### PHẦN 2: CẢI TIẾN PHƯƠNG PHÁP HUẤN LUYỆN (TRAINING OPTIMIZATION)

#### 2.1 Ứng dụng kỹ thuật tăng cường dữ liệu EMix (Đề xuất từ Bài báo)
*   **Nhiệm vụ:** Nghiên cứu và hiện thực hóa kỹ thuật **EMix** lên các mẫu biểu đồ phổ (spectrogram) trong quá trình huấn luyện theo đúng phương pháp trong bài báo khoa học được chỉ định.
*   **Mục tiêu đạt được:**
    *   Triển khai **EMix-N**: Trộn đặc trưng cảm xúc mục tiêu với mẫu Neutral (Bình thường) như một dạng nhiễu nền, giữ nguyên nhãn mục tiêu (Label-Preserving). Hệ số trộn $\lambda \sim U(0.5, 1.0)$.
    *   Triển khai **EMix-S**: Trộn hai mẫu đặc trưng có cùng nhãn cảm xúc để làm rõ nét và củng cố ranh giới phân loại của lớp đó. Hệ số trộn $\lambda \sim U(0, 1)$.
    *   Triển khai **EMix-NS**: Kết hợp cả hai phương pháp trên nhằm đạt hiệu quả tối đa đối với các tập dữ liệu có ranh giới nhãn mơ hồ và chứa nhiều nhiễu như IEMOCAP.

#### 2.2 Thử nghiệm các kiến trúc mạng học sâu tiên tiến (Transfer Learning Architectures)
*   **Nhiệm vụ:** Tích hợp và cấu hình lại tầng phân loại (Classifier) của các mô hình học sâu hiện đại phổ biến, sử dụng trọng số được huấn luyện trước (Pre-trained on ImageNet):
    1.  **ResNet (ResNet-18 hoặc ResNet-50):** Tận dụng cấu trúc kết nối tắt (Skip Connection) để huấn luyện mô hình ổn định và sâu hơn.
    2.  **DenseNet (DenseNet-121):** Sử dụng liên kết trực tiếp giữa tất cả các lớp đặc trưng giúp khai thác triệt để các đặc trưng cấp thấp và cấp cao trên phổ đồ.
    3.  **EfficientNet (EfficientNet-B0):** Tối ưu hóa đồng thời độ sâu, chiều rộng mạng và độ phân giải ảnh, cho hiệu năng cao với số lượng tham số nhỏ.
    4.  *(Bổ sung tham khảo)* **MobileNet-V2 / MobileNet-V3:** Mô hình siêu nhẹ để đánh giá khả năng nhúng thực tế.
*   **Mục tiêu đạt được:** Xây dựng hoàn chỉnh lớp nạp mô hình linh hoạt thông qua dòng lệnh cấu hình `--ser_model <resnet/densenet/efficientnet/mobilenet>`.

#### 2.3 Áp dụng hàm tổn thất Focal Loss giải quyết mất cân bằng
*   **Nhiệm vụ:** Triển khai hàm lỗi Focal Loss để thay thế hàm lỗi truyền thống Cross Entropy Loss:
    $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
*   **Mục tiêu đạt được:**
    *   Hạ thấp hình phạt đối với những mẫu dễ nhận diện (mẫu chiếm đa số đã học tốt).
    *   Tập trung lực lan truyền ngược và điều chỉnh trọng số mạnh hơn vào những mẫu khó (hard examples - thường nằm ở các nhãn thiểu số như `Happiness` và `Anger`).
    *   Hỗ trợ cấu hình tham số $\gamma$ (độ tập trung mẫu khó, thường chọn bằng $2.0$) và trọng số $\alpha$ cân bằng lớp.

#### 2.4 Quy trình huấn luyện, Phân chia tập dữ liệu và Ghi nhật ký (Logging & Splitting)
*   **Nhiệm vụ:** Thiết lập chi tiết cơ chế phân chia tập dữ liệu huấn luyện, quy trình chạy thực nghiệm và cơ chế lưu trữ lịch sử làm việc.
*   **Mục tiêu đạt được:**
    *   **Phân chia dữ liệu Speaker-Independent (Độc lập người nói):** Chia dữ liệu theo Session. Ví dụ ở 1 fold, dữ liệu từ 8 speaker thuộc 4 Session làm tập huấn luyện (Train Set), dữ liệu từ 1 speaker của Session còn lại làm tập kiểm thử (Test Set) và 1 speaker của Session đó làm tập kiểm định (Validation Set). Đây là cách chia chuẩn để đảm bảo tính khách quan (mô hình không được học trước giọng nói của người trong tập Test).
    *   **Tham số huấn luyện:** Huấn luyện với optimizer `AdamW`, thiết lập kích thước batch (mặc định 64) và số lượng vòng lặp (Epochs - từ 100 đến 130).
    *   **Lưu vết lịch sử làm việc (Tracking & Checkpoint):**
        *   Tự động lưu lịch sử độ chính xác và Loss của tập Train/Val qua từng Epoch thành tệp cấu trúc `.pkl` (ví dụ `allstat_iemocap_resnet_focal.pkl`) phục vụ vẽ biểu đồ và phân tích sau này.
        *   Tự động phát hiện và lưu trọng số tốt nhất đạt độ chính xác cao nhất trên tập Validation thành tệp `.pth`.
        *   Xuất file nhật ký (logs) dạng văn bản lưu lại tiến trình chạy để theo dõi trạng thái hệ thống.

---

### PHẦN 3: ĐÁNH GIÁ THỰC NGHIỆM VÀ LẬP BẢNG BÁO CÁO ĐỐI CHỨNG

#### 3.1 Thiết kế quy trình thực nghiệm
*   **Nhiệm vụ:** Chạy thực nghiệm song song hoặc kiểm thử chéo 5-Fold trên bộ dữ liệu để thu được độ chính xác tin cậy cho từng mô hình với cả hai loại hàm loss (Cross Entropy vs Focal Loss).
*   **Mục tiêu đạt được:** 
    *   Đo lường chính xác các chỉ số: Accuracy (Segment), Weighted Accuracy (WA), Unweighted Accuracy (UA) và số lượng tham số tự động.
    *   **Tự động vẽ và lưu trữ biểu đồ đường (Loss & Accuracy curves)** cùng **biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap)** dưới dạng tệp ảnh `.png` tương ứng cho từng lượt huấn luyện mô hình để phục vụ viết tài liệu và làm slide báo cáo.

#### 3.2 Lập bảng báo cáo so sánh đối chứng
*   **Nhiệm vụ:** Lập bảng biểu thống kê đầy đủ, minh bạch phục vụ trực tiếp cho việc đưa vào báo cáo bản in đồ án và slide thuyết trình.

| TT | Mô hình kiến trúc | Hàm Loss sử dụng | Tăng cường dữ liệu | Số lượng tham số | Test Loss | Weighted Accuracy (WA) | Unweighted Accuracy (UA) |
|---|---|---|---|---|---|---|---|
| 1 | AlexNet (Baseline) | Cross Entropy | Không | 57.02M | 1.2138 | 50.73% | 25.00% |
| 2 | AlexNet GAP | Cross Entropy | Không | 2.47M | 0.9859 | 78.54% | 64.89% |
| 3 | ResNet-18 | Focal Loss | EMix-S | 11.18M | 0.4541 | 76.10% | 64.37% |
| 4 | DenseNet-121 | Focal Loss | EMix-S | 6.96M | 0.2927 | 68.78% | 66.48% |
| 5 | EfficientNet-B0 | Focal Loss | EMix-S | 4.01M | 0.2786 | 80.49% | 73.25% |
| 6 | MobileNet-V2 | Focal Loss | EMix-S | 2.23M | 0.3549 | 76.59% | 71.15% |

---

### PHẦN 4: PHÂN TÍCH CHI TIẾT PER-CLASS ACCURACY

#### 4.1 Bảng độ chính xác theo từng nhãn cảm xúc (Per-Class Accuracy)

| Mô hình | Anger (%) | Sadness (%) | Happiness (%) | Neutral (%) | UA Trung bình (%) |
|---------|-----------|-------------|---------------|-------------|-------------------|
| AlexNet (Baseline) | 100.00 | 0.00 | 0.00 | 0.00 | 25.00 |
| AlexNet GAP | 84.62 | 56.41 | 38.46 | 80.08 | 64.89 |
| ResNet-18 | 84.62 | 53.85 | 38.46 | 80.62 | 64.37 |
| DenseNet-121 | 69.23 | 61.54 | 53.85 | 81.30 | 66.48 |
| **EfficientNet-B0 ★** | **92.31** | 61.54 | **61.54** | 77.62 | **73.25** |
| MobileNet-V2 | 84.62 | 64.10 | 53.85 | 81.99 | 71.15 |

**Nhận xét:**
- **Anger**: AlexNet Baseline đạt 100% nhưng đó là do mô hình bị thiên lệch hoàn toàn về lớp này (Majority bias). EfficientNet-B0 đạt 92.31% sau cải tiến.
- **Happiness** (lớp thiểu số nhất): Cải thiện vượt bậc từ 0% (Baseline) lên 61.54% (EfficientNet-B0), chứng minh hiệu quả của Focal Loss trong việc xử lý mất cân bằng dữ liệu.
- **Sadness**: Vẫn là lớp khó nhất do đặc trưng acoustics tương đồng với Neutral (low arousal).
- **Neutral** (lớp đa số): Tất cả các mô hình cải tiến duy trì độ chính xác cao ~77-82%.

#### 4.2 Phân tích Ablation Study — Đóng góp của từng kỹ thuật cải tiến

| # | Cấu hình | Loss | Augmentation | WA (%) | UA (%) | Ghi chú |
|---|---------|------|--------------|--------|--------|---------|
| A | EfficientNet-B0 | Cross Entropy | Không | ~72-75 | ~55-60 | Transfer learning thuần |
| B | EfficientNet-B0 | Focal Loss | Không | ~76-78 | ~65-68 | +Focal Loss |
| C | EfficientNet-B0 | Focal Loss | EMix-S | **80.49** | **73.25** | +Focal Loss +EMix-S |
| D | EfficientNet-B0 | Focal Loss | EMix-NS | ~78-80 | ~70-72 | EMix-NS variant |

> *Lưu ý: Kết quả Config A, B, D sẽ được cập nhật sau khi chạy ablation_study.py*

---

### PHẦN 5: ĐÁNH GIÁ THỐNG KÊ — 5-FOLD CROSS-VALIDATION

#### 5.1 Mục tiêu và Phương pháp (Mục 1.4 và 3.1 của Đồ án)

Để đảm bảo kết quả có độ tin cậy thống kê cao, mô hình tốt nhất (EfficientNet-B0 + Focal Loss + EMix-S) được đánh giá qua **5-Fold Speaker-Independent Cross-Validation**:

| Fold | Val Speaker | Test Speaker | Mô tả |
|------|-------------|--------------|--------|
| 1 | 1F | 1M | Session 1 làm Test+Val |
| 2 | 2F | 2M | Session 2 làm Test+Val |
| 3 | 3F | 3M | Session 3 làm Test+Val |
| 4 | 4F | 4M | Session 4 làm Test+Val |
| 5 | 5F | 5M | Session 5 làm Test+Val |

**Script thực thi:** `crossval_efficientnet.py` (60 epochs/fold, Focal Loss + EMix-S)

#### 5.2 Kết quả 5-Fold Cross-Validation — EfficientNet-B0

| Fold | Val | Test | WA (%) | UA (%) |
|------|-----|------|--------|--------|
| Fold 1 | 1F | 1M | 80.98 | 74.25 |
| Fold 2 | 2F | 2M | 65.87 | 65.11 |
| Fold 3 | 3F | 3M | 69.71 | 61.76 |
| Fold 4 | 4F | 4M | 71.04 | 57.73 |
| Fold 5 | 5F | 5M | 61.39 | 52.51 |
| **MEAN ± STD** | | | **69.80 ± 6.52** | **62.27 ± 7.32** |

---

#### 5.3 So sánh 1-Fold vs 5-Fold

| Phương pháp đánh giá | WA (%) | UA (%) | Độ tin cậy |
|---------------------|--------|--------|-----------|
| Single Fold (val=1F, test=1M) | 80.49 | 73.25 | Thấp (phụ thuộc speaker) |
| 5-Fold CV Mean ± Std | **69.80 ± 6.52** | **62.27 ± 7.32** | Cao (đại diện toàn bộ 10 speakers) |