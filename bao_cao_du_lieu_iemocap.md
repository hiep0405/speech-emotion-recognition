# BÁO CÁO PHÂN TÍCH CHI TIẾT: CƠ SỞ DỮ LIỆU IEMOCAP VÀ QUY TRÌNH TIỀN XỬ LÝ
**Đề tài:** Nhận dạng cảm xúc giọng nói (SER) trên bộ dữ liệu IEMOCAP  

---

## 1. Giới thiệu tổng quan về bộ dữ liệu USC-IEMOCAP
Bộ dữ liệu **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** được thu thập bởi phòng thí nghiệm SAIL (Speech Analysis and Interpretation Laboratory) thuộc Đại học Nam California (USC), công bố năm 2008. Đây là một trong những bộ dữ liệu chuẩn mực và phổ biến nhất trên thế giới dùng để nghiên cứu nhận dạng cảm xúc đa phương thức.

### Các thông số cốt lõi:
*   **Quy mô:** Khoảng 12 giờ dữ liệu bao gồm: âm thanh (audio), khẩu hình/video mặt, chuyển động cơ thể (motion capture) và văn bản hội thoại (transcriptions).
*   **Đối tượng:** 10 diễn viên chuyên nghiệp (5 Nam, 5 Nữ), chia thành 5 cặp đôi đối thoại ngẫu nhiên theo từng Session (từ Session 1 đến Session 5).
*   **Kịch bản diễn xuất:** Chia làm 2 loại:
    1.  **Improvised (Ngẫu hứng):** Diễn viên diễn xuất tự nhiên dựa trên các tình huống giả lập gợi cảm xúc mạnh (ví dụ: nghe tin người thân mất, trúng số, cãi nhau với nhân viên dịch vụ...).
    2.  **Scripted (Kịch bản):** Diễn xuất theo lời thoại viết sẵn trong các tác phẩm kịch.
*   **Đánh giá cảm xúc (Annotation):** Mỗi câu nói (utterance) được đánh giá cảm xúc bởi ít nhất 3 chuyên gia theo 10 nhãn phân loại: *Angry (Tức giận), Happy (Vui vẻ), Sad (Buồn bã), Neutral (Bình thường), Frustrated (Thất vọng), Excited (Hào hứng), Fearful (Sợ hãi), Surprised (Ngạc nhiên), Disgusted (Ghê tởm), Other (Khác)*. Nhãn cuối cùng được xác định theo biểu quyết đa số (majority vote).

---

## 2. Phân tích cấu trúc thư mục thực tế của `D:\DO_AN_CS\IEMOCAP_full_release`
Thư mục gốc chứa dữ liệu giải nén của IEMOCAP được tổ chức phân cấp khoa học theo từng phiên đối thoại (Session):

```text
D:\DO_AN_CS\IEMOCAP_full_release
│   README.txt (File mô tả chính thức từ USC SAIL)
│
├───Session1
│   ├───dialog
│   │   ├───EmoEvaluation (Chứa tệp nhãn đánh giá cảm xúc dạng .txt cho từng câu)
│   │   ├───transcriptions (Bản ghi hội thoại dạng text)
│   │   └───Evaluation (Báo cáo đánh giá chi tiết của từng chuyên gia)
│   └───sentences
│       ├───wav
│       │   ├───Ses01F_impro01 (Thư mục chứa các file .wav từng câu của đối thoại ngẫu hứng 01)
│       │   │   ├── Ses01F_impro01_F000.wav
│       │   │   ├── Ses01F_impro01_M000.wav
│       │   │   └── ...
│       │   └───...
│       └───ForcedAlignment (Chi tiết căn lề thời gian cấp âm vị, từ)
│
├───Session2 (Tương tự Session 1)
├───Session3 (Tương tự Session 1)
├───Session4 (Tương tự Session 1)
└───Session5 (Tương tự Session 1)
```

### Quy tắc đặt tên tệp tin hội thoại (ví dụ `Ses01F_impro01_F000.wav`):
*   `Ses01`: Session 1 (Cặp diễn viên số 1).
*   `F`: Nữ diễn viên là người đeo cảm biến MoCap trong buổi ghi hình đó (hoặc `M` nếu là Nam).
*   `impro01`: Cuộc hội thoại ngẫu hứng số 01 (hoặc `script01` nếu là hội thoại theo kịch bản).
*   `_F000`: Đoạn nói số 000 của diễn viên Nữ (Female). Nếu kết thúc bằng `_M000` là của diễn viên Nam (Male).

---

## 3. Quy trình Trích chọn & Tiền xử lý dữ liệu trong Mã nguồn

Hệ thống xử lý của dự án thực hiện tiền xử lý tín hiệu số thông qua 4 bước khép kín để biến đổi từ tệp âm thanh `.wav` thô thành tensor hình ảnh sẵn sàng đưa vào mạng CNN.

### Bước 1: Lọc dữ liệu thô (Data Filtering)
Để đảm bảo tính tự nhiên nhất của cảm xúc và sự tập trung của mô hình, mã nguồn thực hiện lọc dữ liệu theo các tiêu chí:
1.  **Chỉ lấy dữ liệu Ngẫu hứng (Improvised):** Bỏ qua dữ liệu kịch bản (`include_scripted=False`) vì dữ liệu ngẫu hứng phản ánh chân thực cao độ và ngữ điệu tự nhiên của giọng nói khi có cảm xúc.
2.  **Lọc 4 lớp cảm xúc chính:** Lọc ra 4 nhãn cảm xúc phổ biến nhất để phân loại: `ang` (Angry), `sad` (Sad), `hap` (Happy) và `neu` (Neutral). Các nhãn không có sự đồng thuận đa số (`xxx`) hoặc cảm xúc khác bị loại bỏ.
3.  **Phân tách Speaker độc lập:** Phân chia dữ liệu theo 10 Speaker (gồm Session 1-5, Nam/Nữ ký hiệu là 1M, 1F, ..., 5F) để chuẩn bị cho quá trình huấn luyện chéo.

> [!NOTE]
> **So sánh với tập dữ liệu gốc:**
> * Bộ dữ liệu IEMOCAP gốc chứa tổng cộng **10,039 câu nói (utterances)** bao gồm cả hội thoại ngẫu hứng và kịch bản cùng nhiều nhãn cảm xúc phụ khác.
> * Việc lọc lấy **hội thoại ngẫu hứng (Improvised)** và thu hẹp vào **4 lớp cảm xúc chính** làm giảm tổng số mẫu xuống **2,280 câu nói**, giúp mô hình tập trung tối đa vào các mẫu cảm xúc tự nhiên, tránh nhiễu từ các yếu tố kịch bản diễn xuất sân khấu.
> * **Về nhãn Excited (Hào hứng):** Trong một số nghiên cứu SER khác, nhãn `Excited` thường được gộp chung với `Happiness` (do sự tương đồng về arousal và valence). Tuy nhiên, mã nguồn dự án này **không gộp** hai nhãn này mà loại bỏ `Excited`, giữ nguyên nhóm `Happiness` độc lập. Điều này giải thích tại sao `Happiness` là nhóm thiểu số nhất (chỉ 284 câu), tăng tính thách cực cho mô hình phân loại và chứng minh tầm quan trọng của hàm lỗi **Focal Loss** được áp dụng sau này.

#### Thống kê lượng mẫu sau lọc (Cực kỳ chính xác):
*   **Tổng số câu nói (Utterances):** **2280 câu**.
    *   `Neutral` (Bình thường): **1099** câu (chiếm 48.2%)
    *   `Sadness` (Buồn bã): **608** câu (chiếm 26.7%)
    *   `Anger` (Tức giận): **289** câu (chiếm 12.7%)
    *   `Happiness` (Vui vẻ): **284** câu (chiếm 12.5%)

### Bước 2: Trích xuất đặc trưng âm học dạng Log-Spectrogram (DSP)
Mã nguồn sử dụng thư viện `librosa` để trích xuất đặc trưng hình ảnh tần số từ sóng âm thanh:
1.  **Resampling:** Đưa tần số lấy mẫu của tất cả các file âm thanh về chuẩn chung là $16\text{ kHz}$.
2.  **Bộ lọc tiền nhấn (Pre-emphasis Filter):** Áp dụng bộ lọc nhằm nhấn mạnh dải tần số cao (nơi chứa nhiều phụ âm và biểu cảm tinh tế của giọng nói):
    $$y(t) = x(t) - 0.97 \cdot x(t-1)$$
3.  **Biến đổi Fourier ngắn hạn (STFT):**
    *   **Cửa sổ phân tích (Window):** Hamming window.
    *   **Độ dài cửa sổ (Window Length):** $40\text{ ms}$ (tương đương $0.04 \times 16000 = 640$ mẫu sóng).
    *   **Bước trượt (Hop Length):** $10\text{ ms}$ (tương đương $160$ mẫu sóng) để tạo sự chồng lấp đặc trưng.
    *   **Số điểm FFT (N_FFT):** $800$ điểm.
4.  **Chuyển đổi Log-scale (Decibel):** Chuyển biên độ sang thang đo decibel (dB) để mô phỏng chính xác cách tai người cảm thụ âm lượng lớn nhỏ:
    $$\text{dB} = 20 \log_{10}(\text{Amplitude})$$
5.  **Lọc băng tần số:** Chỉ giữ lại **200 băng tần số thấp nhất** (trong số 401 băng tần của STFT), vì dải tần số dưới $4\text{ kHz}$ chứa hầu hết các thông tin ngữ điệu cảm xúc của con người.

### Bước 3: Phân đoạn ma trận phổ đồ (Segmentation)
Mỗi câu nói có thời lượng dài ngắn khác nhau, tạo ra các ma trận phổ đồ có chiều dài thời gian khác nhau. Để đưa vào mạng CNN, ma trận này cần có kích thước cố định:
*   Mã nguồn cắt phổ đồ thành các đoạn có kích thước cố định là **300 khung thời gian** (tương đương khoảng $3$ giây âm thanh).
*   **Đệm dữ liệu (Padding):** Nếu câu nói ngắn hơn 300 khung thời gian, phổ đồ được đệm thêm giá trị 0 (zero padding).
*   **Cắt phân đoạn:** Nếu câu nói dài hơn, nó sẽ được chia thành nhiều đoạn dài 300 khung độc lập.
*   **Tổng số phân đoạn phổ đồ thu được:** **4436 phân đoạn**.
    *   `Neutral`: **2019** phân đoạn (45.5%)
    *   `Sadness`: **1307** phân đoạn (29.5%)
    *   `Anger`: **585** phân đoạn (13.2%)
    *   `Happiness`: **525** phân đoạn (11.8%)

### Bước 4: Chuẩn hóa & Định dạng ảnh đầu vào CNN
1.  **Chuẩn hóa giá trị:** Áp dụng `MinMaxScaler` trên tập Train và chuyển tiếp sang tập Val/Test để đưa dải giá trị phổ đồ từ $[-80\text{ dB}, 0\text{ dB}]$ về khoảng $[0.0, 1.0]$.
2.  **Định dạng ảnh 3 kênh xám (Pseudo-RGB):**
    *   Mã nguồn lật ngược chiều ma trận theo trục tần số (flip) để tần số thấp nằm dưới và tần số cao nằm trên giống biểu đồ phổ tiêu chuẩn.
    *   Nhân bản dữ liệu 1 kênh này thành 3 kênh giống hệt nhau tạo thành ảnh 3 kênh màu xám (kích thước $3 \times 200 \times 300$).
3.  **Tiền xử lý ảnh của ImageNet:** Áp dụng bộ biến đổi PyTorch `transforms`:
    *   Resize ảnh về kích thước chuẩn của AlexNet/ResNet: **$256 \times 256$ pixels** (hoặc CenterCrop về $224 \times 224$ pixels tùy thuộc vào mô hình).
    *   Chuẩn hóa phân phối chuẩn theo ImageNet:
        $$\text{Ảnh chuẩn hóa} = \frac{\text{Ảnh} - \text{Mean}}{\text{Std}}$$
        Với $\text{Mean} = [0.485, 0.456, 0.406]$ và $\text{Std} = [0.229, 0.224, 0.225]$.

---

## 4. Tóm tắt luồng dữ liệu của hệ thống (Phục vụ vẽ sơ đồ báo cáo)

$$\text{Tệp âm thanh thô .wav (16kHz)} \xrightarrow{\text{Tiền nhấn (Pre-emphasis)}} x_{flt}(t)$$
$$\downarrow$$
$$x_{flt}(t) \xrightarrow{\text{STFT (Hamming, 40ms, 10ms)}} \text{Biên độ phổ } A(f, t)$$
$$\downarrow$$
$$A(f, t) \xrightarrow{\text{Thang đo Log (dB) \& Lọc 200 băng}} \text{Spectrogram (200 } \times \text{ T)}$$
$$\downarrow$$
$$\text{Spectrogram (200 } \times \text{ T)} \xrightarrow{\text{Cắt phân đoạn \& Pad 0}} \text{Phân đoạn phổ (200 } \times \text{ 300)}$$
$$\downarrow$$
$$\text{Phân đoạn phổ} \xrightarrow{\text{Chuẩn hóa [0,1] \& Nhân bản 3 kênh}} \text{Ảnh xám Pseudo-RGB (3 } \times \text{ 200 } \times \text{ 300)}$$
$$\downarrow$$
$$\text{Pseudo-RGB} \xrightarrow{\text{Resize (256x256) \& ImageNet Z-Score}} \text{Tensor đầu vào mạng Deep Learning (3 } \times \text{ 256 } \times \text{ 256)}$$
