import sys
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import SER_AlexNet

# Cấu hình các hằng số (phải khớp với lúc train)
EMOTIONS = {0: 'Tức giận (Anger)', 1: 'Buồn bã (Sadness)', 2: 'Vui vẻ (Happiness)', 3: 'Bình thường (Neutral)'}
PARAMS = {
    'window': 'hamming',
    'win_length': 40,  # msec
    'hop_length': 10,  # msec
    'ndft': 800,
    'nfreq': 200,
    'segment_size': 300
}

def preprocess_audio(audio_path):
    # 1. Load audio
    x, sr = librosa.load(audio_path, sr=None)
    
    # 2. Pre-emphasis filter
    x = librosa.effects.preemphasis(x, zi=[0.0])
    
    # 3. Extract Log-Spectrogram
    win_len = int((PARAMS['win_length'] / 1000) * sr)
    hop_len = int((PARAMS['hop_length'] / 1000) * sr)
    spec = np.abs(librosa.stft(x, n_fft=PARAMS['ndft'], hop_length=hop_len,
                               win_length=win_len, window=PARAMS['window']))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    spec = spec[:PARAMS['nfreq']] # Lấy 200 bins đầu tiên
    
    # 4. Normalize (MinMax -80 to 0 -> 0 to 1)
    spec = (spec + 80.0) / 80.0
    spec = np.clip(spec, 0.0, 1.0)
    
    # 5. Phân đoạn (Segmentation)
    # Vì model nhận đầu vào là ảnh 300 frames, ta sẽ cắt/pad đoạn audio
    time_frames = spec.shape[1]
    if time_frames < PARAMS['segment_size']:
        spec = np.pad(spec, ((0, 0), (0, PARAMS['segment_size'] - time_frames)), mode='constant')
    else:
        # Lấy đoạn giữa nếu audio dài
        start = (time_frames - PARAMS['segment_size']) // 2
        spec = spec[:, start:start + PARAMS['segment_size']]
    
    # 6. Chuyển thành ảnh RGB cho AlexNet (Flip freq axis, repeat 3 channels)
    spec = np.flip(spec, axis=0) # Tần số thấp ở dưới
    spec_uint8 = (spec * 255.0).astype(np.uint8)
    img = Image.fromarray(spec_uint8, mode='L').convert('RGB')
    
    # 7. AlexNet Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return preprocess(img).unsqueeze(0) # Thêm dimension batch

def predict(audio_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo model
    model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=False).to(device)
    
    # Load trọng số
    print(f"--- Đang tải model từ: {model_path} ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Xử lý âm thanh
    input_tensor = preprocess_audio(audio_path).to(device)
    
    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    print("\n" + "="*30)
    print(f"KẾT QUẢ DỰ ĐOÁN CHO: {audio_path}")
    print("="*30)
    for i, label in EMOTIONS.items():
        print(f"{label:<20}: {probs[0][i]*100:>6.2f}%")
    print("-" * 30)
    print(f"==> CẢM XÚC CHÍNH: {EMOTIONS[pred.item()]} ({conf.item()*100:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python test_demo.py <đường_dẫn_file_wav>")
    else:
        wav_file = sys.argv[1]
        # Mặc định dùng model tốt nhất cuối cùng
        predict(wav_file, 'alexnet_final.pth')
