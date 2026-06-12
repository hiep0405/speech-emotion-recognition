import os
import sys
import pickle
import time
import subprocess
import re
import shutil

# Tránh xung đột thư viện OpenMP trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Thêm đường dẫn hiện tại vào sys.path để python nhận diện các module cục bộ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    features_file = 'IEMOCAP_logspec200.pkl'
    val_id = '1F'
    # Phải đặt đúng test_id và val_id tương thích với các mô hình trước đó
    # Trong logspec dataset, test_id = '1M', val_id = '1F'
    test_id = '1M' 
    num_epochs = 40  
    batch_size = 64  
    lr = 0.0001
    seed = 100
    gpu = 1
    python_exe = r"D:\miniforge3\envs\pt310\python.exe"

    print("=" * 60)
    print("BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG CÁC MÔ HÌNH HỌC SÂU (SER)")
    print(f"Cấu hình mặc định: Epochs={num_epochs}, Batch Size={batch_size}")
    print("=" * 60)

    # 6 thí nghiệm tương ứng với 6 dòng trong bảng báo cáo
    models_to_test = [
        # (model_key, display_name, loss_type, emix_type, pth_filename)
        ('alexnet', 'AlexNet (Baseline)', 'ce', None, 'alexnet_baseline'),
        ('alexnet_gap', 'AlexNet GAP', 'ce', None, 'alexnet_gap_cross_entropy'),
        ('resnet', 'ResNet-18', 'focal', 'emix-s', 'resnet_focal_loss'),
        ('densenet', 'DenseNet-121', 'focal', 'emix-s', 'densenet_focal_loss'),
        ('efficientnet', 'EfficientNet-B0', 'focal', 'emix-s', 'efficientnet_focal_loss'),
        ('mobilenet', 'MobileNet-V2', 'focal', 'emix-s', 'mobilenet_focal_loss')
    ]

    results_table = []

    for model_key, model_name, loss_type, emix_type, save_label in models_to_test:
        pth_path = f"{save_label}.pth"
        already_trained = os.path.exists(pth_path)
        
        epochs_to_run = 0 if already_trained else num_epochs
        
        print("\n" + "#" * 60)
        if already_trained:
            print(f"Đã phát hiện {pth_path}. CHỈ CHẠY ĐÁNH GIÁ (EVALUATION)...")
        else:
            print(f"Chưa có {pth_path}. CHẠY HUẤN LUYỆN {epochs_to_run} EPOCHS...")
        print(f"Mô hình: {model_name} (Loss={loss_type}, Augmentation={emix_type})")
        print("#" * 60)

        # Xây dựng lệnh command line cho subprocess
        cmd = [
            python_exe, 'train_ser.py', features_file,
            '--ser_model', model_key,
            '--val_id', val_id,
            '--test_id', test_id,
            '--num_epochs', str(epochs_to_run),
            '--batch_size', str(batch_size),
            '--lr', str(lr),
            '--seed', str(seed),
            '--gpu', str(gpu),
            '--save_label', save_label,
            '--loss_type', loss_type,
        ]
        
        if emix_type:
            cmd.extend(['--emix', emix_type])
        if model_key in ['resnet', 'densenet', 'efficientnet', 'mobilenet']:
            cmd.append('--pretrained')

        # Chạy subprocess
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            duration = time.time() - start_time
            stdout = result.stdout
            stderr = result.stderr
            
            # In logs từ subprocess
            print(stdout)
            if result.returncode != 0:
                print(f"Lỗi khi chạy lệnh (Exit code {result.returncode}):\n{stderr}")
                test_loss, test_wa, test_ua, param_count_m = "Error", 0.0, 0.0, "N/A"
            else:
                # Trích xuất số tham số
                param_match = re.search(r"Number of trainable parameters:\s*(\d+)", stdout)
                if param_match:
                    params_val = int(param_match.group(1))
                    param_count_m = f"{params_val / 1_000_000:.2f}M"
                else:
                    param_count_m = "N/A"

                # Trích xuất kết quả đánh giá Test
                metrics_match = re.search(r"Loss:\s*([\d\.-]+)\s*WA:\s*([\d\.-]+)\s*UA:\s*([\d\.-]+)", stdout)
                if metrics_match:
                    test_loss = metrics_match.group(1)
                    test_wa = float(metrics_match.group(2))
                    test_ua = float(metrics_match.group(3))
                else:
                    test_loss, test_wa, test_ua = "Error", 0.0, 0.0
                
                print(f"Hoàn thành {model_name} trong {duration:.2f}s. Kết quả: WA={test_wa}%, UA={test_ua}%")

        except Exception as e:
            print(f"Lỗi hệ thống khi gọi subprocess: {e}")
            test_loss, test_wa, test_ua, param_count_m = "Error", 0.0, 0.0, "N/A"

        # Lưu lại kết quả
        results_table.append({
            'name': model_name,
            'loss': 'Cross Entropy' if loss_type == 'ce' else 'Focal Loss',
            'aug': 'EMix-S' if emix_type == 'emix-s' else 'Không',
            'params': param_count_m,
            'test_loss': test_loss,
            'wa': test_wa,
            'ua': test_ua
        })

        # Nghỉ 5 giây giữa các mô hình để làm mát GPU
        time.sleep(5)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH HUẤN LUYỆN TOÀN BỘ CÁC MÔ HÌNH")
    print("=" * 60)

    # Cập nhật kết quả vào muc_tieu_do_an.md và walkthrough.md
    try:
        update_markdown_tables(results_table)
    except Exception as e:
        print(f"Lỗi khi cập nhật file báo cáo: {e}")

def update_markdown_tables(results):
    # Tạo nội dung bảng Markdown
    table_lines = [
        "| TT | Mô hình kiến trúc | Hàm Loss sử dụng | Tăng cường dữ liệu | Số lượng tham số | Test Loss | Weighted Accuracy (WA) | Unweighted Accuracy (UA) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(results):
        table_lines.append(
            f"| {idx+1} | {r['name']} | {r['loss']} | {r['aug']} | {r['params']} | {r['test_loss']} | {r['wa']:.2f}% | {r['ua']:.2f}% |"
        )
    table_content = "\n".join(table_lines)

    # Cập nhật muc_tieu_do_an.md
    target_file = 'muc_tieu_do_an.md'
    if os.path.exists(target_file):
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        start_idx = content.find("| TT | Mô hình kiến trúc |")
        if start_idx != -1:
            end_idx = content.find("\n\n", start_idx)
            if end_idx == -1:
                end_idx = len(content)
            new_content = content[:start_idx] + table_content + content[end_idx:]
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("Đã cập nhật bảng so sánh vào muc_tieu_do_an.md")

    # Cập nhật walkthrough.md của conversation hiện tại
    walkthrough_file = r"C:\Users\ADMIN\.gemini\antigravity-ide\brain\03c6ed6b-0a37-4f8b-a942-0062e0fa6986\walkthrough.md"
    
    # Tạo nội dung bảng Markdown cho walkthrough
    wt_table_lines = [
        "| Tên mô hình | Số tham số (M) | Loss Function | Data Augmentation | Test Loss | Weighted Accuracy | Unweighted Accuracy (UA) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in results:
        wt_table_lines.append(
            f"| **{r['name']}** | {r['params']} | {r['loss']} | {r['aug']} | {r['test_loss']} | {r['wa']:.2f}% | {r['ua']:.2f}% |"
        )
    new_wt_table = "\n".join(wt_table_lines)

    # Nếu chưa có file walkthrough.md, tạo mới
    if not os.path.exists(walkthrough_file):
        with open(walkthrough_file, 'w', encoding='utf-8') as f:
            f.write("# BÁO CÁO THỰC NGHIỆM ĐỒ ÁN SER\n\n### Kết quả các mô hình so sánh đối chứng:\n\n")
            f.write(new_wt_table)
            f.write("\n\n")
    else:
        # Nếu đã có, ghi đè hoặc chèn vào
        with open(walkthrough_file, 'r', encoding='utf-8') as f:
            wt_content = f.read()
        wt_start = wt_content.find("| Tên mô hình |")
        if wt_start != -1:
            wt_end = wt_content.find("\n\n", wt_start)
            if wt_end == -1:
                wt_end = len(wt_content)
            new_wt_content = wt_content[:wt_start] + new_wt_table + wt_content[wt_end:]
            with open(walkthrough_file, 'w', encoding='utf-8') as f:
                f.write(new_wt_content)
        else:
            with open(walkthrough_file, 'a', encoding='utf-8') as f:
                f.write("\n\n### Kết quả các mô hình so sánh đối chứng:\n\n" + new_wt_table + "\n\n")
    print("Đã cập nhật bảng so sánh vào walkthrough.md")

    # Sao chép các tệp ảnh curves và conf mới tạo sang thư mục artifact
    artifact_dir = os.path.dirname(walkthrough_file)
    for r in results:
        label = r['name'].split()[0].lower().replace("-18", "").replace("-121", "").replace("-b0", "").replace("-v2", "")
        if "GAP" in r['name']:
            label = "alexnet_gap"
        
        curves_img = f"{label}_{r['loss'].lower().replace(' ', '_')}_curves.png"
        conf_img = f"{label}_{r['loss'].lower().replace(' ', '_')}_conf.png"
        
        if os.path.exists(curves_img):
            shutil.copy(curves_img, os.path.join(artifact_dir, curves_img))
        if os.path.exists(conf_img):
            shutil.copy(conf_img, os.path.join(artifact_dir, conf_img))
    print("Đã sao chép các tệp hình ảnh curves và conf vào thư mục artifact.")

if __name__ == '__main__':
    main()
