"""
5-Fold Speaker-Independent Cross-Validation cho EfficientNet-B0
================================================================
Mỗi fold: 1 session làm test, 1 session làm val, 8 speakers còn lại làm train.
Chạy tuần tự qua subprocess để tránh VRAM leak giữa các fold.

Fold assignment (theo chuẩn IEMOCAP 5-session):
  Fold 1: val=1F, test=1M
  Fold 2: val=2F, test=2M
  Fold 3: val=3F, test=3M
  Fold 4: val=4F, test=4M
  Fold 5: val=5F, test=5M
"""

import os
import sys
import subprocess
import re
import time
import json
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PYTHON_EXE  = r"D:\miniforge3\envs\pt310\python.exe"
FEATURES    = "IEMOCAP_logspec200.pkl"
MODEL       = "efficientnet"
LOSS        = "focal"
EMIX        = "emix-s"
EPOCHS      = 60        # đủ để hội tụ, không quá nhiều
BATCH_SIZE  = 64
LR          = 0.0001
SEED        = 100
GPU         = 1
PRETRAINED  = True

# Các fold theo chuẩn Speaker-Independent
FOLDS = [
    {"fold": 1, "val_id": "1F", "test_id": "1M"},
    {"fold": 2, "val_id": "2F", "test_id": "2M"},
    {"fold": 3, "val_id": "3F", "test_id": "3M"},
    {"fold": 4, "val_id": "4F", "test_id": "4M"},
    {"fold": 5, "val_id": "5F", "test_id": "5M"},
]

RESULTS_JSON = "crossval_efficientnet_results.json"


def run_fold(fold_info):
    """Chạy huấn luyện một fold qua subprocess. Trả về dict kết quả."""
    fold_n   = fold_info["fold"]
    val_id   = fold_info["val_id"]
    test_id  = fold_info["test_id"]
    label    = f"efficientnet_focal_cv_fold{fold_n}"

    print(f"\n{'='*60}")
    print(f"FOLD {fold_n}/5  |  val={val_id}  test={test_id}  → {label}.pth")
    print(f"{'='*60}")

    cmd = [
        PYTHON_EXE, "train_ser.py", FEATURES,
        "--ser_model",   MODEL,
        "--val_id",      val_id,
        "--test_id",     test_id,
        "--num_epochs",  str(EPOCHS),
        "--batch_size",  str(BATCH_SIZE),
        "--lr",          str(LR),
        "--seed",        str(SEED),
        "--gpu",         str(GPU),
        "--save_label",  label,
        "--loss_type",   LOSS,
        "--emix",        EMIX,
        "--pretrained",
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    elapsed = time.time() - start

    stdout = result.stdout
    stderr = result.stderr

    # In đầy đủ output
    print(stdout[-6000:] if len(stdout) > 6000 else stdout)

    if result.returncode != 0:
        print(f"[ERROR] Fold {fold_n} thất bại (exit={result.returncode})")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
        return {
            "fold": fold_n, "val_id": val_id, "test_id": test_id,
            "wa": None, "ua": None, "loss": None,
            "params": None, "elapsed_s": elapsed, "label": label, "error": True
        }

    # --- Trích xuất metrics ---
    param_match = re.search(r"Number of trainable parameters:\s*(\d+)", stdout)
    params_m = f"{int(param_match.group(1))/1e6:.2f}M" if param_match else "N/A"

    metrics_match = re.search(
        r"Loss:\s*([\d.\-]+)\s*\t?WA:\s*([\d.\-]+)\s*\t?UA:\s*([\d.\-]+)",
        stdout
    )
    if metrics_match:
        test_loss = float(metrics_match.group(1))
        test_wa   = float(metrics_match.group(2))
        test_ua   = float(metrics_match.group(3))
    else:
        # fallback: dòng cuối "Loss:X.XXXX  WA: XX.XX  UA: XX.XX"
        alt_match = re.search(
            r"Loss:([\d.\-]+)\s+WA:\s*([\d.\-]+)\s+UA:\s*([\d.\-]+)",
            stdout
        )
        if alt_match:
            test_loss = float(alt_match.group(1))
            test_wa   = float(alt_match.group(2))
            test_ua   = float(alt_match.group(3))
        else:
            test_loss, test_wa, test_ua = None, None, None

    print(f"\n✓ Fold {fold_n} hoàn thành trong {elapsed/60:.1f} phút → "
          f"WA={test_wa}%  UA={test_ua}%")

    return {
        "fold": fold_n, "val_id": val_id, "test_id": test_id,
        "wa": test_wa, "ua": test_ua, "loss": test_loss,
        "params": params_m, "elapsed_s": elapsed, "label": label, "error": False
    }


def compute_stats(values):
    """Tính mean và std của list, bỏ qua None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    return float(np.mean(valid)), float(np.std(valid))


def print_summary(fold_results):
    was = [r["wa"] for r in fold_results]
    uas = [r["ua"] for r in fold_results]
    wa_mean, wa_std = compute_stats(was)
    ua_mean, ua_std = compute_stats(uas)

    print("\n" + "="*70)
    print("KẾT QUẢ 5-FOLD CROSS-VALIDATION — EfficientNet-B0 + Focal + EMix-S")
    print("="*70)
    print(f"{'Fold':<8} {'Val':<6} {'Test':<6} {'WA (%)':>10} {'UA (%)':>10}")
    print("-"*45)
    for r in fold_results:
        wa_s = f"{r['wa']:.2f}" if r['wa'] is not None else "Error"
        ua_s = f"{r['ua']:.2f}" if r['ua'] is not None else "Error"
        print(f"  {r['fold']:<6} {r['val_id']:<6} {r['test_id']:<6} {wa_s:>10} {ua_s:>10}")
    print("-"*45)
    print(f"  {'MEAN':<12}{'':6} "
          f"{f'{wa_mean:.2f}':>10} {f'{ua_mean:.2f}':>10}")
    print(f"  {'STD':<12}{'':6} "
          f"{f'±{wa_std:.2f}':>10} {f'±{ua_std:.2f}':>10}")
    print("="*70)
    print(f"\n5-Fold WA: {wa_mean:.2f}% ± {wa_std:.2f}%")
    print(f"5-Fold UA: {ua_mean:.2f}% ± {ua_std:.2f}%")
    return wa_mean, wa_std, ua_mean, ua_std


def save_results_json(fold_results, wa_mean, wa_std, ua_mean, ua_std):
    """Lưu kết quả dạng JSON để dùng cho script vẽ đồ thị."""
    output = {
        "model": "EfficientNet-B0",
        "loss": "Focal Loss",
        "augmentation": "EMix-S",
        "epochs": EPOCHS,
        "folds": fold_results,
        "summary": {
            "wa_mean": wa_mean, "wa_std": wa_std,
            "ua_mean": ua_mean, "ua_std": ua_std,
        }
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_JSON)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu kết quả JSON vào: {path}")


def main():
    print("="*60)
    print("BẮT ĐẦU 5-FOLD CROSS-VALIDATION — EfficientNet-B0")
    print(f"Epochs/fold={EPOCHS}, Batch={BATCH_SIZE}, LR={LR}")
    print("="*60)

    fold_results = []
    total_start = time.time()

    for fold_info in FOLDS:
        fold_res = run_fold(fold_info)
        fold_results.append(fold_res)

        # Lưu kết quả trung gian sau mỗi fold (an toàn nếu bị ngắt)
        tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "crossval_tmp_results.json"
        )
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(fold_results, f, indent=2, ensure_ascii=False)

        # Nghỉ 10 giây để GPU giải phóng VRAM hoàn toàn
        print("\nNghỉ 10 giây để GPU làm mát...")
        time.sleep(10)

    total_elapsed = time.time() - total_start
    print(f"\nTổng thời gian: {total_elapsed/60:.1f} phút")

    wa_mean, wa_std, ua_mean, ua_std = print_summary(fold_results)
    save_results_json(fold_results, wa_mean, wa_std, ua_mean, ua_std)

    # Xóa file tạm
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crossval_tmp_results.json")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("\n✅ 5-Fold Cross-Validation hoàn thành!")
    print(f"   EfficientNet-B0 (Focal+EMix-S): WA={wa_mean:.2f}%±{wa_std:.2f}%  UA={ua_mean:.2f}%±{ua_std:.2f}%")


if __name__ == "__main__":
    main()
