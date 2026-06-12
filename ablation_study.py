"""
ablation_study.py
=================
Thực nghiệm ablation study (nghiên cứu loại bỏ) cho EfficientNet-B0:
  - Config A: EfficientNet + Cross Entropy (không EMix) — baseline transfer learning
  - Config B: EfficientNet + Focal Loss (không EMix)    — chỉ Focal Loss
  - Config C: EfficientNet + Focal Loss + EMix-S         — đầy đủ (đã có sẵn)
  - Config D: EfficientNet + Focal Loss + EMix-NS        — EMix-NS variant

Mục đích: Chứng minh đóng góp riêng lẻ của từng kỹ thuật (Focal Loss, EMix).
Kết quả giúp hoàn thiện mục 2.1 (EMix variants) và mục 2.3 (Focal Loss) của đồ án.
"""

import os
import sys
import subprocess
import re
import time
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PYTHON_EXE  = sys.executable
FEATURES    = "IEMOCAP_logspec200.pkl"
VAL_ID      = "1F"
TEST_ID     = "1M"
EPOCHS      = 40
BATCH_SIZE  = 64
LR          = 0.0001
SEED        = 100
GPU         = 1
RESULTS_JSON = "ablation_results.json"

# Cấu hình các thực nghiệm ablation
ABLATION_CONFIGS = [
    # (label, display_name, loss_type, emix_type)
    ("efficientnet_ce_no_emix",    "EfficientNet + CE (no EMix)",     "ce",    None),
    ("efficientnet_focal_no_emix", "EfficientNet + Focal (no EMix)",  "focal", None),
    # Config C đã có sẵn: efficientnet_focal_loss.pth
    ("efficientnet_focal_emix_ns", "EfficientNet + Focal + EMix-NS",  "focal", "emix-ns"),
    ("efficientnet_focal_emix_n",  "EfficientNet + Focal + EMix-N",   "focal", "emix-n"),
]


def run_config(label, display_name, loss_type, emix_type):
    pth_path = f"{label}.pth"
    already_trained = os.path.exists(pth_path)

    epochs_to_run = 0 if already_trained else EPOCHS

    print(f"\n{'='*60}")
    print(f"CONFIG: {display_name}")
    if already_trained:
        print(f"Da co {pth_path}. Chi chay danh gia...")
    else:
        print(f"Huan luyen {epochs_to_run} epochs...")
    print(f"{'='*60}")

    cmd = [
        PYTHON_EXE, "train_ser.py", FEATURES,
        "--ser_model",   "efficientnet",
        "--val_id",      VAL_ID,
        "--test_id",     TEST_ID,
        "--num_epochs",  str(epochs_to_run),
        "--batch_size",  str(BATCH_SIZE),
        "--lr",          str(LR),
        "--seed",        str(SEED),
        "--gpu",         str(GPU),
        "--save_label",  label,
        "--loss_type",   loss_type,
        "--pretrained",
    ]
    if emix_type:
        cmd.extend(["--emix", emix_type])

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    elapsed = time.time() - start
    stdout = result.stdout
    stderr = result.stderr

    print(stdout[-5000:] if len(stdout) > 5000 else stdout)

    if result.returncode != 0:
        print(f"[ERROR] Config '{display_name}' that bai (exit={result.returncode})")
        print(stderr[-2000:])
        return {"label": label, "name": display_name, "error": True,
                "wa": None, "ua": None, "loss": None, "elapsed": elapsed}

    # Trích xuất metrics
    metrics_match = re.search(
        r"Loss:\s*([\d.\-]+)\s*\t?WA:\s*([\d.\-]+)\s*\t?UA:\s*([\d.\-]+)", stdout
    )
    if not metrics_match:
        metrics_match = re.search(
            r"Loss:([\d.\-]+)\s+WA:\s*([\d.\-]+)\s+UA:\s*([\d.\-]+)", stdout
        )

    if metrics_match:
        test_loss = float(metrics_match.group(1))
        test_wa   = float(metrics_match.group(2))
        test_ua   = float(metrics_match.group(3))
    else:
        test_loss, test_wa, test_ua = None, None, None

    print(f"\nConfig '{display_name}' xong ({elapsed/60:.1f} min) -> WA={test_wa}%  UA={test_ua}%")

    return {
        "label": label, "name": display_name, "error": False,
        "wa": test_wa, "ua": test_ua, "loss": test_loss, "elapsed": elapsed
    }


def main():
    print("=" * 60)
    print("ABLATION STUDY — EfficientNet-B0")
    print("Ket qua cua config C (Focal+EMix-S) da co san.")
    print("=" * 60)

    # Kết quả đã biết từ thực nghiệm trước
    known_result = {
        "label": "efficientnet_focal_loss",
        "name": "EfficientNet + Focal + EMix-S (Confirmed)",
        "error": False,
        "wa": 80.49, "ua": 73.25, "loss": 0.2786,
        "elapsed": 0
    }

    all_results = [known_result]

    for label, display_name, loss_type, emix_type in ABLATION_CONFIGS:
        res = run_config(label, display_name, loss_type, emix_type)
        all_results.append(res)

        # Nghỉ 8 giây giữa các config
        print("Nghi 8 giay cho GPU lam mat...")
        time.sleep(8)

    # In bảng tổng kết
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS — EfficientNet-B0")
    print("=" * 70)
    print(f"{'Config':<45} {'WA (%)':>10} {'UA (%)':>10}")
    print("-" * 70)
    for r in all_results:
        wa_s = f"{r['wa']:.2f}" if r['wa'] is not None else "Error"
        ua_s = f"{r['ua']:.2f}" if r['ua'] is not None else "Error"
        print(f"  {r['name']:<43} {wa_s:>10} {ua_s:>10}")
    print("=" * 70)

    # Lưu JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu ket qua vao: {json_path}")
    print("\nAblation Study hoan thanh!")


if __name__ == "__main__":
    main()
