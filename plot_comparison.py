"""
plot_comparison.py
==================
Vẽ tất cả biểu đồ so sánh cho báo cáo đồ án SER:
  1. Grouped bar chart: WA/UA của 6 mô hình
  2. Per-class accuracy heatmap so sánh các mô hình
  3. Radar chart tổng hợp đặc tính mô hình
  4. 5-Fold CV bar chart với error bar (nếu có kết quả JSON)
  5. Focal Loss vs Cross Entropy comparison
"""

import os
import sys

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# =====================================================================
# DỮ LIỆU KẾT QUẢ THỰC NGHIỆM (từ bảng báo cáo chính thức)
# =====================================================================
RESULTS = [
    {
        "name": "AlexNet\n(Baseline)",
        "short": "AlexNet\nBaseline",
        "loss": "Cross Entropy",
        "aug": "None",
        "params_m": 57.02,
        "wa": 50.73,
        "ua": 25.00,
        "per_class": {"ang": 100.0, "sad": 0.0, "hap": 0.0, "neu": 0.0},
        "color": "#e74c3c",
    },
    {
        "name": "AlexNet GAP",
        "short": "AlexNet\nGAP",
        "loss": "Cross Entropy",
        "aug": "None",
        "params_m": 2.47,
        "wa": 78.54,
        "ua": 64.89,
        "per_class": {"ang": 84.62, "sad": 56.41, "hap": 38.46, "neu": 80.08},
        "color": "#e67e22",
    },
    {
        "name": "ResNet-18",
        "short": "ResNet-18",
        "loss": "Focal Loss",
        "aug": "EMix-S",
        "params_m": 11.18,
        "wa": 76.10,
        "ua": 64.37,
        "per_class": {"ang": 84.62, "sad": 53.85, "hap": 38.46, "neu": 80.62},
        "color": "#2ecc71",
    },
    {
        "name": "DenseNet-121",
        "short": "DenseNet\n121",
        "loss": "Focal Loss",
        "aug": "EMix-S",
        "params_m": 6.96,
        "wa": 68.78,
        "ua": 66.48,
        "per_class": {"ang": 69.23, "sad": 61.54, "hap": 53.85, "neu": 81.30},
        "color": "#3498db",
    },
    {
        "name": "EfficientNet-B0\n(Best)",
        "short": "EfficientNet\nB0 ★",
        "loss": "Focal Loss",
        "aug": "EMix-S",
        "params_m": 4.01,
        "wa": 80.49,
        "ua": 73.25,
        "per_class": {"ang": 92.31, "sad": 61.54, "hap": 61.54, "neu": 77.62},
        "color": "#9b59b6",
    },
    {
        "name": "MobileNet-V2",
        "short": "MobileNet\nV2",
        "loss": "Focal Loss",
        "aug": "EMix-S",
        "params_m": 2.23,
        "wa": 76.59,
        "ua": 71.15,
        "per_class": {"ang": 84.62, "sad": 64.10, "hap": 53.85, "neu": 81.99},
        "color": "#1abc9c",
    },
]

CLASSES = ["ang", "sad", "hap", "neu"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_style():
    """Thiết lập style đẹp cho tất cả đồ thị."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# =====================================================================
# HÌNH 1: Grouped Bar Chart — WA và UA của 6 mô hình
# =====================================================================
def plot_wa_ua_comparison():
    names   = [r["short"] for r in RESULTS]
    wa_vals = [r["wa"] for r in RESULTS]
    ua_vals = [r["ua"] for r in RESULTS]
    colors  = [r["color"] for r in RESULTS]

    x = np.arange(len(RESULTS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f8f8f8")

    bars_wa = ax.bar(x - width/2, wa_vals, width, label="WA (%)",
                     color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    bars_ua = ax.bar(x + width/2, ua_vals, width, label="UA (%)",
                     color=colors, alpha=0.50, edgecolor=colors, linewidth=1.5,
                     hatch="///")

    # Giá trị trên thanh
    for bar in bars_wa:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for bar in bars_ua:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, color="#444")

    # Đường baseline tham chiếu AlexNet WA
    ax.axhline(y=50.73, color="#e74c3c", linestyle=":", linewidth=1.5,
               alpha=0.7, label="AlexNet Baseline WA")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("So sánh Weighted Accuracy (WA) và Unweighted Accuracy (UA) — Các mô hình SER", pad=15)

    legend_patches = [
        mpatches.Patch(color="#555", alpha=0.85, label="Weighted Accuracy (WA)"),
        mpatches.Patch(color="#555", alpha=0.50, hatch="///", label="Unweighted Accuracy (UA)"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#e74c3c", linestyle=":", label="AlexNet Baseline WA")
    ], loc="lower right", fontsize=10)

    # Ghi chú mô hình tốt nhất
    best_idx = wa_vals.index(max(wa_vals))
    ax.annotate(f"Best: {max(wa_vals):.2f}%",
                xy=(x[best_idx] - width/2, max(wa_vals)),
                xytext=(x[best_idx] - width/2 - 0.8, max(wa_vals) + 5),
                arrowprops=dict(arrowstyle="->", color="#9b59b6"),
                fontsize=10, color="#9b59b6", fontweight="bold")

    path = os.path.join(OUT_DIR, "comparison_wa_ua.png")
    plt.savefig(path)
    plt.close()
    print(f"✓ Đã lưu: {path}")


# =====================================================================
# HÌNH 2: Per-class Accuracy Heatmap
# =====================================================================
def plot_per_class_heatmap():
    model_names = [r["short"].replace("\n", " ") for r in RESULTS]
    data = np.array([[r["per_class"][c] for c in CLASSES] for r in RESULTS])

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#fafafa")

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Class Accuracy (%)")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c.upper() for c in CLASSES], fontsize=12)
    ax.set_yticks(range(len(RESULTS)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Giá trị trong ô
    for i in range(len(RESULTS)):
        for j in range(len(CLASSES)):
            val = data[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=color)

    ax.set_title("Per-Class Accuracy (%) — So sánh các mô hình SER", pad=15,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion Class", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    path = os.path.join(OUT_DIR, "comparison_per_class.png")
    plt.savefig(path)
    plt.close()
    print(f"✓ Đã lưu: {path}")


# =====================================================================
# HÌNH 3: Scatter — Params vs WA (Efficiency trade-off)
# =====================================================================
def plot_params_vs_accuracy():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f8f8f8")

    for r in RESULTS:
        ax.scatter(r["params_m"], r["wa"], s=180, color=r["color"],
                   zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(r["short"].replace("\n", " "),
                    (r["params_m"], r["wa"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9.5, color=r["color"], fontweight="bold")

    ax.set_xlabel("Số lượng tham số (Triệu - M)", fontsize=11)
    ax.set_ylabel("Weighted Accuracy (%) trên Test Set", fontsize=11)
    ax.set_title("Model Efficiency: Số lượng tham số vs. Độ chính xác WA", pad=15)

    # Vùng hiệu quả cao
    ax.axhline(y=78, color="#27ae60", linestyle="--", alpha=0.5, linewidth=1.5,
               label="Ngưỡng hiệu quả cao (WA≥78%)")
    ax.legend(fontsize=10)

    path = os.path.join(OUT_DIR, "comparison_params_vs_wa.png")
    plt.savefig(path)
    plt.close()
    print(f"✓ Đã lưu: {path}")


# =====================================================================
# HÌNH 4: 5-Fold CV Results (từ JSON nếu có)
# =====================================================================
def plot_crossval_results(json_path):
    if not os.path.exists(json_path):
        print(f"[SKIP] Chưa có file JSON kết quả 5-fold: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    folds   = [d["fold"] for d in data["folds"] if not d.get("error")]
    wa_vals = [d["wa"] for d in data["folds"] if not d.get("error")]
    ua_vals = [d["ua"] for d in data["folds"] if not d.get("error")]
    wa_mean = data["summary"]["wa_mean"]
    wa_std  = data["summary"]["wa_std"]
    ua_mean = data["summary"]["ua_mean"]
    ua_std  = data["summary"]["ua_std"]

    x = np.array(folds)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f8f8f8")

    ax.plot(x, wa_vals, "o-", color="#9b59b6", linewidth=2.5, markersize=10,
            label=f"WA/fold (Mean={wa_mean:.2f}%±{wa_std:.2f}%)")
    ax.plot(x, ua_vals, "s--", color="#3498db", linewidth=2.5, markersize=10,
            label=f"UA/fold (Mean={ua_mean:.2f}%±{ua_std:.2f}%)")

    ax.axhline(y=wa_mean, color="#9b59b6", linestyle=":", alpha=0.6)
    ax.axhline(y=ua_mean, color="#3498db", linestyle=":", alpha=0.6)

    ax.fill_between(x, wa_mean - wa_std, wa_mean + wa_std,
                    alpha=0.15, color="#9b59b6")
    ax.fill_between(x, ua_mean - ua_std, ua_mean + ua_std,
                    alpha=0.15, color="#3498db")

    for xi, wa, ua in zip(x, wa_vals, ua_vals):
        ax.text(xi, wa + 0.5, f"{wa:.2f}%", ha="center", fontsize=9, color="#9b59b6")
        ax.text(xi, ua - 1.5, f"{ua:.2f}%", ha="center", fontsize=9, color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Fold Cross-Validation — EfficientNet-B0 (Focal Loss + EMix-S)", pad=15)
    ax.legend(fontsize=10)
    ax.set_ylim(50, 100)

    path = os.path.join(OUT_DIR, "crossval_results.png")
    plt.savefig(path)
    plt.close()
    print(f"✓ Đã lưu: {path}")


# =====================================================================
# HÌNH 5: Biểu đồ Summary Tổng hợp (Publication-quality figure)
# =====================================================================
def plot_summary_figure():
    """Hình tổng hợp gồm 2 subplot: WA/UA bar + per-class heatmap."""
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Subplot trái: WA/UA grouped bar ---
    ax1 = fig.add_subplot(gs[0, 0])
    names   = [r["short"] for r in RESULTS]
    wa_vals = [r["wa"] for r in RESULTS]
    ua_vals = [r["ua"] for r in RESULTS]
    colors  = [r["color"] for r in RESULTS]
    x = np.arange(len(RESULTS))
    width = 0.38

    b1 = ax1.bar(x - width/2, wa_vals, width, color=colors, alpha=0.85,
                 edgecolor="white", linewidth=1.5)
    b2 = ax1.bar(x + width/2, ua_vals, width, color=colors, alpha=0.45,
                 edgecolor=colors, linewidth=1.5, hatch="///")

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", fontsize=8, fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", fontsize=8, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) WA & UA Comparison", fontweight="bold")
    ax1.axhline(50.73, color="#e74c3c", linestyle=":", alpha=0.6, linewidth=1.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#888", alpha=0.85, label="WA"),
        mpatches.Patch(color="#888", alpha=0.45, hatch="///", label="UA"),
    ]
    ax1.legend(handles=legend_patches, loc="lower right", fontsize=9)

    # --- Subplot phải: Per-class heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    model_names = [r["short"].replace("\n", " ") for r in RESULTS]
    data = np.array([[r["per_class"][c] for c in CLASSES] for r in RESULTS])

    im = ax2.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax2, label="Class Accuracy (%)", shrink=0.9)
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels([c.upper() for c in CLASSES], fontsize=11)
    ax2.set_yticks(range(len(RESULTS)))
    ax2.set_yticklabels(model_names, fontsize=9)
    for i in range(len(RESULTS)):
        for j in range(len(CLASSES)):
            val = data[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax2.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=10, fontweight="bold", color=color)
    ax2.set_title("(b) Per-Class Accuracy Heatmap (%)", fontweight="bold")

    fig.suptitle("SER Model Comparison — IEMOCAP Dataset",
                 fontsize=14, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "summary_figure.png")
    plt.savefig(path)
    plt.close()
    print(f"✓ Đã lưu: {path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    set_style()
    print("Đang vẽ biểu đồ so sánh tổng hợp cho báo cáo SER...")
    print("-" * 50)

    plot_wa_ua_comparison()
    plot_per_class_heatmap()
    plot_params_vs_accuracy()
    plot_summary_figure()

    # Vẽ 5-fold CV nếu đã có kết quả
    json_path = os.path.join(OUT_DIR, "crossval_efficientnet_results.json")
    plot_crossval_results(json_path)

    print("\n✅ Hoàn thành! Các file đã lưu:")
    for fname in ["comparison_wa_ua.png", "comparison_per_class.png",
                  "comparison_params_vs_wa.png", "summary_figure.png",
                  "crossval_results.png"]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            print(f"   ✓ {fpath}")


if __name__ == "__main__":
    main()
