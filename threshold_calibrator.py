"""
===========================================================
VisionRestore AI
Phase 2.5 : Threshold Calibration
===========================================================

This script analyzes every processed image and computes
quality metric distributions for threshold calibration.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from quality_analyzer.py
from quality_analyzer import (
    load_image,
    analyze_image
)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_ROOT = Path(r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/data/processed")

OUTPUT_DIR = Path("reports")

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# ANALYZE DATASET
# ==========================================================

def analyze_dataset():

    rows = []

    folders = [
        "original",
        "gaussian_blur",
        "motion_blur",
        "gaussian_noise",
        "salt_pepper",
        "low_light",
        "jpeg",
        "combined"
    ]

    for folder in folders:

        folder_path = DATASET_ROOT / folder

        print(f"\nAnalyzing {folder} ...")

        for image_path in folder_path.glob("*"):

            image = load_image(image_path)

            report = analyze_image(image)

            rows.append({

                "folder": folder,

                "image": image_path.name,

                "blur": report["Blur Score"],

                "noise": report["Noise Score"],

                "brightness": report["Brightness"],

                "contrast": report["Contrast"],

                "entropy": report["Entropy"]

            })

    return pd.DataFrame(rows)


# ==========================================================
# PRINT STATISTICS
# ==========================================================

def print_statistics(df):

    metrics = [
        "blur",
        "noise",
        "brightness",
        "contrast",
        "entropy"
    ]

    print("\n")
    print("=" * 70)
    print("QUALITY METRIC STATISTICS")
    print("=" * 70)

    for folder in df["folder"].unique():

        print(f"\n{folder.upper()}")

        subset = df[df["folder"] == folder]

        for metric in metrics:

            print(
                f"{metric:12s} | "
                f"Mean={subset[metric].mean():8.2f} "
                f"Std={subset[metric].std():8.2f} "
                f"Min={subset[metric].min():8.2f} "
                f"Max={subset[metric].max():8.2f}"
            )


# ==========================================================
# SAVE CSV
# ==========================================================

def save_csv(df):

    output = OUTPUT_DIR / "quality_metrics.csv"

    df.to_csv(output, index=False)

    print(f"\n[OK] CSV saved to: {output}")


# ==========================================================
# HISTOGRAMS
# ==========================================================

def plot_histograms(df):

    metrics = [
        "blur",
        "noise",
        "brightness",
        "contrast",
        "entropy"
    ]

    for metric in metrics:

        plt.figure(figsize=(8,5))

        for folder in df["folder"].unique():

            subset = df[df["folder"] == folder]

            plt.hist(
                subset[metric],
                bins=25,
                alpha=0.5,
                label=folder
            )

        plt.title(metric.upper())

        plt.xlabel(metric)

        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{metric}_histogram.png",
            dpi=300
        )

        plt.show()


# ==========================================================
# RECOMMEND THRESHOLDS
# ==========================================================

def recommend_thresholds(df):

    print("\n")
    print("=" * 70)
    print("SUGGESTED THRESHOLDS")
    print("=" * 70)

    original = df[df["folder"] == "original"]

    print("\nBlur")
    print(
        f"Suggested threshold ≈ "
        f"{original['blur'].mean() - original['blur'].std():.2f}"
    )

    print("\nNoise")
    print(
        f"Suggested threshold ≈ "
        f"{original['noise'].mean() + original['noise'].std():.2f}"
    )

    print("\nBrightness")
    print(
        f"Dark threshold ≈ "
        f"{original['brightness'].mean() - original['brightness'].std():.2f}"
    )

    print("\nContrast")
    print(
        f"Low contrast threshold ≈ "
        f"{original['contrast'].mean() - original['contrast'].std():.2f}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    df = analyze_dataset()

    save_csv(df)

    print_statistics(df)

    recommend_thresholds(df)

    plot_histograms(df)


if __name__ == "__main__":

    main()