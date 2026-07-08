"""
===========================================================
VisionRestore AI
Phase 4 : Restoration Evaluation

Part 1
-----------------------------------------------------------
• Imports
• Configuration
• Image Loader
• PSNR
• SSIM
• MSE
• OCR Extraction
• OCR Confidence
• CER
• WER
===========================================================
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import easyocr

from jiwer import cer, wer

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)

# ==========================================================
# CONFIGURATION
# ==========================================================

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = REPORT_DIR / "evaluation_results.csv"

# ----------------------------------------------------------

OCR_READER = easyocr.Reader(
    ['en'],
    gpu=False
)

# ==========================================================
# IMAGE LOADER
# ==========================================================

def load_image(image_path):
    """
    Load image from disk.

    Returns
    -------
    RGB image
    """

    image = cv2.imread(str(image_path))

    if image is None:

        raise FileNotFoundError(
            f"Unable to load {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ==========================================================
# MSE
# ==========================================================

def compute_mse(
        original,
        restored
):
    """
    Mean Squared Error

    Lower is better.
    """

    score = mean_squared_error(
        original,
        restored
    )

    return round(float(score), 4)


# ==========================================================
# PSNR
# ==========================================================

def compute_psnr(
        original,
        restored
):
    """
    Peak Signal-to-Noise Ratio

    Higher is better.
    """

    score = peak_signal_noise_ratio(

        original,

        restored,

        data_range=255

    )

    return round(float(score), 2)


# ==========================================================
# SSIM
# ==========================================================

def compute_ssim(
        original,
        restored
):
    """
    Structural Similarity Index.

    Higher is better.
    """

    score = structural_similarity(

        original,

        restored,

        channel_axis=2,

        data_range=255

    )

    return round(float(score), 4)


# ==========================================================
# OCR
# ==========================================================

def extract_text(image):
    """
    Perform OCR using EasyOCR.

    Returns
    -------
    extracted_text
    average_confidence
    """

    results = OCR_READER.readtext(
        image
    )

    text = []

    confidence = []

    for item in results:

        text.append(item[1])

        confidence.append(item[2])

    extracted = " ".join(text)

    if len(confidence) == 0:

        avg_conf = 0.0

    else:

        avg_conf = np.mean(confidence)

    return extracted, round(float(avg_conf), 3)


# ==========================================================
# OCR ACCURACY
# ==========================================================

def compute_ocr_accuracy(
        reference_text,
        predicted_text
):
    """
    OCR Accuracy based on CER.

    Accuracy = (1-CER)*100
    """

    if len(reference_text.strip()) == 0:

        return 0

    error = cer(
        reference_text,
        predicted_text
    )

    accuracy = max(

        0,

        (1 - error) * 100

    )

    return round(float(accuracy), 2)


# ==========================================================
# CHARACTER ERROR RATE
# ==========================================================

def compute_cer(
        reference_text,
        predicted_text
):
    """
    Character Error Rate.
    Lower is better.
    """

    if len(reference_text.strip()) == 0:

        return 1.0

    score = cer(
        reference_text,
        predicted_text
    )

    return round(float(score), 4)


# ==========================================================
# WORD ERROR RATE
# ==========================================================

def compute_wer(
        reference_text,
        predicted_text
):
    """
    Word Error Rate.

    Lower is better.
    """

    if len(reference_text.strip()) == 0:

        return 1.0

    score = wer(

        reference_text,

        predicted_text

    )

    return round(float(score), 4)


# ==========================================================
# TIMER
# ==========================================================

class Timer:
    """
    Utility timer for inference benchmarking.
    """

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self):

        elapsed = time.perf_counter() - self.start_time

        return round(elapsed, 3)
# ==========================================================
# IMPORT QUALITY ANALYZER
# ==========================================================

from quality_analyzer import (

    analyze_image,

    classify_image

)


# ==========================================================
# QUALITY METRICS
# ==========================================================

def evaluate_quality_metrics(
        original,
        degraded,
        restored
):
    """
    Compare quality metrics before and after restoration.

    Returns
    -------
    Dictionary containing all quality improvements.
    """

    original_report = classify_image(
        analyze_image(original)
    )

    degraded_report = classify_image(
        analyze_image(degraded)
    )

    restored_report = classify_image(
        analyze_image(restored)
    )

    metrics = {

        # ----------------------------------
        # Blur
        # ----------------------------------

        "Blur Before":
            degraded_report["Blur Score"],

        "Blur After":
            restored_report["Blur Score"],

        "Blur Improvement":
            round(
                restored_report["Blur Score"]
                -
                degraded_report["Blur Score"],
                2
            ),

        # ----------------------------------
        # Noise
        # ----------------------------------

        "Noise Before":
            degraded_report["Noise Score"],

        "Noise After":
            restored_report["Noise Score"],

        "Noise Reduction":
            round(
                degraded_report["Noise Score"]
                -
                restored_report["Noise Score"],
                2
            ),

        # ----------------------------------
        # Brightness
        # ----------------------------------

        "Brightness Before":
            degraded_report["Brightness"],

        "Brightness After":
            restored_report["Brightness"],

        # ----------------------------------
        # Contrast
        # ----------------------------------

        "Contrast Before":
            degraded_report["Contrast"],

        "Contrast After":
            restored_report["Contrast"],

        # ----------------------------------
        # Entropy
        # ----------------------------------

        "Entropy Before":
            degraded_report["Entropy"],

        "Entropy After":
            restored_report["Entropy"],

        # ----------------------------------
        # Overall Quality
        # ----------------------------------

        "Quality Before":
            degraded_report["Overall Quality"],

        "Quality After":
            restored_report["Overall Quality"],

        "Quality Score Before":
            degraded_report["Overall Quality Score"],

        "Quality Score After":
            restored_report["Overall Quality Score"]

    }

    return metrics


# ==========================================================
# COMPLETE IMAGE EVALUATION
# ==========================================================

def evaluate_image(

        original,

        degraded,

        restored,

        inference_time=0.0

):
    """
    Performs a complete evaluation.

    Returns
    -------
    Dictionary containing all evaluation metrics.
    """

    # --------------------------------------
    # Full Reference Metrics
    # --------------------------------------
    # ==========================================================
    # Resize restored image for fair comparison
    # ==========================================================

    if original.shape != restored.shape:

        restored = cv2.resize(

            restored,

            (
                original.shape[1],   # width
                original.shape[0]    # height
            ),

            interpolation=cv2.INTER_CUBIC

    )
    mse = compute_mse(
        original,
        restored
    )

    psnr = compute_psnr(
        original,
        restored
    )

    ssim = compute_ssim(
        original,
        restored
    )

    # --------------------------------------
    # OCR
    # --------------------------------------

    gt_text, _ = extract_text(
        original
    )

    restored_text, confidence = extract_text(
        restored
    )

    ocr_accuracy = compute_ocr_accuracy(
        gt_text,
        restored_text
    )

    cer_score = compute_cer(
        gt_text,
        restored_text
    )

    wer_score = compute_wer(
        gt_text,
        restored_text
    )

    # --------------------------------------
    # Quality Metrics
    # --------------------------------------

    quality = evaluate_quality_metrics(

        original,

        degraded,

        restored

    )

    # --------------------------------------
    # Final Dictionary
    # --------------------------------------

    report = {

        "MSE":
            mse,

        "PSNR":
            psnr,

        "SSIM":
            ssim,

        "OCR Accuracy":
            ocr_accuracy,

        "OCR Confidence":
            confidence,

        "CER":
            cer_score,

        "WER":
            wer_score,

        "Inference Time":
            round(
                inference_time,
                3
            )

    }

    report.update(
        quality
    )

    return report

# ==========================================================
# VISUALIZATION
# ==========================================================

def visualize_results(
        original,
        degraded,
        restored,
        report,
        image_name="Image"
):
    """
    Display Original, Degraded and Restored images.
    """

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(degraded)
    plt.title("Degraded")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(restored)
    plt.title("Restored")
    plt.axis("off")

    plt.suptitle(
        f"{image_name}\n"
        f"PSNR={report['PSNR']} dB | "
        f"SSIM={report['SSIM']} | "
        f"OCR={report['OCR Accuracy']}%"
    )

    plt.tight_layout()

    plt.show()


# ==========================================================
# SAVE CSV
# ==========================================================

def save_csv(results):

    df = pd.DataFrame(results)

    df.to_csv(
        CSV_PATH,
        index=False
    )

    print(f"\n✓ CSV Saved : {CSV_PATH}")


# ==========================================================
# SUMMARY REPORT
# ==========================================================

def print_summary(results):

    df = pd.DataFrame(results)

    print("\n")

    print("="*70)
    print("VISIONRESTORE AI - FINAL EVALUATION")
    print("="*70)

    print(f"Images Evaluated      : {len(df)}")

    print("-"*70)

    print(f"Average PSNR          : {df['PSNR'].mean():.2f} dB")

    print(f"Average SSIM          : {df['SSIM'].mean():.4f}")

    print(f"Average MSE           : {df['MSE'].mean():.2f}")

    print("-"*70)

    print(f"Average OCR Accuracy  : {df['OCR Accuracy'].mean():.2f}%")

    print(f"Average OCR Confidence: {df['OCR Confidence'].mean():.2f}")

    print(f"Average CER           : {df['CER'].mean():.4f}")

    print(f"Average WER           : {df['WER'].mean():.4f}")

    print("-"*70)

    print(f"Blur Improvement      : {df['Blur Improvement'].mean():.2f}")

    print(f"Noise Reduction       : {df['Noise Reduction'].mean():.2f}")

    print("-"*70)

    print(f"Average Runtime       : {df['Inference Time'].mean():.3f} sec")

    print("="*70)


# ==========================================================
# BATCH EVALUATION
# ==========================================================

def evaluate_dataset(

        original_dir,

        degraded_dir,

        restored_dir,

        visualize=False

):
    """
    Evaluate all restored images.
    """

    results = []

    original_images = sorted(
        Path(original_dir).glob("*")
    )

    for original_path in original_images:

        name = original_path.name

        degraded_path = Path(degraded_dir) / name

        restored_path = Path(restored_dir) / name

        if not degraded_path.exists():

            continue

        if not restored_path.exists():

            continue

        original = load_image(original_path)

        degraded = load_image(degraded_path)

        restored = load_image(restored_path)

        report = evaluate_image(

            original,

            degraded,

            restored

        )

        report["Image"] = name

        results.append(report)

        print(f"✓ {name}")

        if visualize:

            visualize_results(

                original,

                degraded,

                restored,

                report,

                name

            )

    save_csv(results)

    print_summary(results)

    return results


# ==========================================================
# MAIN
# ==========================================================

def main():

    ORIGINAL_DIR = Path(
        r"data/processed/original"
    )

    DEGRADED_DIR = Path(
        r"data/processed/gaussian_noise"
    )

    RESTORED_DIR = Path(
        r"outputs/restored"
    )

    evaluate_dataset(

        ORIGINAL_DIR,

        DEGRADED_DIR,

        RESTORED_DIR,

        visualize=False

    )
# ==========================================================
# SINGLE IMAGE EVALUATION
# ==========================================================

def evaluate_restoration(
    original,
    restored,
    degraded=None,
    inference_time=0.0
):
    """
    Evaluate a single restored image.

    If degraded image is not provided,
    original image is used as reference for
    quality comparison.
    """

    if degraded is None:
        degraded = original

    report = evaluate_image(
        original=original,
        degraded=degraded,
        restored=restored,
        inference_time=inference_time
    )

    print("\n")
    print("=" * 70)
    print("RESTORATION EVALUATION REPORT")
    print("=" * 70)

    for key, value in report.items():
        print(f"{key:25s}: {value}")

    print("=" * 70)

    return report

if __name__ == "__main__":

    main()