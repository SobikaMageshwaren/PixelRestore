"""
============================================================
VisionRestore AI
Main Pipeline
============================================================

Workflow

1. Load Image
2. Analyze Image Quality
3. Generate Restoration Recommendation
4. Restore Image
5. Evaluate Restoration
6. Display Final Results
============================================================
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from quality_analyzer import (
    load_image,
    analyze_image,
    classify_image,
    recommend_restoration,
    generate_diagnosis,
    display_report
)

from restoration_engine import restore_image

from evaluation import evaluate_restoration


# ==========================================================
# VISUALIZATION
# ==========================================================

def visualize_results(original, restored):
    """
    Display original and restored images.
    """

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(restored)
    plt.title("Restored Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    IMAGE_PATH = Path(
        r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/data/processed/combined/00040534.png"
    )

    print("\n")
    print("="*65)
    print("VISIONRESTORE AI")
    print("="*65)

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    original = load_image(IMAGE_PATH)

    # ------------------------------------------------------
    # Quality Analysis
    # ------------------------------------------------------

    report = analyze_image(original)

    report = classify_image(report)

    report["Recommendation"] = recommend_restoration(report)

    report["Diagnosis"] = generate_diagnosis(report)

    display_report(report)

    # ------------------------------------------------------
    # Restoration
    # ------------------------------------------------------

    print("\n")
    print("="*65)
    print("RESTORATION ENGINE")
    print("="*65)

    restored = restore_image(
        IMAGE_PATH,
        report["Recommendation"],
        report
    )

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    print("\n")
    print("="*65)
    print("RESTORATION EVALUATION")
    print("="*65)

    evaluate_restoration(
        original,
        restored
    )

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------

    visualize_results(
        original,
        restored
    )

    print("\n")
    print("="*65)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*65)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()