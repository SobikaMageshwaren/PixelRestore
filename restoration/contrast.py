"""
============================================================
VisionRestore AI
Phase 3 : Contrast Enhancement
CLAHE Enhancement
============================================================

Uses Contrast Limited Adaptive Histogram Equalization (CLAHE)
to improve local contrast while preserving document details.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# CLAHE PARAMETERS
# ==========================================================

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


# ==========================================================
# CONTRAST ENHANCEMENT
# ==========================================================

def enhance_contrast(image):
    """
    Enhance image contrast using CLAHE.

    Parameters
    ----------
    image : numpy.ndarray (RGB)

    Returns
    -------
    numpy.ndarray (RGB)
    """

    # Convert RGB → LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    # Enhance only luminance channel
    enhanced_l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced_rgb


# ==========================================================
# IMAGE COMPARISON
# ==========================================================

def visualize(original, enhanced):
    """
    Display original and enhanced images.
    """

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced)
    plt.title("CLAHE Enhanced")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN (Testing)
# ==========================================================

def main():

    IMAGE_PATH = (
        r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/"
        r"data/processed/low_light/00040534.png"
    )

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {IMAGE_PATH}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    enhanced = enhance_contrast(image)

    visualize(image, enhanced)


if __name__ == "__main__":

    main()