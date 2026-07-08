"""
============================================================
VisionRestore AI
Phase 3 : Salt & Pepper Noise Removal
Median Filter
============================================================

Uses OpenCV Median Blur to remove impulse noise
while preserving document edges.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# PARAMETERS
# ==========================================================

MEDIAN_KERNEL = 3


# ==========================================================
# MEDIAN FILTER
# ==========================================================

def median_denoise(image):
    """
    Remove Salt & Pepper noise using Median Filter.

    Parameters
    ----------
    image : RGB image

    Returns
    -------
    RGB image
    """

    denoised = cv2.medianBlur(
        image,
        MEDIAN_KERNEL
    )

    return denoised


# ==========================================================
# VISUALIZATION
# ==========================================================

def visualize(original, restored):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(restored)
    plt.title("Median Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    IMAGE_PATH = (
        r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/"
        r"data/processed/salt_pepper/00040534.png"
    )

    image = cv2.imread(IMAGE_PATH)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    restored = median_denoise(image)

    visualize(image, restored)


if __name__ == "__main__":
    main()