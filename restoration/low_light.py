"""
============================================================
VisionRestore AI
Phase 3 : Low-Light Enhancement
Gamma Correction + CLAHE
============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# PARAMETERS
# ==========================================================

GAMMA = 1.8

CLAHE_CLIP = 2.0

CLAHE_GRID = (8,8)


# ==========================================================
# GAMMA CORRECTION
# ==========================================================

def gamma_correction(image):

    inv_gamma = 1.0 / GAMMA

    table = np.array([

        ((i / 255.0) ** inv_gamma) * 255

        for i in np.arange(256)

    ]).astype("uint8")

    return cv2.LUT(image, table)


# ==========================================================
# CLAHE
# ==========================================================

def clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(

        clipLimit=CLAHE_CLIP,

        tileGridSize=CLAHE_GRID

    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    enhanced = cv2.cvtColor(

        lab,

        cv2.COLOR_LAB2RGB

    )

    return enhanced


# ==========================================================
# COMPLETE LOW LIGHT ENHANCEMENT
# ==========================================================

def enhance_low_light(image):

    gamma = gamma_correction(image)

    enhanced = clahe(gamma)

    return enhanced


# ==========================================================
# VISUALIZATION
# ==========================================================

def visualize(original, enhanced):

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.imshow(original)
    plt.title("Low Light")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(enhanced)
    plt.title("Enhanced")
    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    IMAGE_PATH = (

        r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/"

        r"data/processed/low_light/00040534.png"

    )

    image = cv2.imread(IMAGE_PATH)

    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    enhanced = enhance_low_light(image)

    visualize(image, enhanced)


if __name__ == "__main__":

    main()