"""
=========================================================
VisionRestore AI
Phase 1 : Dataset Preparation
Part 1 : Dataset Loading & Visualization
=========================================================

Author : Sobika Mageshwaren
Project : VisionRestore AI

"""

import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile


# =========================================================
# CONFIGURATION
# =========================================================

# Change this path to your FUNSD training images folder
RAW_DATASET_PATH = Path(
    "D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/dataset/training_data/images"
)

# Output folders
OUTPUT_ROOT = Path("data/processed")

OUTPUT_FOLDERS = {
    "original": OUTPUT_ROOT / "original",
    "gaussian_blur": OUTPUT_ROOT / "gaussian_blur",
    "motion_blur": OUTPUT_ROOT / "motion_blur",
    "gaussian_noise": OUTPUT_ROOT / "gaussian_noise",
    "salt_pepper": OUTPUT_ROOT / "salt_pepper",
    "low_light": OUTPUT_ROOT / "low_light",
    "jpeg": OUTPUT_ROOT / "jpeg",
    "combined": OUTPUT_ROOT / "combined",
}

REPORTS_FOLDER = Path("reports")


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():
    """
    Creates all required folders for the project.
    """

    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

    for folder in OUTPUT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    print("\n[OK] Output directories created.\n")


# =========================================================
# LOAD DATASET
# =========================================================

def load_images(dataset_path):
    """
    Loads all images from the FUNSD dataset.

    Returns
    -------
    image_paths : list
        List containing all image paths.
    """

    valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

    image_paths = []

    for file in sorted(dataset_path.iterdir()):

        if file.suffix.lower() in valid_extensions:
            image_paths.append(file)

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Dataset Location : {dataset_path}")
    print(f"Total Images     : {len(image_paths)}")
    print("=" * 60)

    return image_paths


# =========================================================
# VISUALIZE SAMPLE IMAGES
# =========================================================

def visualize_dataset(image_paths, num_samples=5):
    """
    Displays and saves random sample images.
    """

    if len(image_paths) == 0:
        print("No images found.")
        return

    samples = random.sample(image_paths, min(num_samples, len(image_paths)))

    fig, axes = plt.subplots(1, len(samples), figsize=(18, 5))

    if len(samples) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, samples):

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")

    plt.suptitle("FUNSD Dataset Samples", fontsize=16)

    plt.tight_layout()

    save_path = REPORTS_FOLDER / "sample_dataset.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"\n[OK] Sample visualization saved to:\n{save_path}\n")


# =========================================================
# DATASET SUMMARY
# =========================================================

def print_summary(image_paths):

    print("\n")
    print("=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)

    print(f"Images Loaded : {len(image_paths)}")

    print("\nOutput Structure")

    for key, value in OUTPUT_FOLDERS.items():
        print(f"{key:20s} --> {value}")

    print("=" * 60)
    print()

# =========================================================
# DEGRADATION FUNCTIONS
# =========================================================

import numpy as np


def gaussian_blur(image, severity="medium"):
    """
    Apply Gaussian Blur
    """

    kernel_sizes = {
        "low": (3, 3),
        "medium": (7, 7),
        "high": (11, 11)
    }

    kernel = kernel_sizes.get(severity, (7, 7))

    blurred = cv2.GaussianBlur(image, kernel, 0)

    return blurred


# ---------------------------------------------------------


def motion_blur(image, severity="medium"):
    """
    Apply Motion Blur
    """

    kernel_lengths = {
        "low": 7,
        "medium": 15,
        "high": 25
    }

    size = kernel_lengths.get(severity, 15)

    kernel = np.zeros((size, size))

    kernel[size // 2, :] = np.ones(size)

    kernel = kernel / size

    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


# ---------------------------------------------------------


def gaussian_noise(image, severity="medium"):
    """
    Add Gaussian Noise
    """

    sigma_levels = {
        "low": 10,
        "medium": 25,
        "high": 45
    }

    sigma = sigma_levels.get(severity, 25)

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


# ---------------------------------------------------------


def salt_pepper_noise(image,
                      severity="medium"):
    """
    Apply Salt & Pepper Noise
    """

    amount_levels = {
        "low": 0.005,
        "medium": 0.015,
        "high": 0.03
    }

    amount = amount_levels.get(severity, 0.015)

    output = image.copy()

    total_pixels = image.shape[0] * image.shape[1]

    num_salt = int(total_pixels * amount / 2)

    num_pepper = int(total_pixels * amount / 2)

    # Salt

    coords = (
        np.random.randint(
            0,
            image.shape[0],
            num_salt
        ),
        np.random.randint(
            0,
            image.shape[1],
            num_salt
        )
    )

    output[coords] = 255

    # Pepper

    coords = (
        np.random.randint(
            0,
            image.shape[0],
            num_pepper
        ),
        np.random.randint(
            0,
            image.shape[1],
            num_pepper
        )
    )

    output[coords] = 0

    return output


# ---------------------------------------------------------


def low_light(image,
              severity="medium"):
    """
    Simulate Low Light
    """

    brightness_levels = {
        "low": 0.75,
        "medium": 0.50,
        "high": 0.30
    }

    factor = brightness_levels.get(
        severity,
        0.50
    )

    dark = image.astype(np.float32)

    dark = dark * factor

    dark = np.clip(
        dark,
        0,
        255
    )

    return dark.astype(np.uint8)


# =========================================================
# PREVIEW ALL DEGRADATIONS
# =========================================================

def preview_degradations(image_paths):
    """
    Display one sample image with all
    degradation techniques.
    """

    if len(image_paths) == 0:
        return

    sample = random.choice(image_paths)

    image = cv2.imread(str(sample))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    images = [

    ("Original", image),

    ("Gaussian Blur",
     gaussian_blur(image)),

    ("Motion Blur",
     motion_blur(image)),

    ("Gaussian Noise",
     gaussian_noise(image)),

    ("Salt & Pepper",
     salt_pepper_noise(image)),

    ("Low Light",
     low_light(image)),

    ("JPEG",
     jpeg_compression(image)),

    ("Combined",
     combined_degradation(image))
]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8)
    )

    axes = axes.flatten()

    for ax, (title, img) in zip(axes, images):

        ax.imshow(img)

        ax.set_title(title)

        ax.axis("off")

    plt.tight_layout()

    save_path = REPORTS_FOLDER / \
        "degradation_preview_part2A.png"

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.show()

    print(
        f"\n[OK] Preview saved at:\n{save_path}\n"
    )
# =========================================================
# JPEG COMPRESSION
# =========================================================

def jpeg_compression(image, severity="medium"):
    """
    Apply JPEG compression artifacts.
    """

    quality_levels = {
        "low": 85,
        "medium": 50,
        "high": 20
    }

    quality = quality_levels.get(severity, 50)

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]

    success, encoded = cv2.imencode(".jpg", image, encode_param)

    if not success:
        return image.copy()

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return decoded


# =========================================================
# COMBINED DEGRADATION
# =========================================================

def combined_degradation(image):
    """
    Randomly applies 2-4 degradations.
    """

    operations = [
        gaussian_blur,
        motion_blur,
        gaussian_noise,
        salt_pepper_noise,
        low_light,
        jpeg_compression
    ]

    output = image.copy()

    n = random.randint(2, 4)

    selected = random.sample(operations, n)

    for operation in selected:
        output = operation(output)

    return output


# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(image, save_path):
    """
    Save RGB image to disk.
    """

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(save_path), bgr)


# =========================================================
# PROCESS COMPLETE DATASET
# =========================================================

def process_dataset(image_paths):
    """
    Generate all degraded images.
    """

    print("\nGenerating degraded dataset...\n")

    for img_path in tqdm(image_paths):

        image = cv2.imread(str(img_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        filename = img_path.name

        # Original

        save_image(
            image,
            OUTPUT_FOLDERS["original"] / filename
        )

        # Gaussian Blur

        save_image(
            gaussian_blur(image),
            OUTPUT_FOLDERS["gaussian_blur"] / filename
        )

        # Motion Blur

        save_image(
            motion_blur(image),
            OUTPUT_FOLDERS["motion_blur"] / filename
        )

        # Gaussian Noise

        save_image(
            gaussian_noise(image),
            OUTPUT_FOLDERS["gaussian_noise"] / filename
        )

        # Salt & Pepper

        save_image(
            salt_pepper_noise(image),
            OUTPUT_FOLDERS["salt_pepper"] / filename
        )

        # Low Light

        save_image(
            low_light(image),
            OUTPUT_FOLDERS["low_light"] / filename
        )

        # JPEG

        save_image(
            jpeg_compression(image),
            OUTPUT_FOLDERS["jpeg"] / filename
        )

        # Combined

        save_image(
            combined_degradation(image),
            OUTPUT_FOLDERS["combined"] / filename
        )

    print("\n[OK] Dataset generation completed.\n")


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 60)
    print("VisionRestore AI")
    print("Phase 1 : Dataset Preparation")
    print("=" * 60)

    create_directories()

    image_paths = load_images(
        RAW_DATASET_PATH
    )

    visualize_dataset(image_paths)

    preview_degradations(image_paths)

    process_dataset(image_paths)

    print_summary(image_paths)

    print("\n[OK] Phase 1 Completed Successfully.\n")

if __name__ == "__main__":
    main()