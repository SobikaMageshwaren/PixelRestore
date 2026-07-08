"""
===========================================================
VisionRestore AI
Phase 2 : Image Quality Analyzer
Part 1 : Image Quality Metrics
===========================================================
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(image_path):
    """
    Load image from disk and return RGB image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image : {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ==========================================================
# BLUR SCORE
# ==========================================================

def compute_blur(image):
    """
    Variance of Laplacian.

    Lower value -> More blurry.
    Higher value -> Sharper.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return round(score, 2)


# ==========================================================
# NOISE ESTIMATION
# ==========================================================

def compute_noise(image):
    """
    Estimate noise using high-frequency residual.

    Higher value -> More noisy.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = gray.astype(np.float32) - blur.astype(np.float32)

    score = np.std(residual)

    return round(float(score), 2)


# ==========================================================
# BRIGHTNESS
# ==========================================================

def compute_brightness(image):
    """
    Average brightness.

    Range : 0 - 255
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    score = np.mean(gray)

    return round(float(score), 2)


# ==========================================================
# CONTRAST
# ==========================================================

def compute_contrast(image):
    """
    Standard deviation of grayscale intensity.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    score = np.std(gray)

    return round(float(score), 2)


# ==========================================================
# ENTROPY
# ==========================================================

def compute_entropy(image):
    """
    Shannon Entropy.

    Higher value -> More information.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.ravel()

    probability = histogram / histogram.sum()

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability * np.log2(probability)
    )

    return round(float(entropy), 2)


# ==========================================================
# RESOLUTION
# ==========================================================

def compute_resolution(image):
    """
    Return image dimensions.
    """

    h, w = image.shape[:2]

    return f"{w} x {h}"


# ==========================================================
# ANALYZE IMAGE
# ==========================================================

def analyze_image(image):

    report = {

        "Blur Score":
            compute_blur(image),

        "Noise Score":
            compute_noise(image),

        "Brightness":
            compute_brightness(image),

        "Contrast":
            compute_contrast(image),

        "Entropy":
            compute_entropy(image),

        "Resolution":
            compute_resolution(image)

    }

    return report
# ==========================================================
# CLASSIFICATION FUNCTIONS
# ==========================================================

def classify_blur(score):
    """
    Lower Laplacian variance means more blur.
    Thresholds may be adjusted later.
    """

    if score < 1000:
        return "High Blur"
    elif score < 3500:
        return "Medium Blur"
    else:
        return "Low Blur"


# ----------------------------------------------------------


def classify_noise(score):
    """
    Higher residual standard deviation means more noise.
    """

    if score < 12:
        return "Low Noise"
    elif score < 20:
        return "Medium Noise"
    else:
        return "High Noise"


# ----------------------------------------------------------


def classify_brightness(score):
    """
    Brightness ranges from 0-255.
    """

    if score < 150:
        return "Low Brightness"

    elif score < 220:
        return "Medium Brightness"

    else:
        return "High Brightness"


# ----------------------------------------------------------


def classify_contrast(score):
    """
    Contrast based on grayscale standard deviation.
    """

    if score < 30:
        return "Low Contrast"

    elif score < 40:
        return "Medium Contrast"

    else:
        return "High Contrast"


# ----------------------------------------------------------


def classify_entropy(score):
    """
    Shannon entropy classification.
    """

    if score <2:
        return "Low Information"

    elif score < 4:
        return "Medium Information"

    else:
        return "High Information"


# ----------------------------------------------------------


def classify_resolution(resolution):
    """
    Resolution format:
    'width x height'
    """

    width, height = resolution.split(" x ")

    width = int(width)
    height = int(height)

    pixels = width * height

    if pixels < 500000:
        return "Low Resolution"

    elif pixels < 1000000:
        return "Medium Resolution"

    else:
        return "High Resolution"
# ==========================================================
# OVERALL QUALITY SCORE
# ==========================================================

def compute_quality_score(report):
    """
    Computes an overall quality score (0-100)
    based on all quality metrics.
    """

    score = 100

    # Blur
    if report["Blur Status"] == "High Blur":
        score -= 30
    elif report["Blur Status"] == "Medium Blur":
        score -= 15

    # Noise
    if report["Noise Status"] == "High Noise":
        score -= 20
    elif report["Noise Status"] == "Medium Noise":
        score -= 10

    # Brightness
    # Brightness

    if report["Brightness Status"] == "Low Brightness":
        score -= 20

    elif report["Brightness Status"] == "High Brightness":
        score -= 10

    # Contrast
    if report["Contrast Status"] == "Low Contrast":
        score -= 15
    elif report["Contrast Status"] == "Medium Contrast":
        score -= 5

    score = max(0, score)

    if score >= 90:
        label = "Excellent"

    elif score >= 75:
        label = "Good"

    elif score >= 60:
        label = "Fair"

    elif score >= 40:
        label = "Poor"

    else:
        label = "Very Poor"

    return score, label
# ==========================================================
# RESTORATION DIFFICULTY
# ==========================================================

def restoration_difficulty(score):

    if score >= 85:
        return "Easy"

    elif score >= 65:
        return "Moderate"

    else:
        return "Severe"


# ==========================================================
# ANALYZER CONFIDENCE
# ==========================================================

def compute_confidence(report):
    """
    Computes confidence based on how far each metric
    is from its decision boundary.
    """

    confidence = []

    # ---------------- Blur ----------------

    blur = report["Blur Score"]

    if blur >= 3500:
        confidence.append(min(100, 70 + ((blur - 3500) / 100)))

    elif blur <= 1000:
        confidence.append(min(100, 70 + ((1000 - blur) / 20)))

    else:
        confidence.append(75)

    # ---------------- Noise ----------------

    noise = report["Noise Score"]

    if noise >= 20:
        confidence.append(min(100, 70 + ((noise - 20) * 8)))

    elif noise <= 12:
        confidence.append(min(100, 70 + ((12 - noise) * 8)))

    else:
        confidence.append(75)

    # ---------------- Brightness ----------------

    brightness = report["Brightness"]

    if brightness >= 220:
        confidence.append(min(100, 70 + ((brightness - 220) / 2)))

    elif brightness <= 150:
        confidence.append(min(100, 70 + ((150 - brightness) / 2)))

    else:
        confidence.append(75)

    return round(sum(confidence) / len(confidence))
def classify_image(report):
    """
    Adds qualitative labels to all metrics.
    """

    report["Blur Status"] = classify_blur(
        report["Blur Score"]
    )

    report["Noise Status"] = classify_noise(
        report["Noise Score"]
    )

    report["Brightness Status"] = classify_brightness(
        report["Brightness"]
    )

    report["Contrast Status"] = classify_contrast(
        report["Contrast"]
    )

    report["Entropy Status"] = classify_entropy(
        report["Entropy"]
    )

    report["Resolution Status"] = classify_resolution(
        report["Resolution"]
    )
    quality_score, quality_label = compute_quality_score(report)
    report["Restoration Difficulty"] = restoration_difficulty(
    quality_score
)
    report["Overall Quality Score"] = quality_score
    report["Overall Quality"] = quality_label
    report["Confidence"] = compute_confidence(report)

    return report
# ==========================================================
# RESTORATION RECOMMENDATION ENGINE
# ==========================================================
# ==========================================================
# RESTORATION RECOMMENDATION ENGINE
# ==========================================================

def recommend_restoration(report):
    """
    Generates an executable restoration pipeline based on
    the detected image degradations.
    """

    detected_problems = []

    pipeline = []

    step = 1

    # --------------------------------------------------
    # Blur
    # --------------------------------------------------

    if report["Blur Status"] != "Low Blur":

        detected_problems.append(report["Blur Status"])

        pipeline.append({
            "step": step,
            "operation": "Deblur",
            "model": "SwinIR"
        })

        step += 1

    # --------------------------------------------------
    # Gaussian Noise
    # --------------------------------------------------

    if report["Noise Status"] != "Low Noise":

        detected_problems.append(report["Noise Status"])

        pipeline.append({
            "step": step,
            "operation": "Denoise",
            "model": "DnCNN"
        })

        step += 1

    # --------------------------------------------------
    # Low Brightness
    # --------------------------------------------------

    if report["Brightness Status"] == "Low Brightness":

        detected_problems.append("Low Brightness")

        pipeline.append({
            "step": step,
            "operation": "Low-Light Enhancement",
            "model": "Gamma + CLAHE"
        })

        step += 1

    # --------------------------------------------------
    # Low Contrast
    # --------------------------------------------------

    if report["Contrast Status"] == "Low Contrast":

        detected_problems.append("Low Contrast")

        pipeline.append({
            "step": step,
            "operation": "Contrast Enhancement",
            "model": "CLAHE"
        })

        step += 1

    # --------------------------------------------------
    # Super Resolution
    # --------------------------------------------------

    if report["Resolution Status"] == "Low Resolution":

        detected_problems.append("Low Resolution")

        pipeline.append({
            "step": step,
            "operation": "Super Resolution",
            "model": "SwinIR"
        })

        step += 1

    # --------------------------------------------------
    # No Restoration Required
    # --------------------------------------------------

    if len(pipeline) == 0:

        pipeline.append({

            "step": 1,

            "operation": "No Restoration Needed",

            "model": "None"

        })

    return {

        "detected_problems": detected_problems,

        "recommended_pipeline": pipeline

    }
# ==========================================================
# IMAGE DIAGNOSIS
# ==========================================================
def generate_diagnosis(report):

    issues = report["Recommendation"]["detected_problems"]

    if len(issues) == 0:

        return (
            "The image quality is excellent. "
            "No restoration is required."
        )

    return (
        f"Primary issue: {issues[0]}. "
        f"Total detected issues: {len(issues)}."
    )
# ==========================================================
# DISPLAY REPORT
# ==========================================================

def display_report(report):

    print("\n")

    print("=" * 60)
    print("IMAGE QUALITY ANALYSIS REPORT")
    print("=" * 60)

    print(f"{'Metric':25s}{'Value'}")
    print("-" * 60)

    print(f"{'Blur Score':25s}{report['Blur Score']}")
    print(f"{'Blur Status':25s}{report['Blur Status']}")

    print("-" * 60)

    print(f"{'Noise Score':25s}{report['Noise Score']}")
    print(f"{'Noise Status':25s}{report['Noise Status']}")

    print("-" * 60)

    print(f"{'Brightness':25s}{report['Brightness']}")
    print(f"{'Brightness Status':25s}{report['Brightness Status']}")

    print("-" * 60)

    print(f"{'Contrast':25s}{report['Contrast']}")
    print(f"{'Contrast Status':25s}{report['Contrast Status']}")

    print("-" * 60)

    print(f"{'Entropy':25s}{report['Entropy']}")
    print(f"{'Entropy Status':25s}{report['Entropy Status']}")

    print("-" * 60)

    print(f"{'Resolution':25s}{report['Resolution']}")
    print(f"{'Resolution Status':25s}{report['Resolution Status']}")

    print("=" * 60)
    print("-" * 60)

    print(f"{'Overall Quality':25s}{report['Overall Quality']}")

    print(f"{'Quality Score':25s}{report['Overall Quality Score']}/100")
    print(f"{'Analyzer Confidence':25s}{report['Confidence']}%")
    print(f"{'Difficulty':25s}{report['Restoration Difficulty']}")
    print("\nImage Diagnosis")
    print("-" * 60)

    print(report["Diagnosis"])

    print("\nRecommended Pipeline")
    
    print("-" * 60)

    for i, step in enumerate(
        report["Recommendation"]["recommended_pipeline"],
        start=1
    ):

        print(
        f"{step['step']}. {step['operation']} ({step['model']})"
    )
        
    print("\nDetected Problems")
    print("-"*60)

    for item in report["Recommendation"]["detected_problems"]:

        print("•", item)

# ==========================================================
# VISUALIZATION
# ==========================================================

def visualize(image, title="Image"):

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    IMAGE_PATH = Path(
        r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/data/processed/gaussian_noise/00040534.png"
    )

    image = load_image(IMAGE_PATH)

    visualize(image)

    report = analyze_image(image)

    report = classify_image(report)

    report["Recommendation"] = recommend_restoration(report)

    report["Diagnosis"] = generate_diagnosis(report)

    display_report(report)


if __name__ == "__main__":

    main()