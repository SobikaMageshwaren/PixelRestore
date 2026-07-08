"""
============================================================
VisionRestore AI
DnCNN Restoration Module
============================================================
"""

from pathlib import Path
import time

import cv2
import numpy as np
import torch
import deepinv as dinv
import sys


# ==========================================================
# SIGMA ESTIMATOR
# ==========================================================

def estimate_sigma(noise_score):
    """
    Convert the quality analyzer's Noise Score
    into an estimated Gaussian sigma for DnCNN.

    Mapping is based on the statistics observed
    during threshold calibration.
    """

    if noise_score < 8:
        sigma = 5

    elif noise_score < 12:
        sigma = 10

    elif noise_score < 16:
        sigma = 15

    elif noise_score < 20:
        sigma = 25

    elif noise_score < 24:
        sigma = 35

    else:
        sigma = 50

    return sigma
class DnCNNRestorer:

    """
    Wrapper around pretrained DnCNN.

    Performs:
        • preprocessing
        • inference
        • postprocessing
        • inference timing
    """

    def __init__(self):

        print("\nLoading DnCNN...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = dinv.models.DnCNN(

            in_channels=3,

            out_channels=3,

            depth=20,

            nf=64,

            pretrained="download",

            device=self.device

        )

        self.model.eval()

        print(f"✓ Model Loaded ({self.device.upper()})")

    # ---------------------------------------------------

    def preprocess(self, image):

        image = image.astype(np.float32)

        image /= 255.0

        tensor = torch.from_numpy(image)

        tensor = tensor.permute(2,0,1)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # ---------------------------------------------------

    def postprocess(self, tensor):

        image = tensor.squeeze(0)

        image = image.permute(1,2,0)

        image = image.cpu().numpy()

        image = np.clip(image,0,1)

        image *= 255

        return image.astype(np.uint8)

    # ---------------------------------------------------

    @torch.no_grad()
    def restore(self, image, noise_score):

        sigma = estimate_sigma(noise_score)

        tensor = self.preprocess(image)

        sigma_tensor = torch.tensor(
            [sigma / 255.0],
            device=self.device
        )

        start = time.perf_counter()

        restored = self.model(
            tensor,
            sigma_tensor
        )

        inference_time = time.perf_counter() - start

        restored = self.postprocess(restored)

        return restored, inference_time, sigma
# ==========================================================
# Utility
# ==========================================================

def load_image(path):

    image = cv2.imread(str(path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def save_image(image,path):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(path),
        image
    )


# ==========================================================
# Demo
# ==========================================================

def main():

    image_path = Path(
        r"data/processed/gaussian_noise/00040534.png"
    )

    output_dir = Path(
        "outputs/restored"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = load_image(image_path)

    restorer = DnCNNRestorer()


    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))
    from quality_analyzer import (
        analyze_image,
        classify_image
    )
    report = analyze_image(image)

    report = classify_image(report)

    noise_score = report["Noise Score"]

    restored, time_taken, sigma = restorer.restore(
        image,
        noise_score
    )

    output_path = output_dir / image_path.name

    save_image(
        restored,
        output_path
    )

    print("\n==============================")

    print("\n==============================================")
    print("DnCNN Restoration")
    print("==============================================")

    print(f"Image Name        : {image_path.name}")
    print(f"Detected Noise    : {noise_score:.2f}")
    print(f"Estimated Sigma   : {sigma}")
    print(f"Inference Time    : {time_taken:.3f} sec")
    print(f"Saved Output      : {output_path}")


if __name__=="__main__":

    main()