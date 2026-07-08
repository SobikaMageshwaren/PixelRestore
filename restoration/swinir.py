"""
===========================================================
VisionRestore AI

SwinIR Restorer

Part 1
-----------------------------------------------------------
• Imports
• Repository Detection
• SwinIR Model Initialization
• Load Pretrained Checkpoint
===========================================================
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

# ==========================================================
# Locate SwinIR Repository
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWINIR_ROOT = PROJECT_ROOT / "models" / "SwinIR"

if str(SWINIR_ROOT) not in sys.path:
    sys.path.insert(0, str(SWINIR_ROOT))

# ==========================================================
# Import SwinIR
# ==========================================================

try:

    from models.network_swinir import SwinIR

except Exception as e:

    raise ImportError(

        f"""
Unable to import SwinIR.

Expected Repository:

{SWINIR_ROOT}

Original Error:

{e}
"""

    )

# ==========================================================
# SwinIR Restorer
# ==========================================================

class SwinIRRestorer:

    def __init__(

        self,

        checkpoint=None,

        scale=4,

        device=None

    ):

        self.device = (

            device

            if device

            else

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.scale = scale

        # --------------------------------------------------

        if checkpoint is None:

            checkpoint = (

                PROJECT_ROOT
                / "models"
                / "SwinIR"
                / "model_zoo"
                / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"

            )

        self.checkpoint = Path(checkpoint)

        if not self.checkpoint.exists():

            raise FileNotFoundError(

                f"\nCheckpoint not found:\n{self.checkpoint}"

            )

        print("\nLoading SwinIR...")

        # --------------------------------------------------
        # Build Network
        # --------------------------------------------------

        self.model = SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,          # ✅ FIXED
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv"
    )

        # --------------------------------------------------
        # Load Weights
        # --------------------------------------------------

        checkpoint = torch.load(

            self.checkpoint,

            map_location=self.device

        )

        if "params_ema" in checkpoint:

            state_dict = checkpoint["params_ema"]

        elif "params" in checkpoint:

            state_dict = checkpoint["params"]

        else:

            state_dict = checkpoint

        self.model.load_state_dict(

            state_dict,

            strict=True

        )

        self.model.to(self.device)

        self.model.eval()

        print("✓ SwinIR Loaded")

        print("Device :", self.device)

        print("Scale  :", self.scale)
    # ======================================================
    # PREPROCESS IMAGE
    # ======================================================

    def preprocess(self, image):
        """
        Convert RGB uint8 image to normalized PyTorch tensor.
        """

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        tensor = torch.from_numpy(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # ======================================================
    # PAD IMAGE
    # ======================================================

    def pad_image(self, tensor):
        """
        SwinIR expects height and width to be multiples
        of the window size (8).
        """

        _, _, h, w = tensor.shape

        window = 8

        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window

        if pad_h > 0 or pad_w > 0:

            tensor = torch.nn.functional.pad(

                tensor,

                (0, pad_w, 0, pad_h),

                mode="reflect"

            )

        return tensor, h, w

    # ======================================================
    # REMOVE PADDING
    # ======================================================

    def remove_padding(
        self,
        tensor,
        original_h,
        original_w
    ):

        return tensor

    # ======================================================
    # POSTPROCESS
    # ======================================================

    def postprocess(self, tensor):
        """
        Convert output tensor to RGB uint8 image.
        """

        image = tensor.squeeze(0)

        image = image.detach().cpu().float()

        image = image.clamp(0, 1)

        image = image.numpy()

        image = np.transpose(

            image,

            (1, 2, 0)

        )

        image = (image * 255.0).round()

        image = image.astype(np.uint8)

        return image

    # ======================================================
    # UPSCALE
    # ======================================================

    def upscale(self, image):
        """
        Run SwinIR inference.
        """

        input_tensor = self.preprocess(image)

        input_tensor, h, w = self.pad_image(
            input_tensor
        )

        start = time.time()

        with torch.no_grad():

            output = self.model(input_tensor)

        inference_time = time.time() - start

        output = self.remove_padding(

            output,

            h,

            w

        )

        output = self.postprocess(output)

        return output, inference_time
    # ======================================================
    # RESTORE (PIPELINE FRIENDLY API)
    # ======================================================

    def restore(self, image_path, save_path=None):
        """
        Main function used by pipeline.

        Returns:
        - output image path
        - inference time
        """

        if isinstance(image_path, (str, Path)):

            image_path = Path(image_path)

            image = cv2.imread(str(image_path))

            if image is None:

                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        else:

            image = image_path

        print("\n===================================")
        print("SwinIR Super Resolution")
        print("===================================")

        output, inference_time = self.upscale(image)

        # --------------------------------------

        if save_path is None:

            save_dir = Path("outputs/restored")

            save_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            save_path = save_dir / "swinir_restored.png"

        else:

            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        # --------------------------------------

        cv2.imwrite(
            str(save_path),
            cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        )

        print("Image Saved :", save_path)
        print("Inference Time :", round(inference_time, 3), "sec")

        print("===================================\n")

        return str(save_path)

    # ======================================================
    # MAIN TEST FUNCTION
    # ======================================================


def main():

    IMAGE_PATH = r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/data/processed/gaussian_blur/00040534.png"

    CHECKPOINT = r"D:/PLACEMENT/HYPERVERGE/PIXEL_RESTORE/models/SwinIR/model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"

    restorer = SwinIRRestorer(
        checkpoint=CHECKPOINT,
        scale=4
    )

    result_path = restorer.restore(
        IMAGE_PATH
    )

    print("\nFinal Output Saved At:")
    print(result_path)


if __name__ == "__main__":
    main()