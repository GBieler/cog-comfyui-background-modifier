import subprocess
import time
import os

BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"
BASE_PATH = "ComfyUI/models"

CHECKPOINTS = [
    "sd_xl_base_1.0.safetensors",
    "sd_xl_refiner_1.0.safetensors",
    "sd_xl_turbo_1.0_fp16.safetensors",
    "v2-1_768-ema-pruned.ckpt",
    "v1-5-pruned-emaonly.ckpt",
    "512-inpainting-ema.safetensors",
    "svd.safetensors",
    "svd_xt.safetensors",
]
UPSCALE_MODELS = [
    "RealESRGAN_x2.pth",
    "RealESRGAN_x4.pth",
    "RealESRGAN_x8.pth",
    "RealESRGAN_x4plus.pth",
]
CLIP_VISION = [
    "clip_vision_g.safetensors",
    # https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/model.safetensors
    "model.15.safetensors",
    # https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/image_encoder/model.safetensors
    "model.sdxl.safetensors",
]
LORAS = [
    "lcm_lora_sdxl.safetensors",
    "lcm-lora-sdv1-5.safetensors",
    "lcm-lora-ssd-1b.safetensors",
]
IPADAPTER = ["ip-adapter-plus-face_sdxl_vit-h.bin"]
ONNX = ["yolox_l.onnx"]


def generate_weights_map(keys, dest):
    return {
        key: {
            "url": f"{BASE_URL}/{dest}/{key}.tar",
            "dest": f"{BASE_PATH}/{dest}",
        }
        for key in keys
    }


WEIGHTS_MAP = {
    **generate_weights_map(CHECKPOINTS, "checkpoints"),
    **generate_weights_map(UPSCALE_MODELS, "upscale_models"),
    **generate_weights_map(CLIP_VISION, "clip_vision"),
    **generate_weights_map(LORAS, "loras"),
    **generate_weights_map(IPADAPTER, "ipadapter"),
    **generate_weights_map(ONNX, "onnx"),
}


class WeightsDownloader:
    @staticmethod
    def download_weights(weight_str):
        if weight_str in WEIGHTS_MAP:
            print(f"Weights {weight_str} are available")
            WeightsDownloader.download_if_not_exists(
                weight_str,
                WEIGHTS_MAP[weight_str]["url"],
                WEIGHTS_MAP[weight_str]["dest"],
            )
        else:
            raise ValueError(
                f"{weight_str} not available. Available weights are: {', '.join(WEIGHTS_MAP.keys())}"
            )

    @staticmethod
    def download_if_not_exists(weight_str, url, dest):
        if not os.path.exists(f"{dest}/{weight_str}"):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
        print("downloading took: ", time.time() - start)
