import os
import requests

def set_env():
    # CUDA , Nvidia Configs
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if requests.get("https://huggingface.co").status_code == 200:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    else:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    return True

if __name__ == "__main__":
    set_env()