import os

def set_env():
    # CUDA , Nvidia Configs
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    return True

if __name__ == "__main__":
    set_env()