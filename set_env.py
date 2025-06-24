import os

def set_env():
    # CUDA , Nvidia Configs
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Model and dataset configuration
    os.environ["MODEL_NAME"] = os.getenv("MODEL_NAME")
    os.environ["TRAINED_MODEL_NAME"] = os.getenv("TRAINED_MODEL_NAME")
    os.environ["DATASET_NAME"] = os.getenv("DATASET_NAME", "Intel/orca_dpo_pairs")

    # API tokens (you'll need to set these)
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    os.environ["WANDB_TOKEN"] = os.getenv("WANDB_TOKEN", "")

    # Training hyperparameters
    os.environ["EPOCHS"] = os.getenv("EPOCHS", "3")
    os.environ["LEARNING_RATE"] = os.getenv("LEARNING_RATE", "5e-5")
    os.environ["BATCH_SIZE"] = os.getenv("BATCH_SIZE", "4")
    os.environ["GA"] = os.getenv("GA", "4")  # gradient accumulation steps
    os.environ["WEIGHT_DECAY"] = os.getenv("WEIGHT_DECAY", "0.01")
    os.environ["SEED"] = os.getenv("SEED", "42")
    os.environ["MAX_SEQ_LENGTH"] = os.getenv("MAX_SEQ_LENGTH", "512")

    return True

if __name__ == "__main__":
    set_env()