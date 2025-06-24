from set_env import set_env

val = set_env()
print("set_env() : ", val)


# Importing Libraries
import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import wandb
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Setting up the model

# Model Name
model_name = os.getenv("MODEL_NAME")
trained_model_name = os.getenv("TRAINED_MODEL_NAME")

# Dataset Name
dataset_name = os.getenv("DATASET_NAME")

hf_token = os.getenv("HF_TOKEN")
wandb_token = os.getenv("WANDB_TOKEN")

# Config
epochs = int(os.getenv("EPOCHS"))
learning_rate = os.getenv("LEARNING_RATE")
batch_size = int(os.getenv("BATCH_SIZE"))
ga = int(os.getenv("GA"))
weight_decay = float(os.getenv("WEIGHT_DECAY"))
seed = int(os.getenv("SEED"))
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH"))


# Set up Hugging Face & Wandb
login(token=hf_token)
wandb.login(key=wandb_token)

# ===============================

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Dataset Tranformations
dataset = load_dataset(dataset_name)['train']

# Chatml Format
def chatml_format(example):
    # bos and eos tokens
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    system_msg = example.get('system', '') or ''
    # Format system and user message together
    if len(system_msg) > 0:
        # System message is embedded in the first user message
        system_content = f"<<SYS>>\n{example['system']}\n<</SYS>>\n\n{example['question']}"
    else:
        system_content = example['question']

    # Format the prompt with user message
    prompt = f"{bos_token}[INST] {system_content.strip()} [/INST]"

    # Format chosen answer (assistant response)
    chosen = f" {example['chosen'].strip()} {eos_token}"

    # Format rejected answer (assistant response)
    rejected = f" {example['rejected'].strip()} {eos_token}"

    return {
	    "prompt": prompt,
	    "chosen": chosen,
	    "rejected": rejected,
	}

# Save columns
original_columns = dataset.column_names

# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns,
)

# Print sample
dataset[1]


dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
print("==================")
print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print("==================")


# ===============================
# Training arguments
training_args = DPOConfig(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=ga,
    gradient_checkpointing=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    #max_steps=200,
    num_train_epochs=epochs,
    warmup_ratio=0.03,
    save_strategy="no",
    logging_steps=1,
    output_dir=trained_model_name,
    optim="adamw_8bit",
    warmup_steps=10,
    bf16=True,
    report_to="wandb",
    weight_decay=weight_decay,
    beta=0.1,
    save_safetensors=True,
    max_prompt_length=max_seq_length,
    max_length=max_seq_length,
    dataloader_num_workers=0,  # Reduced for stability
    dataloader_pin_memory=False,
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset, # 90%
    processing_class=tokenizer,
    eval_dataset=eval_dataset #10%
)

# Fine-tune model with DPO
dpo_trainer.train()

# ===============================

dpo_trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

# Loading the checkpoints
model =  AutoModelForCausalLM.from_pretrained(
    "final_checkpoint",
    return_dict=True,
    torch_dtype=torch.bfloat16,

)
tokenizer = AutoTokenizer.from_pretrained("final_checkpoint")

# Uploading to Huggingface
model.push_to_hub(trained_model_name, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(trained_model_name, use_temp_dir=False, token=hf_token)

print("Training Completed")
