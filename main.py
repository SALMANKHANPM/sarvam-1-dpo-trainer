from set_env import set_env

print("======= SETTING UP THE ENVIRONMENT ========")
val = set_env()
print("set_env() : ", val)
print("======= ENVIRONMENT SET UP SUCCESSFULLY ========")

print("======= IMPORTING LIBRARIES ========")
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

print("======= LIBRARIES IMPORTED SUCCESSFULLY ========")

print("======= LOADING .env FILE ========")
load_dotenv()
print("======= .env FILE LOADED SUCCESSFULLY ========")

print("======= SETTING UP THE MODEL ========")
# Setting up the model

# Model Name
model_name = os.getenv("MODEL_NAME")
trained_model_name = os.getenv("TRAINED_MODEL_NAME")

# Dataset Name
datasetName= os.getenv("DATASET_NAME")

hf_token = os.getenv("HF_TOKEN")
wandb_token = os.getenv("WANDB_TOKEN")

# Config
epochs = os.getenv("EPOCHS")
batch_size = os.getenv("BATCH_SIZE")
ga = os.getenv("GA")
weight_decay = os.getenv("WEIGHT_DECAY")
seed = os.getenv("SEED")
max_seq_length = os.getenv("MAX_SEQ_LENGTH")

print(f"""Model Name      : {model_name}
Dataset Name    : {datasetName}
Epochs          : {epochs}
Batch Size      : {batch_size}
GA              : {ga}
Weight Decay    : {weight_decay}
Seed            : {seed}
Max Seq Length  : {max_seq_length}
""")

print("======= SETTING UP HUGGING FACE & WANDB ========")
login(token=hf_token)
wandb.login(key=wandb_token)
print("======= HUGGING FACE & WANDB SET UP SUCCESSFULLY ========")

# ===============================

print("======= LOADING THE MODEL ========")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
print("======= MODEL LOADED SUCCESSFULLY ========")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("======= TOKENIZER LOADED SUCCESSFULLY ========")

# Dataset Tranformations
print("======= LOADING THE DATASET ========")
dataset = load_dataset(datasetName, split='train')
print("======= DATASET LOADED SUCCESSFULLY ========")

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
print("======= TRANSFORMING THE DATASET ========")
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns,
)
print("======= DATASET TRANSFORMED SUCCESSFULLY ========")

# Print sample
dataset[1]

print("======= SPLITTING THE DATASET ========")
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
print("======= DATASET SPLIT SUCCESSFULLY ========")

print("======= PRINTING THE DATASET ========")
print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print("======= DATASET PRINTED SUCCESSFULLY ========")

# ===============================
# Training arguments
print("======= SETTING UP THE TRAINING ARGUMENTS ========")
training_args = DPOConfig(
    per_device_train_batch_size=int(batch_size),
    gradient_accumulation_steps=int(ga),
    gradient_checkpointing=True,
    learning_rate=8e-5,
    lr_scheduler_type="cosine",
    #max_steps=200,
    num_train_epochs=int(epochs),
    warmup_ratio=0.03,
    save_strategy="no",
    logging_steps=1,
    output_dir=trained_model_name,
    optim="adamw_8bit",
    warmup_steps=10,
    bf16=True,
    report_to="wandb",
    weight_decay=float(weight_decay),
    beta=0.1,
    save_safetensors=True,
    max_prompt_length=1024,
    max_length=int(max_seq_length),
    dataloader_num_workers=0,  # Reduced for stability
    dataloader_pin_memory=False,
)
print("======= TRAINING ARGUMENTS SET UP SUCCESSFULLY ========")

# Create DPO trainer
print("======= CREATING THE DPO TRAINER ========")
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset, # 90%
    processing_class=tokenizer,
    eval_dataset=eval_dataset #10%
)
print("======= DPO TRAINER CREATED SUCCESSFULLY ========")

# Fine-tune model with DPO
print("======= TRAINING THE MODEL ========")
dpo_trainer.train()
print("======= TRAINING COMPLETED ========")

print("======= SAVING THE MODEL ========")
dpo_trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")
print("======= MODEL SAVED SUCCESSFULLY ========")

print("======= LOADING THE CHECKPOINTS ========")
model =  AutoModelForCausalLM.from_pretrained(
    "final_checkpoint",
    return_dict=True,
    torch_dtype=torch.bfloat16,

)
tokenizer = AutoTokenizer.from_pretrained("final_checkpoint")
print("======= CHECKPOINTS LOADED SUCCESSFULLY ========")

print("======= UPLOADING TO HUGGINGFACE ========")
model.push_to_hub(trained_model_name, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(trained_model_name, use_temp_dir=False, token=hf_token)
print("======= UPLOADED TO HUGGINGFACE SUCCESSFULLY ========")

print("========== TASK COMPLETED SUCCESSFULLY ==========")
