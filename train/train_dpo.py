# train_dpo.py
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

################
# Arguments
################
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
data_files = "dataset/rlhf_dataset_dpo.json"
output_dir = "train/ckpt/deepseek-coder-6.7b-dpo-lora"
eval_ratio = 0.1

training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=12,
    learning_rate=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=3,
    max_steps=-1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="deepseek-coder-6.7b-dpo-lora",
    gradient_checkpointing=True,
    max_length=2048,
    seed=10,
    data_seed=10,
    save_only_model=True,
)
model_args = ModelConfig(
    use_peft=True,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_task_type="CAUSAL_LM",
    attn_implementation="flash_attention_2",
)
model_kwargs = dict(
    attn_implementation=model_args.attn_implementation,
    # device_map="auto",  # <-- Should be commented if using `accelerate launch`
    # torch_dtype=torch.bfloat16,
)

################
# Model & Tokenizer
################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    **model_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

################
# Dataset
################
dataset = load_dataset("json", data_files=data_files)["train"]
dataset = dataset.train_test_split(test_size=eval_ratio, seed=42)

################
# Training
################
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
trainer.train()
trainer.save_model(f"{output_dir}/best_model")
