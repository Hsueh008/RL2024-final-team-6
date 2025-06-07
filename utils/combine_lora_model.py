from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    num_labels=1
)

adapter_path = "train/ckpt/deepseek-coder-6.7b-dpo-lora/checkpoint-xxx" # You need to input your ckpt path here.    
try:
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
except Exception as e:
    raise ValueError(f'Please replace "checkpoint-xxx" with the actual checkpoint directory name in "combine_model.py" source code.')

model = model.merge_and_unload()

output_dir = "model/deepseek-coder-6.7b-dpo-lora"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
