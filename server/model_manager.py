import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from typing import List, OrderedDict
import numpy as np

from utils.config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, GLOBAL_MODEL_DIR, BASE_MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

def load_base_model():
    """Load the base model and tokenizer, applying LoRA."""
    logger.info(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer

def get_parameters(peft_model) -> List[np.ndarray]:
    """Extract peft parameters as a list of numpy arrays for Flower."""
    params = [val.cpu().numpy() for _, val in peft_model.state_dict().items() if "lora_" in _]
    return params

def set_parameters(peft_model, parameters: List[np.ndarray]) -> None:
    """Load list of numpy arrays back into the peft model state dict."""
    state_dict = peft_model.state_dict()
    keys = [k for k in state_dict.keys() if "lora_" in k]
    assert len(keys) == len(parameters), "Number of target parameters and given parameters doesn't match."
    
    for k, p in zip(keys, parameters):
        state_dict[k] = torch.tensor(p, dtype=state_dict[k].dtype)
    
    peft_model.load_state_dict(state_dict, strict=False)

def save_global_model(peft_model, tokenizer, round_num=None):
    """Save the aggregated global model."""
    save_dir = os.path.join(GLOBAL_MODEL_DIR, f"round_{round_num}") if round_num else GLOBAL_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving global model to {save_dir}")
    peft_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
