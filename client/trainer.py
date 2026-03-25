import torch
from transformers import Trainer, TrainingArguments
from utils.config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from utils.logger import get_logger

logger = get_logger(__name__)

def train_local_model(model, train_dataset, data_collator, output_dir="./local_training_output"):
    """Run local LoRA fine-tuning using HuggingFace Trainer."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting local model training...")
    trainer.train()
    logger.info("Local training completed.")
    
    return model
