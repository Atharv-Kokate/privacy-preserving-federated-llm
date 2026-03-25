import json
from datasets import Dataset

def load_local_dataset(filepath, tokenizer, max_length=512):
    """Load and tokenize local JSON dataset for model fine-tuning."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Format data into instructions
    formatted_data = []
    for item in data:
        text = f"<|user|>\n{item['question']}\n\n<|assistant|>\n{item['assistant']}<|endoftext|>"
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def get_data_collator(tokenizer):
    """Get the appropriate data collator."""
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(tokenizer, mlm=False)
