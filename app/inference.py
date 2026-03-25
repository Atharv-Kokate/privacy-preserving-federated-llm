import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.config import MODEL_NAME, GLOBAL_MODEL_DIR, NUM_ROUNDS
from utils.logger import get_logger

logger = get_logger(__name__)

class InferenceHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        logger.info(f"Loading final global model for inference... (device: {self.device})")
        
        # Determine latest round model directory
        final_model_dir = os.path.join(GLOBAL_MODEL_DIR, f"round_{NUM_ROUNDS}")
        if not os.path.exists(final_model_dir):
            final_model_dir = GLOBAL_MODEL_DIR  # fallback to generic dir

        if not os.path.exists(final_model_dir) or not os.listdir(final_model_dir):
            logger.warning(f"Global model directory {final_model_dir} not found or empty.")
            logger.warning("Loading base model instead for testing purposes.")
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            return

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, final_model_dir).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
        logger.info("Model loaded successfully.")

    def generate_response(self, question: str, max_length=256) -> str:
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded."

        prompt = f"<|user|>\n{question}\n\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant part
        if "<|assistant|>\n" in response:
            answer = response.split("<|assistant|>\n")[-1].strip()
        else:
            answer = response.replace(prompt, "").strip()
            
        return answer
