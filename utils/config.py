import os

# Server settings
SERVER_IP = os.getenv("SERVER_IP", "0.0.0.0")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
SERVER_ADDRESS = f"{SERVER_IP}:{SERVER_PORT}"
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "3"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
GLOBAL_MODEL_DIR = os.path.join("models", "global")
BASE_MODEL_DIR = os.path.join("models", "base")

# Training & LoRA parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-4

# Inference settings
API_HOST = "0.0.0.0"
API_PORT = 8000
