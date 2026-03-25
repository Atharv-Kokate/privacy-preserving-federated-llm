import os
import argparse
import flwr as fl
from client.dataset import load_local_dataset, get_data_collator
from client.trainer import train_local_model
from server.model_manager import load_base_model, get_parameters, set_parameters
from utils.config import SERVER_PORT, MAX_SEQ_LENGTH
from utils.logger import get_logger

logger = get_logger(__name__)

class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, data_collator):
        self.model = model
        self.train_dataset = train_dataset
        self.data_collator = data_collator

    def get_parameters(self, config):
        """Extract updated parameters from local model to send to the server."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the local model using dataset and global parameters."""
        logger.info("Received parameters from server. Setting to local model...")
        set_parameters(self.model, parameters)
        
        # Train locally using HuggingFace trainer
        train_local_model(self.model, self.train_dataset, self.data_collator)
        
        # Return updated parameters, number of local examples, and metrics
        return get_parameters(self.model), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate accuracy locally (optional for LLMs, returning loss=0.0)."""
        logger.info("Evaluating local model. Returning dummy metrics.")
        set_parameters(self.model, parameters)
        return 0.0, len(self.train_dataset), {"accuracy": 1.0}

def main():
    parser = argparse.ArgumentParser(description="Healthcare Federated Local Client")
    parser.add_argument("--server-ip", type=str, required=True, help="IP address of Flower Server")
    parser.add_argument("--dataset", type=str, required=True, help="Path to local mock dataset (e.g., datasets/hospital_A.json)")
    args = parser.parse_args()

    server_address = f"{args.server_ip}:{SERVER_PORT}"
    
    logger.info(f"Initializing local model and loading dataset from {args.dataset}")
    model, tokenizer = load_base_model()
    
    # Load dataset
    train_dataset = load_local_dataset(args.dataset, tokenizer, max_length=MAX_SEQ_LENGTH)
    data_collator = get_data_collator(tokenizer)

    client = HealthcareClient(model, train_dataset, data_collator)

    logger.info(f"Starting connection to server at {server_address}...")
    fl.client.start_client(server_address=server_address, client=client.to_client())

if __name__ == "__main__":
    main()
