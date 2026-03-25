import flwr as fl
from flwr.common import ndarrays_to_parameters
from server.strategy import SaveModelStrategy
from server.model_manager import load_base_model, get_parameters
from utils.config import SERVER_ADDRESS, NUM_ROUNDS, MIN_AVAILABLE_CLIENTS
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Initializing Flower Server...")
    
    # Initialize base model for the first round
    model, _ = load_base_model()
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = SaveModelStrategy(
        fraction_fit=1.0,               # Sample 100% of available clients for training
        fraction_evaluate=1.0,          # Sample 100% of available clients for evaluation
        min_fit_clients=MIN_AVAILABLE_CLIENTS, # Minimum number of clients to train
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        initial_parameters=initial_parameters,
    )

    logger.info(f"Starting Flower Server on {SERVER_ADDRESS}")
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    logger.info("Federated Learning process completed.")

if __name__ == "__main__":
    main()
