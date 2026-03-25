import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import os

from utils.logger import get_logger
from server.model_manager import load_base_model, set_parameters, save_global_model

logger = get_logger(__name__)

class SaveModelStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model, self.tokenizer = load_base_model()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        logger.info(f"Aggregating fit results for round {server_round} from {len(results)} clients.")
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert parameters back to ndarrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Apply aggregated weights to global model
            set_parameters(self.global_model, aggregated_ndarrays)
            
            # Save final model after last round, or optionally every round
            from utils.config import NUM_ROUNDS
            if server_round == NUM_ROUNDS:
                logger.info(f"Final round {server_round} completed. Saving final global model.")
                save_global_model(self.global_model, self.tokenizer)
            else:
                logger.info(f"Round {server_round} completed.")

        return aggregated_parameters, metrics_aggregated
