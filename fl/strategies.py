"""Custom federated learning strategies."""

import flwr as fl
from flwr.server.strategy import FedProx as FlwrFedProx
import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, List, Tuple


class CustomFedProx(FlwrFedProx):
    """Custom FedProx strategy (backup implementation).

    This class is preserved for reference/backups.
    The project now uses flwr.server.strategy.FedProx directly.

    Extends Flower FedProx with better support for PiXTime training.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, fl.common.Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, fl.common.Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
        mu: float = 1.0,
        epsilon: float = 1e-5,
        beta: float = 1.0,
    ):
        """Initialize FedProx strategy.

        Args:
            mu: Proximal term coefficient
            Other args: Same as fl.strategy.FedProx
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            mu=mu,
            epsilon=epsilon,
            beta=beta,
        )


def compute_proximal_loss(
    local_model: nn.Module,
    global_parameters: List[torch.nn.Parameter],
    mu: float,
) -> torch.Tensor:
    """Compute the proximal term for FedProx.

    Loss_total = Loss_task + mu/2 * ||w_local - w_global||^2

    Args:
        local_model: Local model
        global_parameters: Global model parameters from server
        mu: Proximal term coefficient

    Returns:
        Proximal loss term
    """
    proximal_loss = 0.0
    for local_param, global_param in zip(
        local_model.parameters(), global_parameters
    ):
        proximal_loss += torch.sum((local_param - global_param) ** 2)
    return (mu / 2) * proximal_loss


def apply_proximal_loss(
    task_loss: torch.Tensor,
    local_model: nn.Module,
    global_parameters: List[torch.nn.Parameter],
    mu: float,
) -> torch.Tensor:
    """Apply proximal term to task loss.

    Args:
        task_loss: Task loss (e.g., MSE)
        local_model: Local model
        global_parameters: Global model parameters from server
        mu: Proximal term coefficient

    Returns:
        Total loss with proximal term
    """
    if mu > 0:
        proximal = compute_proximal_loss(local_model, global_parameters, mu)
        return task_loss + proximal
    return task_loss
