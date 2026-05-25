"""Federated learning server implementation."""

import flwr as flwr
from flwr.server.strategy import FedAvg, FedAdam, FedProx, Strategy
from typing import Optional, Dict, Callable, Tuple, List, Any


def get_fl_strategy(
    strategy_name: str,
    num_clients: int,
    mu: float = 0.0,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable[[int], Dict]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict]] = None,
    evaluate_metrics_aggregation_fn: Optional[Callable] = None,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
    initial_parameters: Optional[Any] = None,
) -> Strategy:
    """Get federated learning strategy by name.

    Args:
        strategy_name: Strategy name ('fedavg', 'fedadam', 'fedprox')
        num_clients: Total number of clients
        mu: FedProx proximal term coefficient
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        evaluate_fn: Evaluation function
        on_fit_config_fn: Function to generate fit config
        on_evaluate_config_fn: Function to generate evaluate config
        evaluate_metrics_aggregation_fn: Function to aggregate evaluation metrics
        fit_metrics_aggregation_fn: Function to aggregate fit metrics

    Returns:
        Flower Strategy instance
    """
    strategy_name = strategy_name.lower()

    # 使用传入值或默认值
    _min_fit = min_fit_clients if min_fit_clients is not None else int(num_clients * fraction_fit)
    _min_evaluate = min_evaluate_clients if min_evaluate_clients is not None else int(num_clients * fraction_evaluate)

    if strategy_name == "fedavg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=_min_fit,
            min_evaluate_clients=_min_evaluate,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    elif strategy_name == "fedadam":
        strategy = FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=_min_fit,
            min_evaluate_clients=_min_evaluate,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
            eta=1e-3,
            tau=1e-3,
        )
    elif strategy_name == "fedprox":
        strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=_min_fit,
            min_evaluate_clients=_min_evaluate,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            proximal_mu=mu,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Use 'fedavg', 'fedadam', or 'fedprox'.")

    return strategy


def get_fl_server(
    strategy_name: str,
    num_clients: int,
    mu: float = 0.0,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    server_address: str = "[::]:8080",
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable[[int], Dict]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict]] = None,
):
    """Create and configure Flower server.

    Args:
        strategy_name: Strategy name ('fedavg', 'fedadam', 'fedprox')
        num_clients: Total number of clients
        mu: FedProx proximal term coefficient
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        server_address: Server address
        evaluate_fn: Evaluation function
        on_fit_config_fn: Function to generate fit config
        on_evaluate_config_fn: Function to generate evaluate config

    Returns:
        Flower server
    """
    strategy = get_fl_strategy(
        strategy_name=strategy_name,
        num_clients=num_clients,
        mu=mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
    )

    server = flwr.server.start_server(
        server_address=server_address,
        strategy=strategy,
    )

    return server
