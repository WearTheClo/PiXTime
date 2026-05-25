"""Flower Client implementation for multi-model time series forecasting."""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class Client(fl.client.NumPyClient):
    """Flower client supporting multiple time series forecasting models.

    Supports: DLinear, PatchTST, iTransformer, TimeXer, PiXTime, Autoformer

    Server-controlled parameters (received via fit/evaluate config):
    - seq_len, label_len, patch_len, pred_len: data dimensions
    - features, target: prediction mode and target variable
    - partition_method, dirichlet_alpha, seed: data partitioning
    - d_model, dropout, factor, n_heads, en_d_ff, de_d_ff, en_layers, de_layers: model hyperparameters
    - fl_strategy, mu: FL strategy parameters
    """

    # Models that require dec_inp (encoder-decoder architecture)
    DEC_INP_MODELS = {"PiXTime", "Autoformer"}
    # Models that support features parameter (MS mode)
    FEATURES_MODELS = {"PiXTime", "TimeXer"}

    def __init__(
        self,
        cid: int,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        device: torch.device,
        model: Optional[nn.Module] = None,
        model_type: str = "PiXTime",
        model_kwargs: Optional[Dict[str, Any]] = None,
        train_loader=None,
        test_loader=None,
        num_clients: int = 1,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        patch_len: int = 16,
    ):
        """Initialize federated learning client.

        Args:
            cid: Client ID for logging
            batch_size: Batch size for training
            local_epochs: Number of local epochs per round
            learning_rate: Learning rate
            device: Device to use for training
            model: Pre-instantiated model
            model_type: Model type name (DLinear, PatchTST, iTransformer, TimeXer, PiXTime, Autoformer)
            model_kwargs: Additional kwargs for model-specific forward pass
            train_loader: Pre-partitioned training DataLoader
            test_loader: Full test DataLoader
            num_clients: Total number of clients (unused in simulation mode)
            seq_len: Input sequence length
            label_len: Label sequence length
            pred_len: Prediction sequence length
            patch_len: Patch length
        """
        self.cid = cid
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.num_clients = num_clients

        # Pre-loaded data and model (simulation mode)
        self.model = model
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Server-controlled parameters
        self.config: Dict[str, Any] = {}
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.label_len = label_len
        self.mu: float = 0.0
        self.global_parameters: Optional[List[torch.nn.Parameter]] = None

        # Optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_func = nn.MSELoss()

        # Initialize optimizer if model is provided
        if self.model is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"  [Client {self.cid}] 客户端初始化完成, model_type={model_type}")

    def get_parameters(self, config: Dict[str, fl.common.Scalar] = None, model: nn.Module = None) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays.

        Args:
            config: Unused, kept for Flower NumPyClient interface compatibility.
            model: If provided, pack this model's parameters instead of self.model.
        """
        target = model if model is not None else self.model
        if target is None:
            return []
        return [val.cpu().numpy() for _, val in target.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from list of numpy arrays."""
        state_dict = self.model.state_dict()
        for (key, _), val in zip(state_dict.items(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def _init_from_config(self, config: Dict[str, Any]) -> None:
        """Update server-controlled parameters from config.

        In simulation mode, model and data are already provided at init.
        This only updates mu and other config-driven parameters.

        Args:
            config: Configuration dict from server
        """
        self.mu = config.get("mu", 0.0)

    def _forward_pass(self, batch_x, batch_x_mark, batch_y, batch_y_mark, pred_len):
        """Execute forward pass based on model type.

        Args:
            batch_x: encoder input [batch, seq_len, n_vars]
            batch_x_mark: encoder time marks [batch, seq_len, n_mark]
            batch_y: target [batch, label_len+pred_len, n_vars]
            batch_y_mark: decoder time marks [batch, label_len+pred_len, n_mark]
            pred_len: prediction length

        Returns:
            model outputs
        """
        # Create decoder input for models that need it
        if self.model_type in self.DEC_INP_MODELS:
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            # DLinear, PatchTST, iTransformer, TimeXer - they ignore dec_inp/x_mark_dec
            outputs = self.model(batch_x, batch_x_mark, None, None)
        return outputs

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """Train model on local data.

        Args:
            parameters: Global model parameters from server
            config: Configuration from server (contains all server-controlled params)

        Returns:
            tuple: (updated_parameters, num_samples, metrics)
        """
        # Store config for evaluation
        self.config = {k: v for k, v in config.items()}

        # Update server-controlled parameters from config
        self._init_from_config(config)

        self.set_parameters(parameters)

        # Update global parameters for FedProx
        if self.mu > 0:
            self.global_parameters = [p.clone().detach() for p in self.model.parameters()]

        # Handle MS mode: determine which dimension to use for loss
        features = self.config.get("features", "M")
        f_dim = -1 if features == "MS" else 0

        # Local training
        self.model.train()
        train_loss = []

        for epoch in range(self.local_epochs):
            epoch_loss = []
            for i, batch in enumerate(self.train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                self.optimizer.zero_grad()

                # Forward pass based on model type
                pred_len = self.config.get("pred_len", self.pred_len)
                outputs = self._forward_pass(batch_x, batch_x_mark, batch_y, batch_y_mark, pred_len)

                # Compute task loss (handle MS mode)
                pred_true = batch_y[:, -pred_len:, f_dim:] if features == "MS" else batch_y[:, -pred_len:, :]
                pred_output = outputs[:, -pred_len:, f_dim:] if features == "MS" else outputs
                task_loss = self.loss_func(pred_output, pred_true)

                # Add proximal term for FedProx
                if self.mu > 0 and self.global_parameters is not None:
                    proximal_loss = 0.0
                    for local_p, global_p in zip(
                        self.model.parameters(), self.global_parameters
                    ):
                        proximal_loss += torch.sum((local_p - global_p) ** 2)
                    total_loss = task_loss + (self.mu / 2) * proximal_loss
                else:
                    total_loss = task_loss

                total_loss.backward()
                self.optimizer.step()
                epoch_loss.append(total_loss.item())

            train_loss.append(np.mean(epoch_loss))

        # Calculate number of samples
        num_samples = len(self.train_loader.dataset)

        metrics = {
            "train_loss": float(np.mean(train_loss)),
            "local_epochs": self.local_epochs,
        }

        # 假设需要本地/全局模块化部署，这里需要重写self.get_parameters，使其能够只打包接收到的模型参数，而不是把整个self.model都打包
        # 或者是直接把本地模块初始化为其他的成员
        return self.get_parameters(config), num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate model on local test data.

        Args:
            parameters: Global model parameters from server
            config: Configuration from server

        Returns:
            tuple: (loss, num_samples, metrics)
        """
        # Update server-controlled parameters from config
        self._init_from_config(config)

        self.set_parameters(parameters)

        # Handle MS mode
        features = config.get("features", "M")
        f_dim = -1 if features == "MS" else 0
        pred_len = config.get("pred_len", self.pred_len)

        self.model.eval()
        val_loss = []
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Forward pass based on model type
                outputs = self._forward_pass(batch_x, batch_x_mark, batch_y, batch_y_mark, pred_len)

                # Handle MS mode for loss and metrics
                pred_true = batch_y[:, -pred_len:, f_dim:]
                pred_output = outputs[:, -pred_len:, f_dim:] if features == "MS" else outputs

                loss = self.loss_func(pred_output, pred_true)
                val_loss.append(loss.item())

                preds.append(pred_output.cpu().numpy())
                trues.append(pred_true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Compute metrics
        mae = float(np.mean(np.abs(trues - preds)))
        mse = float(np.mean((trues - preds) ** 2))

        num_samples = len(self.test_loader.dataset)
        metrics = {
            "cid": self.cid,
            "mae": mae,
            "mse": mse,
        }

        return float(np.mean(val_loss)), num_samples, metrics
