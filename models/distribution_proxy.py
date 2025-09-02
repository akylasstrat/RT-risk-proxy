# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:51:07 2025

@author: a.stratigakos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.utils import clip_grad_norm_

# ----------------------------
# Utility: empirical quantile fn
# ----------------------------
def empirical_quantile(y_samples, q_grid):
    """
    y_samples: tensor (n_samples,)
    q_grid: tensor of quantiles in [0,1], shape (n_quantiles,)
    Returns quantile values (n_quantiles,)
    """
    sorted_y, _ = torch.sort(y_samples)
    idx = (q_grid * (len(sorted_y) - 1)).long()
    return sorted_y[idx]

# ----------------------------
# Model: dictionary + weights
# ----------------------------
class DistributionProxy(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dict_size, n_quantiles=9, activation=nn.ReLU()):
        """
        input_size (int): Number of input features.
        hidden_sizes (list[int]): Hidden layer sizes.
        dict_size: number of dictionary distributions (number of targets)
        n_quantiles: size of quantile grid for approximation
        activation (nn.Module): Activation function.
        """
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.n_quantiles = n_quantiles

        # Network mapping x -> weights over dictionary
        layer_sizes = [input_dim] + hidden_sizes + [dict_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        self.fc = nn.Sequential(*layers)

        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, dict_size)
        # )

        # Dictionary distributions: learnable quantile functions
        # Shape: (dict_size, n_quantiles)
        self.dictionary = nn.Parameter(torch.randn(dict_size, n_quantiles))


    def forward(self, x):
        """
        x: features, shape (batch, input_dim)
        Returns: combined quantile function (batch, n_quantiles)
        """
        logits = self.fc(x)                      # (batch, dict_size)
        weights = F.softmax(logits, dim=-1)      # (batch, dict_size)

        # Weighted combination of dictionary quantile functions
        q_hat = weights @ self.dictionary        # (batch, n_quantiles)
        return q_hat

# ----------------------------
# Wrapper = model + loss
# ----------------------------
class DistributionProxyWrapper:
    def __init__(self, model: nn.Module, q_grid: torch.Tensor, loss_kind="l2"):
        """
        loss_kind: 'l2' ~ squared L2 between quantile functions (≈ W2^2 up to const),
                   'l1' ~ absolute error between quantile functions (≈ W1)
        """
        self.model = model
        self.q_grid = q_grid
        assert loss_kind in {"l1", "l2"}
        self.loss_kind = loss_kind

    def compute_loss(self, batch):
        x_batch, y_list = batch   # x: (B,D), y_list: list of (Ny,)
        q_hat = self.model(x_batch)                           # (B,Q)
        q_true = torch.stack([empirical_quantile(y, self.q_grid) for y in y_list])  # (B,Q)

        if self.loss_kind == "l2":
            loss = torch.mean((q_hat - q_true) ** 2)
        else:
            loss = torch.mean(torch.abs(q_hat - q_true))
        return loss

# ----------------------------
# Training loop with early stopping (adapted to wrapper)
# ----------------------------
def train_model(wrapper, train_loader, val_loader=None,
                optimizer=None, train_loss_fn=None, val_loss_fn=None,
                num_epochs=100, patience=10, device="cpu", max_grad_norm=None, verbose=True):
    """
    Returns: best_model, logs
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=1e-3)

    best_model_state = copy.deepcopy(wrapper.model.state_dict())
    best_val_loss = float("inf")
    early_stop_counter = 0
    logs = {"train_loss": [], "val_loss": []}

    wrapper.model.to(device)
    # put q-grid on device too (used inside loss)
    wrapper.q_grid = wrapper.q_grid.to(device)

    for epoch in range(num_epochs):
        # --- Training ---
        wrapper.model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            # batch = (x_batch, y_list)
            x_batch = batch[0].to(device)
            y_list = [y.to(device) for y in batch[1]]
            batch = (x_batch, y_list)

            loss = train_loss_fn(wrapper, batch) if train_loss_fn else wrapper.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(wrapper.model.parameters(), max_grad_norm)
            optimizer.step()

            B = x_batch.shape[0]
            train_loss += loss.item() * B
            n_train += B

        train_loss /= max(n_train, 1)
        logs["train_loss"].append(train_loss)

        # --- Validation ---
        if val_loader is not None:
            wrapper.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(device)
                    y_list = [y.to(device) for y in batch[1]]
                    batch = (x_batch, y_list)

                    loss = val_loss_fn(wrapper, batch) if val_loss_fn else wrapper.compute_loss(batch)
                    B = x_batch.shape[0]
                    val_loss += loss.item() * B
                    n_val += B

            val_loss /= max(n_val, 1)
            logs["val_loss"].append(val_loss)

            if verbose:
                print(f"[Epoch {epoch+1:03d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss - 1e-9:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(wrapper.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    if verbose:
                        print("Early stopping.")
                    break
        else:
            if verbose:
                print(f"[Epoch {epoch+1:03d}] Train: {train_loss:.4f}")

    # Restore best weights
    wrapper.model.load_state_dict(best_model_state)
    return wrapper.model, logs
