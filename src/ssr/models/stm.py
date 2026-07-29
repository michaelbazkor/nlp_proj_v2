"""STM and MTM architectures (Ophir et al. 2020 style)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(name)


def _fc_stack(in_dim: int, n_layers: int, n_neurons: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(d, n_neurons))
        layers.append(_act(activation))
        d = n_neurons
    return nn.Sequential(*layers)


class STM(nn.Module):
    """Single-task model: input -> FC stack -> suicide logit."""

    def __init__(self, in_dim: int, n_layers: int = 2, n_neurons: int = 32, activation: str = "tanh"):
        super().__init__()
        self.trunk = _fc_stack(in_dim, n_layers, n_neurons, activation)
        self.head = nn.Linear(n_neurons if n_layers > 0 else in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x) if len(list(self.trunk.children())) else x
        logit = self.head(h).squeeze(-1)
        return {"suicide_logit": logit}


class MTM(nn.Module):
    """Multi-task hierarchical model.

    Shared trunk from input; subnets:
      personality (5) -> psychosocial (4) -> psychiatric (2) -> suicide (1)
    Each subnet consumes concat(prev_output, shared_trunk), matching the
    paper's residual skip connections from the shared layers.
    """

    PERSONALITY = ["BFI_O", "BFI_C", "BFI_E", "BFI_A", "BFI_N"]  # 5
    PSYCHOSOCIAL = ["Brooding", "Worry", "Lonely", "SWL"]  # 4
    PSYCHIATRIC = ["PHQ9", "GAD"]  # 2

    def __init__(self, in_dim: int, n_layers: int = 2, n_neurons: int = 32, activation: str = "tanh"):
        super().__init__()
        self.shared = _fc_stack(in_dim, n_layers, n_neurons, activation)
        shared_dim = n_neurons if n_layers > 0 else in_dim

        def subnet(in_d: int, out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_d, n_neurons),
                _act(activation),
                nn.Linear(n_neurons, out_d),
            )

        # First subnet takes shared only
        self.personality = subnet(shared_dim, 5)
        # Subsequent take concat(prev, shared)
        self.psychosocial = subnet(5 + shared_dim, 4)
        self.psychiatric = subnet(4 + shared_dim, 2)
        self.suicide = subnet(2 + shared_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x) if len(list(self.shared.children())) else x
        pers = self.personality(shared)
        psy = self.psychosocial(torch.cat([pers, shared], dim=-1))
        psych = self.psychiatric(torch.cat([psy, shared], dim=-1))
        sui = self.suicide(torch.cat([psych, shared], dim=-1)).squeeze(-1)
        return {
            "suicide_logit": sui,
            "personality": pers,
            "psychosocial": psy,
            "psychiatric": psych,
        }
