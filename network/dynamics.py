import torch
import torch.nn as nn


class StateConditionedDynamics(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, delta_t: float) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_t = float(delta_t)
        self.state_context_cell = nn.LSTMCell(self.state_dim, self.hidden_dim)
        self.diagonal_generator_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.transition_drift_head = nn.Linear(self.hidden_dim, self.state_dim)

        nn.init.zeros_(self.diagonal_generator_head.weight)
        nn.init.zeros_(self.diagonal_generator_head.bias)
        nn.init.zeros_(self.transition_drift_head.weight)
        nn.init.zeros_(self.transition_drift_head.bias)

    def init_context(
        self, initial_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state.ndim != 2 or initial_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"initial_state must have shape [batch, {self.state_dim}]"
            )
        hidden = initial_state.new_zeros(initial_state.shape[0], self.hidden_dim)
        cell = initial_state.new_zeros(initial_state.shape[0], self.hidden_dim)
        return self.state_context_cell(initial_state, (hidden, cell))

    def state_prediction_from_context(
        self, recurrent_state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = recurrent_state
        diagonal_rate = self.diagonal_generator_head(hidden)
        transition_diagonal = 1.0 + self.delta_t * diagonal_rate
        drift_rate = self.transition_drift_head(hidden)
        transition_bias = self.delta_t * drift_rate
        return transition_diagonal, transition_bias

    def update_context(
        self,
        posterior_state: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state_context_cell(posterior_state, recurrent_state)

