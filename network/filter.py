import torch
import torch.nn as nn

from .dynamics import StateConditionedDynamics


class AdaptiveKalmanFilter(nn.Module):
    def __init__(
        self,
        state_dim: int,
        q_min: float,
        r_min: float,
        initial_q: float,
        initial_r: float,
        alpha_q: float,
        alpha_r: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.q_min = float(q_min)
        self.r_min = float(r_min)
        self.initial_q = float(initial_q)
        self.initial_r = float(initial_r)
        self.alpha_q = float(alpha_q)
        self.alpha_r = float(alpha_r)
        self.eps = float(eps)

    def _positive(self, covariance: torch.Tensor, lower: float) -> torch.Tensor:
        return covariance.clamp_min(max(self.eps, lower))

    def forward(
        self,
        measurement_sequence: torch.Tensor,
        dynamics: StateConditionedDynamics,
    ) -> torch.Tensor:
        if measurement_sequence.ndim != 3:
            raise ValueError("measurement_sequence must be [batch, sequence, state]")
        batch_size, sequence_length, state_dim = measurement_sequence.shape
        if state_dim != self.state_dim:
            raise ValueError(f"H=I requires state_dim={self.state_dim}")

        filtered_states = torch.zeros_like(measurement_sequence)
        state_post = measurement_sequence[:, 0]
        q_running = state_post.new_full(state_post.shape, self.initial_q)
        r_running = state_post.new_full(state_post.shape, self.initial_r)
        covariance_post = r_running.clamp_min(self.eps)
        recurrent_state = dynamics.init_context(state_post)
        filtered_states[:, 0] = state_post

        for step_idx in range(1, sequence_length):
            transition_diagonal, bias = dynamics.state_prediction_from_context(
                recurrent_state
            )
            state_pred = transition_diagonal * state_post + bias
            covariance_pred = (
                transition_diagonal.square() * covariance_post + q_running
            ).clamp_min(self.eps)
            innovation = measurement_sequence[:, step_idx] - state_pred
            innovation_covariance = (covariance_pred + r_running).clamp_min(self.eps)
            gain = covariance_pred / innovation_covariance
            correction = gain * innovation
            state_post = state_pred + correction
            residual = measurement_sequence[:, step_idx] - state_post
            covariance_post = (
                (1.0 - gain).square() * covariance_pred
                + gain.square() * r_running
            ).clamp_min(self.eps)

            q_candidate = self._positive(correction.detach().square(), self.q_min)
            r_candidate = self._positive(
                residual.detach().square() + covariance_pred.detach(), self.r_min
            )
            q_running = self._positive(
                self.alpha_q * q_running + (1.0 - self.alpha_q) * q_candidate,
                self.q_min,
            )
            r_running = self._positive(
                self.alpha_r * r_running + (1.0 - self.alpha_r) * r_candidate,
                self.r_min,
            )
            recurrent_state = dynamics.update_context(state_post, recurrent_state)
            filtered_states[:, step_idx] = state_post

        return filtered_states

