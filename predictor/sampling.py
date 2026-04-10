"""Scheduled sampling scheduler.

Linear ramp: at each training step, with probability p replace
ground-truth input tokens with the model's own argmax predictions.
Closes the train/inference gap where the model only sees ground-truth
tokens during training but its own (possibly wrong) predictions at inference.

Do not ramp faster than specified — instability before the model is
competent causes divergence.
"""

import random


class ScheduledSamplingScheduler:
    """Linear ramp of self-prediction probability.

    Args:
        max_p: Maximum probability (default 0.5)
        ramp_steps: Steps over which to linearly ramp (default 100k)
    """

    def __init__(self, max_p: float = 0.5,
                 ramp_steps: int = 100_000) -> None:
        assert 0.0 <= max_p <= 1.0
        assert ramp_steps > 0
        self.max_p = max_p
        self.ramp_steps = ramp_steps

    def get_p(self, step: int) -> float:
        """Get current sampling probability.

        Args:
            step: Current training step

        Returns:
            p in [0, max_p]
        """
        if step >= self.ramp_steps:
            return self.max_p
        return self.max_p * (step / self.ramp_steps)

    def should_use_own_prediction(self, step: int) -> bool:
        """Sample whether to use model's own prediction at this step.

        Args:
            step: Current training step

        Returns:
            True if should use own prediction, False for ground truth
        """
        return random.random() < self.get_p(step)
