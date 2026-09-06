"""
Kahneman-Tversky KTO Utility Loss Skill Client
Pure Python Standard Library implementation of Kahneman-Tversky Optimization (KTO) (Ethayarajh et al.).
Optimizes directly on unpaired binary feedback (thumbs up / thumbs down) by modeling
human loss aversion (lambda_D > 1) and subjective reference points.
"""

import math
from typing import List, Dict, Any, Tuple


class KTOLossEvaluator:
    """
    KTO Prospect Theory loss evaluator for unpaired agent feedback.
    Utility: u(x, y) = beta * log(pi(y|x) / ref(y|x)) - z_ref
    Loss_desirable   = 1 - sigma(u(x, y))
    Loss_undesirable = 1 - sigma(- lambda_D * u(x, y))
    """

    def __init__(self, beta: float = 0.1, lambda_d: float = 1.33, z_ref: float = 0.0):
        """
        :param beta: Temperature scaling.
        :param lambda_d: Loss aversion coefficient (default 1.33 per prospect theory).
        :param z_ref: Subjective reference point KL offset.
        """
        self.beta = beta
        self.lambda_d = lambda_d
        self.z_ref = z_ref

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def evaluate_sample(self, policy_logprob: float, ref_logprob: float, is_desirable: bool) -> Dict[str, float]:
        """
        Evaluate KTO loss on an unpaired sample.
        """
        # Implicit reward r(x, y) = beta * (log pi - log ref)
        implicit_reward = self.beta * (policy_logprob - ref_logprob)
        utility = implicit_reward - self.z_ref

        if is_desirable:
            # Positive sample: maximize sigma(utility) -> minimize (1 - sigma(utility))
            sig = self._sigmoid(utility)
            loss = 1.0 - sig
        else:
            # Negative sample: maximize sigma(- lambda * utility) -> minimize (1 - sigma(- lambda * utility))
            sig = self._sigmoid(-self.lambda_d * utility)
            loss = 1.0 - sig

        return {
            "implicit_reward": implicit_reward,
            "utility": utility,
            "kto_loss": loss,
            "is_desirable": 1.0 if is_desirable else 0.0,
            "satisfaction_probability": sig
        }

    def update_reference_point(self, positive_rewards: List[float], negative_rewards: List[float]):
        """Dynamically update reference point z_ref from average sample rewards."""
        all_rewards = positive_rewards + negative_rewards
        if all_rewards:
            self.z_ref = sum(all_rewards) / len(all_rewards)
