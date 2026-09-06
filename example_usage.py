"""
Example usage of Kahneman-Tversky KTO Utility Loss Skill.
"""

from client import KTOLossEvaluator


def main():
    print("=== Kahneman-Tversky KTO Utility Loss Demonstration ===")
    kto = KTOLossEvaluator(beta=0.1, lambda_d=1.33)

    # Simulated stream of unpaired feedback
    samples = [
        {"pi": -2.0, "ref": -3.5, "desirable": True},   # Good answer approved by user
        {"pi": -1.5, "ref": -2.0, "desirable": True},   # Moderate good answer
        {"pi": -4.0, "ref": -2.0, "desirable": False},  # Hallucinated answer downvoted
        {"pi": -1.2, "ref": -3.0, "desirable": False}   # Confident bad answer severely penalized
    ]

    print(f"Evaluating {len(samples)} unpaired feedback samples (Loss Aversion lambda={kto.lambda_d}):\n")
    for idx, s in enumerate(samples):
        res = kto.evaluate_sample(s["pi"], s["ref"], s["desirable"])
        label = "[DESIRABLE]" if s["desirable"] else "[UNDESIRABLE]"
        print(f"Sample #{idx + 1} {label}:")
        print(f"  Implicit Reward: {res['implicit_reward']:+.4f}")
        print(f"  Utility:         {res['utility']:+.4f}")
        print(f"  KTO Loss:        {res['kto_loss']:.4f}")
        print(f"  Satisfaction P:  {res['satisfaction_probability'] * 100:.1f}%\n")


if __name__ == "__main__":
    main()
