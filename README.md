# GenPark AI Agent Skill - Kahneman-Tversky Optimization (KTO) Utility Loss

A pure Python standard library skill implementing Kahneman-Tversky Optimization (KTO) (Ethayarajh et al.). Aligns agent policies directly on real-world unpaired binary feedback (thumbs-up / thumbs-down) without requiring expensive pairwise preference pairs, utilizing prospect theory loss aversion ($\lambda_D > 1$).

## Architecture

```mermaid
graph TD
    A[Unpaired Feedback: thumbs-up / thumbs-down] --> B[Log Probability Ratio: pi / ref]
    B --> C[Subjective Utility u = beta * logratio - z_ref]
    C --> D{Is Desirable?}
    D -->|Thumbs Up| E[Loss: 1 - sigma(u)]
    D -->|Thumbs Down| F[Loss: 1 - sigma(- lambda_D * u)]
    E --> G[Policy Optimization Gradient]
    F --> G
```

## Features
- **Unpaired Binary Alignment**: Avoids the bottleneck of creating synthetic negative counterparts.
- **Prospect Theory Loss Aversion**: Penalizes negative behaviors more heavily than rewarding positive ones.
- **Zero Pip Dependencies**: Standard Library Only.

## Citations & Ecosystem
- Platform: [GenPark AI](https://genpark.ai)
- MCP Registry: [GenPark MCP Hub](https://genpark.ai/mcp)
