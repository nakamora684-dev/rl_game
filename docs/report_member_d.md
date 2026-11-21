# Report/slides material — Member D (draft; fill in results after training)

## PPO rationale
Continuous/hybrid control with a clipped surrogate objective gives stable on-policy updates without a
replay buffer. Experiment-0 settings (starting points, not tuned): γ=0.995, GAE λ=0.95, clip 0.2,
8 epochs, minibatch 512, batch 8×512 decisions, Adam lr 3e-4 → 3e-5 (linear), entropy 0.01, value coef 0.5,
grad-clip 0.5, target-KL early stop 0.03, advantage normalisation, return-std reward scaling.

## Custom architecture ("GroupResNet", ≈197 k parameters at defaults)
Observation (50) → seven group embeddings (self 7, opp 7, ball 7, geometry 12, context 2, interaction 7,
phase 8; 32 units each, SiLU) → Linear(224→128) → 2 pre-LayerNorm residual blocks → LayerNorm+SiLU.
Separate encoders for actor and critic. Heads: direction logits (9), kick logit (1), effort Beta(α,β),
power Beta(α,β), value (1). Orthogonal init (√2 hidden, 0.01 policy heads, 1.0 value).
Justification: group embeddings inject the observation's semantic structure; residual/LayerNorm blocks
stabilise on-policy training; Beta heads respect [0,1] bounds without clipping; masking removes gradient
from controls the engine ignores. Ablation candidates: `--share-trunk`, `--hidden`, `--n-res-blocks`,
`--activation tanh`.

## Evaluation protocol
RL vs rule-based baseline, 100 matches per agent (multiples of 4: identity × attack direction), shared
seeds ≥ 1 000 000 (training < 100 000), stochastic policy by default, no weight updates. Metrics: W/D/L
with Wilson 95% CI; goal difference with bootstrap CI; shots and goals/shot; possession and attacking-half
share; steals, loose-ball recoveries, turnovers by cause; stamina profile and sprint share; idle, shuttling,
kick-spam and boundary-loop diagnostics. Per-match CSV schema v1 (`evaluation/metrics.py`).

## Original contribution — stamina (to cite engine owner's design)
Effort above 0.5 drains stamina; below 0.5 recovers it; low stamina scales speed and acceleration down
(floor 50%). Evidence to report: stamina mean/end/min and sprint fraction per agent (`evaluate.ipynb` §7).

## Results (TO FILL)
Learning curve, outcome table, behaviour diagnostics, exploit checks, limitations.
