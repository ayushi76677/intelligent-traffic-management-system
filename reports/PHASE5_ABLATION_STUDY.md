# Phase 5: Architecture Ablation Study

Isolating the contribution of the TCN branch, Transformer branch, and Gated Fusion mechanism:

| Architecture Variant | Parameters | Test Total Loss | Latency (ms) | Maneuver Acc | Motion State Acc | Infraction Acc | Approaching Acc | Congestion Acc | Next Disp MAE (px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed TCN-Transformer Gated Hybrid | 226937 | 6.8128 | 17.24 | 0.98 | 0.9576 | 0.9707 | 0.7563 | 0.6565 | 1.45 |
| TCN-Only Model | 117113 | 6.5444 | 10.64 | 0.9492 | 0.8725 | 0.9746 | 0.7817 | 0.6569 | 1.41 |
| Transformer-Only Model | 125817 | 6.4181 | 5.7 | 0.9888 | 0.9245 | 0.9831 | 0.7624 | 0.6011 | 1.45 |
| Hybrid No-Gate (Concat Fusion) | 226937 | 7.2816 | 12.53 | 0.9823 | 0.9199 | 0.9738 | 0.7705 | 0.6095 | 1.58 |
