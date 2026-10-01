# Phase 5: Model Comparison Table

Evaluation across all models using identical train/validation/test splits, feature scaling, and evaluation criteria.

| Model | Parameters | Size (MB) | Train Time (s) | Latency Batch64 (ms) | Val Loss | Test Loss | Congestion Level Acc | Congestion Score MAE | Risk Level Acc | Approaching Acc | Approaching ROC-AUC | Approach Threat MAE | Motion State Acc | Maneuver Acc | Infraction Acc | Infraction ROC-AUC | Infraction Type Acc | Aux Next Disp MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed TCN-Transformer Gated Hybrid | 226937 | 0.87 | 278.0 | 17.24 | 6.6147 | 6.8128 | 0.6565 | 28.11 | 0.6873 | 0.7563 | 0.8265 | 20.65 | 0.9576 | 0.98 | 0.9707 | 0.9968 | 0.9446 | 1.45 |
| Simple Baseline (Mean-Pool MLP) | 23737 | 0.09 | 35.0 | 2.33 | 9.4313 | 9.3322 | 0.6839 | 24.21 | 0.6427 | 0.7659 | 0.7777 | 18.31 | 0.4809 | 0.3546 | 0.8733 | 0.8397 | 0.7917 | 1.71 |
| TCN-Only Model | 117113 | 0.45 | 95.0 | 10.64 | 6.4008 | 6.5444 | 0.6569 | 24.12 | 0.7162 | 0.7817 | 0.8243 | 19.21 | 0.8725 | 0.9492 | 0.9746 | 0.9958 | 0.9557 | 1.41 |
| Transformer-Only Model | 125817 | 0.48 | 95.0 | 5.7 | 5.8876 | 6.4181 | 0.6011 | 28.36 | 0.6469 | 0.7624 | 0.8116 | 17.07 | 0.9245 | 0.9888 | 0.9831 | 0.9979 | 0.973 | 1.45 |
| Hybrid No-Gate (Concat Fusion) | 226937 | 0.87 | 95.0 | 12.53 | 6.6963 | 7.2816 | 0.6095 | 41.05 | 0.6831 | 0.7705 | 0.8115 | 26.5 | 0.9199 | 0.9823 | 0.9738 | 0.997 | 0.9557 | 1.58 |
