# MODEL CARD: Proposed TCN-Transformer Gated Hybrid Architecture
**Model Name:** Proposed TCN-Transformer Gated Hybrid Architecture  
**Document:** `MODEL_CARD.md`  
**Version:** 1.0 (Phase 5 Checkpoint)  
**Checkpoint Path:** [`checkpoints/best_model.pth`](file:///c:/Users/ayush/Emerge%20Root00/checkpoints/best_model.pth)  
**Scaler Path:** [`checkpoints/feature_scaler.joblib`](file:///c:/Users/ayush/Emerge%20Root00/checkpoints/feature_scaler.joblib)  
**Framework:** PyTorch 2.6.0  
**License:** Project Research & Demonstration Use  

---

## 1. Model Details

### 1.1 Architecture Description
The **Proposed TCN-Transformer Gated Hybrid Architecture** combines temporal convolutional receptive fields with multi-head self-attention mechanisms to predict microscopic vehicle states, corridor-level congestion, and safety risks from rolling trajectory sequences:
* **Temporal Convolutional Network (TCN) Branch:** 3 dilated causal residual blocks with kernel size $k=3$ and dilation rates $d \in \{1, 2, 4\}$ capturing local acceleration and deceleration gradients.
* **Transformer Branch:** 2-layer Multi-Head Self-Attention Transformer encoder ($d_{\text{model}}=64$, $n_{\text{heads}}=4$, $d_{\text{ff}}=128$, dropout=0.1) modeling long-range inter-timestep dependencies.
* **Gated Fusion Mechanism:** Learned sigmoid gating unit:
  $$\mathbf{z} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{TCN}}; \mathbf{h}_{\text{Trans}}] + \mathbf{b}_g)$$
  $$\mathbf{h}_{\text{fused}} = \mathbf{z} \odot \mathbf{h}_{\text{TCN}} + (1 - \mathbf{z}) \odot \mathbf{h}_{\text{Trans}}$$
  allowing dynamic weighting between high-frequency kinematic impulses and persistent trends.
* **Multi-Task Decoders:** 10 task heads sharing the fused representation $\mathbf{h}_{\text{fused}}$:
  1. `target_congestion_level`: 3-class classification (LOW, MEDIUM, HIGH)
  2. `target_congestion_score`: Continuous regression ($10.0 - 100.0$)
  3. `target_risk_level`: 3-class classification (SAFE, WARNING, HIGH)
  4. `target_is_approaching`: Binary classification (True/False)
  5. `target_approach_score`: Continuous threat score ($0.0 - 100.0$)
  6. `target_motion_state`: 4-class classification (CRUISING, ACCELERATING, DECELERATING, STOPPED)
  7. `target_maneuver_type`: 4-class classification (LANE_KEEP, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, TURNING)
  8. `target_has_infraction`: Binary classification (COMPLIANT, INFRACTION)
  9. `target_infraction_type`: 3-class classification (NONE, SPEEDING, WRONG_WAY)
  10. `aux_next_displacement`: 2D auxiliary displacement regression $(\Delta x, \Delta y)$

### 1.2 Model Parameters & Dimensions
* **Trainable Parameters:** 226,937
* **Disk Checkpoint Size:** 0.87 MB
* **Input Feature Dimensions:** 20 kinematic features
* **Temporal Sequence Window:** 20 discrete timesteps ($T=20$)
* **Input Tensor Shape:** `[Batch_Size, 20, 20]`

---

## 2. Input Features & Preprocessing

The model requires 20 standardized input features per timestep:
1. `center_x`: Bounding box center X coordinate
2. `center_y`: Bounding box center Y coordinate
3. `delta_x`: Single-step horizontal displacement
4. `delta_y`: Single-step vertical displacement
5. `displacement_image`: 2D Euclidean image displacement
6. `speed_image_px_per_sec`: Instantaneous pixel speed
7. `accel_image_px_per_sec2`: Instantaneous pixel acceleration
8. `heading_rad`: Trajectory heading angle in radians
9. `heading_change_rad`: Angular yaw velocity
10. `ground_x`: Metric horizontal ground coordinate
11. `ground_y`: Metric vertical ground coordinate
12. `ground_delta_x`: Metric horizontal displacement
13. `ground_delta_y`: Metric vertical displacement
14. `displacement_ground`: Metric ground displacement
15. `ground_speed_norm_per_sec`: Metric speed ($m/s$)
16. `box_width`: Bounding box width
17. `box_height`: Bounding box height
18. `box_area`: Bounding box area
19. `stopped_duration`: Cumulative duration stationary ($v < 0.1 \text{ m/s}$)
20. `speed_variance_rolling5`: Sample speed variance over rolling 5 steps

**Feature Scaling:** Standardized strictly with `checkpoints/feature_scaler.joblib` using training dataset population means and variances.

---

## 3. Training & Evaluation Methodology

* **Dataset Source:** Pre-indexed trajectory telemetry from the Vijay Nagar Indore corridor (1,530 annotated frames, 34,679 detection points across 697 unique vehicle tracks).
* **Dataset Splits:** 70% Train, 15% Validation, 15% Test partitioned chronologically by track ID to prevent data leakage.
* **Optimization:** AdamW ($\text{lr}=10^{-3}$, weight decay=$10^{-4}$), ReduceLROnPlateau scheduler, early stopping with patience=10.
* **Loss Function:** Weighted multi-task loss combining Cross-Entropy (with class frequency weighting for risk and infractions) and Smooth L1 Loss.
* **Best Epoch:** Epoch 7 (Validation Loss: 6.6147).

### 3.1 Measured Test Set Performance
* **Motion State Accuracy:** 95.76%
* **Maneuver Classification Accuracy:** 98.00%
* **Infraction Detection Accuracy:** 97.07% (ROC-AUC: 0.9968)
* **Infraction Type Accuracy:** 94.46%
* **Approaching Vehicle Accuracy:** 75.63% (ROC-AUC: 0.8265)
* **Risk Level Accuracy:** 68.73%
* **Congestion Level Accuracy:** 65.65% (Continuous Score MAE: 28.11)
* **Auxiliary Next Displacement MAE:** 1.45 pixels

---

## 4. Intended Use & Deployment Scope

* **Primary Use Case:** Real-time spatial-temporal traffic intelligence, multi-horizon bottleneck detection, and risk scoring in intelligent traffic management systems.
* **Operational Environments:**
  1. **Authority Mode:** Visualizing vehicle risk distributions, detecting traffic rule infractions, and monitoring zone-level congestion.
  2. **Driver Mode:** Providing look-ahead bottleneck warnings, proximity hazard alerts, and routing delay estimations.
  3. **SUMO Co-Simulation:** Guiding autonomous speed harmonization (Variable Speed Limits) and dynamic rerouting in microscopic traffic simulations.

---

## 5. Technical Limitations & Known Risks

1. **Camera Calibration Dependency:** Bounding-box-to-metric feature projection depends on accurate perspective homography matrices (`perspective_matrix.npy`). Miscalibrated camera angles will introduce bias into metric displacement and velocity features.
2. **Warm-up Latency:** The model requires a minimum of 20 consecutive observations ($T=20$) before issuing valid predictions. Observations $t < 20$ remain in `WARMING_UP` state.
3. **Distribution Shift:** Trained primarily on urban intersection camera kinematics under day-lighting conditions. Performance may degrade under severe night-time glare, heavy rain occlusion, or unconventional camera pitch angles without recalibration.
4. **Simulation vs. Real-World Separation:** When used with SUMO, the model observes simulated vehicle kinematics. Simulated control interventions must never be confused with real-world road actuation.
