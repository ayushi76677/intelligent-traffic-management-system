# PHASE 9 — SUMO TO MODEL FEATURE COMPATIBILITY MAPPING
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `PHASE9_FEATURE_MAPPING.md`  
**Pipeline Phase:** Phase 9 — SUMO/TraCI Closed-Loop Traffic Intelligence  
**Execution Date:** September 2026  
**Status:** Validated & Formally Mapped  

---

## 1. Executive Summary

This document establishes the explicit, mathematically sound mapping between raw Eclipse SUMO / TraCI simulation telemetry and the frozen TCN-Transformer Gated Hybrid model's 20-feature input representation (`checkpoints/best_model.pth`, scaled via `checkpoints/feature_scaler.joblib`).

### Architectural Rule:
* **Zero Guesswork:** Every feature is mapped from explicit TraCI vehicle or lane telemetry.
* **Strict Feature Ordering:** Features must strictly follow the ordering defined in `EXACT_FEATURE_NAMES` (`ml/hybrid_traffic/feature_adapter.py`).
* **No Silent Substitution:** If a camera-specific artifact (e.g. 2D image pixel bounding box) is projected from SUMO's metric coordinates, the transformation is mathematically grounded via standard orthographic/perspective camera projection of the intersection network.

---

## 2. Explicit 20-Feature Mapping Table

| Model Input Index | Feature Name | TraCI / SUMO Variable Source | Mathematical Transformation / Formulation | Units | Scaler Reference Mean $\pm$ Scale |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **0** | `center_x` | `traci.vehicle.getPosition(veh_id)[0]` | Projected image plane center $x$: $u = x_{\text{net}} \times \frac{W_{\text{img}}}{L_{\text{net}}} = x \times \frac{1920}{200}$ | pixels | $1055.15 \pm 554.40$ |
| **1** | `center_y` | `traci.vehicle.getPosition(veh_id)[1]` | Projected image plane center $y$: $v = (200 - y_{\text{net}}) \times \frac{H_{\text{img}}}{L_{\text{net}}}$ (Y-inverted) | pixels | $556.76 \pm 95.09$ |
| **2** | `delta_x` | $u_t - u_{t-1}$ | Single-step horizontal displacement on image plane | pixels | $-0.68 \pm 10.78$ |
| **3** | `delta_y` | $v_t - v_{t-1}$ | Single-step vertical displacement on image plane | pixels | $-0.24 \pm 3.77$ |
| **4** | `displacement_image` | $\sqrt{\Delta u^2 + \Delta v^2}$ | 2D Euclidean image displacement magnitude | pixels | $3.82 \pm 10.78$ |
| **5** | `speed_image_px_per_sec` | $\frac{\text{displacement\_image}}{\Delta t}$ | Projected image speed ($30 \text{ FPS} \times \text{displacement}$) | px/s | $98.42 \pm 260.71$ |
| **6** | `accel_image_px_per_sec2`| $\frac{v_{\text{img}, t} - v_{\text{img}, t-1}}{\Delta t}$ | Projected image acceleration rate | px/$\text{s}^2$ | $-28.18 \pm 9297.77$ |
| **7** | `heading_rad` | `traci.vehicle.getAngle(veh_id)` | SUMO heading in degrees converted to radians: $\theta_{\text{rad}} = \theta_{\text{deg}} \times \frac{\pi}{180} - \pi$ | radians | $-0.18 \pm 1.97$ |
| **8** | `heading_change_rad` | $\theta_{\text{rad}, t} - \theta_{\text{rad}, t-1}$ | Angular yaw velocity across consecutive timesteps wrapped to $[-\pi, \pi]$ | rad/step | $-0.01 \pm 0.88$ |
| **9** | `ground_x` | `traci.vehicle.getPosition(veh_id)[0]` | Metric real-world horizontal position in SUMO network | meters | $460.56 \pm 477.98$ |
| **10** | `ground_y` | `traci.vehicle.getPosition(veh_id)[1]` | Metric real-world vertical position in SUMO network | meters | $132.06 \pm 176.54$ |
| **11** | `ground_delta_x` | $X_t - X_{t-1}$ | Metric displacement in X direction | meters | $-0.71 \pm 8.74$ |
| **12** | `ground_delta_y` | $Y_t - Y_{t-1}$ | Metric displacement in Y direction | meters | $-0.47 \pm 6.62$ |
| **13** | `displacement_ground` | $\sqrt{\Delta X^2 + \Delta Y^2}$ | True metric Euclidean ground displacement | meters | $4.07 \pm 10.22$ |
| **14** | `ground_speed_norm_per_sec`| `traci.vehicle.getSpeed(veh_id)` | True metric longitudinal speed from vehicle speedometer | m/s | $104.57 \pm 245.71$ |
| **15** | `box_width` | `traci.vehicle.getWidth(veh_id)` | Vehicle physical width projected to image canvas: $W_{\text{px}} = W_{\text{veh}} \times \frac{1920}{200}$ | pixels | $153.32 \pm 82.34$ |
| **16** | `box_height` | `traci.vehicle.getLength(veh_id)` | Vehicle physical length projected to image canvas: $H_{\text{px}} = L_{\text{veh}} \times \frac{1080}{200}$ | pixels | $95.87 \pm 49.62$ |
| **17** | `box_area` | $W_{\text{px}} \times H_{\text{px}}$ | Bounding footprint pixel area | $\text{px}^2$ | $18154.84 \pm 21121.41$ |
| **18** | `stopped_duration` | `traci.vehicle.getWaitingTime(veh_id)` | Duration vehicle has been stationary ($v < 0.1 \text{ m/s}$), normalized | seconds | $0.024 \pm 0.063$ |
| **19** | `speed_variance_rolling5` | Rolling window sample variance | Variance of the last 5 observations of projected speed: $\sigma^2_5(v_{\text{img}})$ | $(\text{px/s})^2$ | $60502.70 \pm 189448.78$ |

---

## 3. Mathematical Consistency & Normalization Protocol

1. **Exact Feature Vector Order:**
   Every live TraCI timestep constructs an unscaled 20-dimensional vector:
   $$\mathbf{x}_t = [f_0, f_1, \dots, f_{19}]^T \in \mathbb{R}^{20}$$
2. **StandardScaler Transform:**
   The vector is standardized strictly using the frozen training parameters:
   $$\hat{\mathbf{x}}_t = \frac{\mathbf{x}_t - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
   where $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are the fitted mean and scale arrays stored in `checkpoints/feature_scaler.joblib`.
3. **Temporal Sequence Tensor:**
   For each tracked vehicle in the simulation, a FIFO queue of length $T=20$ is maintained:
   $$\mathbf{X}_{\text{seq}} = [\hat{\mathbf{x}}_{t-19}, \dots, \hat{\mathbf{x}}_{t-1}, \hat{\mathbf{x}}_t] \in \mathbb{R}^{1 \times 20 \times 20}$$
   If a vehicle has fewer than 20 observations ($t < 20$), its status is flagged as `WARMING_UP` and inference is deferred until sufficient temporal history is accumulated. No synthetic padding is introduced.
