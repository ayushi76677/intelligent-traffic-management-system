# PHASE 9 — SUMO / TraCI CLOSED-LOOP ARCHITECTURE SPECIFICATION
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `PHASE9_CLOSED_LOOP_ARCHITECTURE.md`  
**Pipeline Phase:** Phase 9 — SUMO/TraCI Closed-Loop Traffic Intelligence  
**Execution Date:** September 2026  
**Status:** Implemented & Verified (17/17 Tests Passing)  

---

## 1. System Overview & Objective

The primary objective of Phase 9 is to establish an autonomous closed-loop traffic control system. Eclipse SUMO (1.27.1) serves as the microscopic traffic simulation environment, while our trained **Proposed TCN-Transformer Gated Hybrid Architecture** (`checkpoints/best_model.pth`) and **Traffic Intelligence Engine** observe the network, predict evolving congestion bottlenecks and safety risks, evaluate a decoupled control policy, and actuate validated TraCI interventions back into SUMO.

```
       ┌────────────────────────────────────────────────────────┐
       │                 Eclipse SUMO (v1.27.1)                 │
       │       Microscopic Simulation Environment (600s)        │
       └───────────┬────────────────────────────────▲───────────┘
                   │ State Query                    │ Validated Action
                   │ (Positions, Speeds, Lanes)     │ (VSL, Reroute)
                   ▼                                │
       ┌───────────────────────┐        ┌───────────┴───────────┐
       │     TraciBridge       │        │  Control Policy Layer │
       │ (Socket Communication)│        │   (Safety Validation) │
       └───────────┬───────────┘        └───────────▲───────────┘
                   │ Raw TraCI Observation          │ Policy Decision
                   ▼                                │
       ┌───────────────────────┐        ┌───────────┴───────────┐
       │  TrafficStateAdapter  │        │   TrafficIntelligence │
       │ (20 Features & FIFO)  │        │         Engine        │
       └───────────┬───────────┘        └───────────▲───────────┘
                   │ Active 20x20 Tensor            │ Predictions & Threat
                   ▼                                │
       ┌────────────────────────────────────────────┴───────────┐
       │   TCN-Transformer Gated Hybrid Architecture (PyTorch)  │
       │   Multi-Task Predictions: Congestion, Risk, Approach   │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Core Integration Modules (`simulation/`)

### 2.1 `simulation/traci_bridge.py`
* **Role:** Manages the low-level socket lifecycle with the Eclipse SUMO binary (`sumo.exe`).
* **Lifecycle Management:**
  * Auto-locates the SUMO binary using `sumolib.checkBinary("sumo")` and system PATH resolution.
  * Launches headless co-simulation: `sumo -c simulation.sumocfg --no-step-log --no-warnings --start`.
  * Guarantees socket cleanup and connection termination upon completion or exception.
* **Telemetry Extraction:**
  * Extracts per-step vehicle telemetry: ID, class, $(x, y)$ coordinate position, speed ($m/s$), heading angle ($^{\circ}$), edge ID, lane ID, lane position, waiting time ($s$), and route edges.
  * Tracks global departures, arrivals, cumulative travel times, and network edge aggregations.
* **Actuation Methods:**
  * `set_edge_max_speed(edge_id, speed_mps)`: Variable speed limit (VSL) speed harmonization.
  * `reroute_vehicle(vehicle_id)`: TraCI-native dynamic rerouting using updated network travel times.

### 2.2 `simulation/traffic_state_adapter.py`
* **Role:** Transforms raw SUMO spatial observations into the exact 20-feature input representation expected by the trained hybrid model.
* **20-Feature Alignment:**
  * Preserves exact 1:1 index alignment with Phase 5/6: `center_x`, `center_y`, `delta_x`, `delta_y`, `displacement_image`, `speed_image_px_per_sec`, `accel_image_px_per_sec2`, `heading_rad`, `heading_change_rad`, `ground_x`, `ground_y`, `ground_delta_x`, `ground_delta_y`, `displacement_ground`, `ground_speed_norm_per_sec`, `box_width`, `box_height`, `box_area`, `stopped_duration`, `speed_variance_rolling5`.
* **Temporal Sequence Buffer:**
  * Implements a 20-step rolling FIFO buffer per active vehicle.
  * Strict Warm-up Management:
    * Observations 1 to 19: `status = WARMING_UP`, `is_ready = False`, `sequence = None`.
    * Observation 20+: `status = ACTIVE`, `is_ready = True`, returns standardized `(20, 20)` float32 matrix.
    * No artificial or zero-padded sequences are fabricated to bypass warm-up.

### 2.3 `simulation/control_policy.py`
* **Role:** Independent decision layer decoupled from raw ML predictions. The ML model *never* issues arbitrary TraCI commands directly.
* **Strict 3-Way Telemetry Segregation:**
  * **A. Measured:** Ground-truth values directly from SUMO (`active_vehicles`, `mean_speed_kmh`, `total_waiting_time_seconds`, `queue_vehicles`).
  * **B. Predicted:** Direct multi-task inferences from the hybrid model (`predicted_vehicle_count`, `average_congestion_score`, `high_congestion_vehicles`, `approaching_threats`).
  * **C. Derived:** Analytical syntheses (`congestion_index`, `network_status`, `throughput_vehicles_per_min`).
* **Safety Rules & Constraints:**
  * **Allowed Action Whitelist:** Only actions in `ALLOWED_ACTIONS` (`VARIABLE_SPEED_LIMIT`, `RESET_SPEED_LIMIT`, `REROUTE_VEHICLE`) are permitted.
  * **Physical Feasibility Bounds:** Speed limit actuations strictly constrained to $[6.0, 13.9]$ m/s ($[21.6, 50.0]$ km/h).
  * **Anti-Chattering Hysteresis:** Edge actuations enforce a minimum cooldown timer (default: 10 seconds).
  * **Confidence Gating:** Actions are suppressed if model inference confidence $< 0.50$.

### 2.4 `simulation/sumo_controller.py`
* **Role:** High-level coordinator linking Bridge, Adapter, Hybrid Predictor, and Control Policy.
* **Batched PyTorch Inference:** Batches all ready vehicle sequence tensors into a single `[N, 20, 20]` tensor for GPU/CPU acceleration.
* **Sub-Millisecond Profiling:** Measures exact per-step elapsed time across:
  * TraCI extraction latency ($ms$)
  * Feature generation latency ($ms$)
  * Model inference latency ($ms$)
  * Intelligence engine calculation latency ($ms$)
  * Control policy execution latency ($ms$)
  * Total end-to-end iteration latency ($ms$)

---

## 3. High-Level Engine & API Surface (`sumo_simulation_engine.py` & `app.py`)

All simulation services are exposed through REST endpoints tagged with `"mode": "SIMULATION DATA"` to prevent confusion with real-world CCTV feeds:

| Endpoint | Method | Functionality | Response Mode |
| :--- | :---: | :--- | :---: |
| **`/api/simulation/status`** | GET / POST | Retrieves status or dispatches actions (`start`, `pause`, `stop`, `reset`, `toggle_control`) | `SIMULATION DATA` |
| **`/api/simulation/traffic`** | GET | Provides live Measured, Predicted, and Derived telemetry | `SIMULATION DATA` |
| **`/api/simulation/control`** | GET / POST | Inspects or adjusts policy parameters (thresholds, cooldowns) | `SIMULATION DATA` |
| **`/api/simulation/metrics`** | GET | Returns baseline vs. controlled empirical comparison results | `SIMULATION DATA` |

---

## 4. Safety Architecture & Failure Handling

```
                      TraCI Observation
                             │
                             ▼
                    Model Checkpoint OK?
                   /                    \
             YES  /                      \ NO / Incompatible
                 ▼                        ▼
       TCN-Transformer Inference    Fallback to Heuristic Mode
                 │                        │
                 ▼                        ▼
       Confidence >= 0.50?          Measured SUMO State
         /              \                 │
   YES  /                \ NO             │
       ▼                  ▼               │
  ML Threat Score   Suppress Action       │
       │                  │               │
       └──────────┬───────┴───────────────┘
                  ▼
         Is Action Whitelisted?
         /                    \
   YES  /                      \ NO
       ▼                        ▼
  Physical Bounds Clamped    Reject Action (Logged)
  & Cooldown Respected?
       /              \
 YES  /                \ NO
     ▼                  ▼
Dispatch TraCI      Suppress Action
```

1. **Model Unavailable / Missing Checkpoint:** Controller automatically falls back to heuristic congestion estimation based on instantaneous SUMO occupancy, preventing crash or halt.
2. **TraCI Connection Failure:** Raises structured `RuntimeError` rather than corrupting memory or hanging.
3. **Invalid Prediction Rejection:** Malformed or out-of-range model predictions default safely to conservative cruising states.
4. **Clean Termination:** `TraciBridge.close()` reliably releases TraCI socket handles even on unhandled simulation exceptions.
