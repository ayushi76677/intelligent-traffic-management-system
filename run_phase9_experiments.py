"""
Phase 9 Simulation Runner: Baseline vs Controlled Scenarios
===========================================================
Executes identical 600-second simulation runs under:
1. Baseline Scenario (uncontrolled, fixed free-flow speed limits)
2. Controlled Scenario (active TCN-Transformer closed-loop traffic control policy)

Saves empirical telemetry to:
- results/sumo_baseline_results.json
- results/sumo_controlled_results.json
"""

import os
import sys
import json
import time

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simulation.sumo_controller import SumoClosedLoopController


def run_experiments(duration: int = 600):
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    baseline_path = os.path.join(BASE_DIR, "results", "sumo_baseline_results.json")
    controlled_path = os.path.join(BASE_DIR, "results", "sumo_controlled_results.json")

    print(f"\n=================================================================")
    print(f"PHASE 9 EXPERIMENTS: SUMO CLOSED-LOOP EVALUATION (Horizon={duration}s)")
    print(f"=================================================================\n")

    # 1. Run Baseline Scenario
    print(">>> [1/2] Executing Baseline Simulation (Control Disabled)...")
    t0 = time.time()
    controller_base = SumoClosedLoopController(
        config_file="sumo/simulation.sumocfg",
        workspace_dir=BASE_DIR,
        control_enabled=False
    )
    baseline_results = controller_base.run_simulation(duration=duration, control_enabled=False)
    elapsed_base = time.time() - t0

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Baseline complete in {elapsed_base:.2f}s wall-clock.")
    print(f"Saved: {baseline_path}")
    print(f"Baseline Metrics: {json.dumps(baseline_results['traffic_metrics'], indent=2)}\n")

    # Short pause to ensure clean socket release
    time.sleep(2.0)

    # 2. Run Controlled Scenario
    print(">>> [2/2] Executing Controlled Simulation (Closed-Loop Policy Enabled)...")
    t1 = time.time()
    controller_ctrl = SumoClosedLoopController(
        config_file="sumo/simulation.sumocfg",
        workspace_dir=BASE_DIR,
        control_enabled=True
    )
    controlled_results = controller_ctrl.run_simulation(duration=duration, control_enabled=True)
    elapsed_ctrl = time.time() - t1

    with open(controlled_path, "w", encoding="utf-8") as f:
        json.dump(controlled_results, f, indent=2)
    print(f"Controlled run complete in {elapsed_ctrl:.2f}s wall-clock.")
    print(f"Saved: {controlled_path}")
    print(f"Controlled Metrics: {json.dumps(controlled_results['traffic_metrics'], indent=2)}\n")

    # 3. Print Objective Comparison
    b_met = baseline_results["traffic_metrics"]
    c_met = controlled_results["traffic_metrics"]

    print("=================================================================")
    print("OBJECTIVE SIMULATION COMPARISON RESULTS")
    print("=================================================================")
    print(f"{'Metric':<35} | {'Baseline':<12} | {'Controlled':<12} | {'Delta':<10}")
    print("-" * 75)

    def print_row(name, b_val, c_val, unit="", lower_better=True):
        delta = c_val - b_val
        pct = (delta / max(1e-4, abs(b_val))) * 100.0 if b_val != 0 else 0.0
        sign = "+" if delta > 0 else ""
        print(f"{name:<35} | {b_val:>10.2f}{unit:<2} | {c_val:>10.2f}{unit:<2} | {sign}{pct:>6.2f}%")

    print_row("Average Travel Time", b_met["average_travel_time_seconds"], c_met["average_travel_time_seconds"], "s", True)
    print_row("Total Travel Time", b_met["total_travel_time_seconds"], c_met["total_travel_time_seconds"], "s", True)
    print_row("Average Speed", b_met["average_speed_kmh"], c_met["average_speed_kmh"], "km/h", False)
    print_row("Average Waiting Time", b_met["average_waiting_time_seconds"], c_met["average_waiting_time_seconds"], "s", True)
    print_row("Final Cumulative Waiting Time", b_met["final_cumulative_waiting_time"], c_met["final_cumulative_waiting_time"], "s", True)
    print_row("Average Queue Length", b_met["average_queue_length"], c_met["average_queue_length"], "veh", True)
    print_row("Peak Queue Length", float(b_met["max_queue_length"]), float(c_met["max_queue_length"]), "veh", True)
    print_row("Throughput", b_met["throughput_vehs_per_hour"], c_met["throughput_vehs_per_hour"], "vph", False)
    print_row("Completed Vehicles", float(b_met["total_vehicles_completed"]), float(c_met["total_vehicles_completed"]), "", False)
    print_row("Average Congestion Index", b_met["average_congestion_index"], c_met["average_congestion_index"], "", True)
    print("=" * 75)
    print(f"Total Control Interventions Executed: {c_met['total_control_actions_executed']}")
    print(f"Baseline Effective Simulation FPS: {baseline_results['latency_metrics']['effective_simulation_fps']}")
    print(f"Controlled Effective Simulation FPS: {controlled_results['latency_metrics']['effective_simulation_fps']}")
    print("=================================================================\n")


if __name__ == "__main__":
    run_experiments(duration=600)
