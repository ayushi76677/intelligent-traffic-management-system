"""
Phase 9 Closed-Loop Traffic Intelligence Demonstration
======================================================
Reproducible script demonstrating the complete closed-loop cycle:
1. Start SUMO
2. Connect through TraCI
3. Advance simulation
4. Extract traffic state
5. Build features
6. Update temporal buffer
7. Run hybrid model
8. Calculate traffic intelligence
9. Evaluate control policy
10. Apply validated TraCI action
11. Advance SUMO
12. Observe changed traffic state
13. Repeat & Log
"""

import os
import sys
import time
import argparse
import logging

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simulation.sumo_controller import SumoClosedLoopController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase9Demo")


def run_demonstration(num_steps: int = 30):
    print("\n" + "=" * 70)
    print("PHASE 9 — CLOSED-LOOP SUMO/TRACI TRAFFIC INTELLIGENCE DEMONSTRATION")
    print("=" * 70)

    # 1. Instantiate Controller
    logger.info("[Step 1] Initializing SUMO Closed-Loop Controller...")
    controller = SumoClosedLoopController(
        config_file="sumo/simulation.sumocfg",
        workspace_dir=BASE_DIR,
        control_enabled=True
    )

    # 2. Start SUMO and connect via TraCI
    logger.info("[Step 2] Launching Eclipse SUMO 1.27.1 and establishing TraCI handshake...")
    success = controller.start()
    if not success:
        logger.error("Failed to start SUMO / TraCI connection.")
        return False

    logger.info(f"TraCI Connection Verified. Status: {controller.bridge.status}")
    print("-" * 70)

    try:
        for i in range(1, num_steps + 1):
            logger.info(f"\n--- [Cycle {i}/{num_steps}] Executing Closed-Loop Turn ---")

            # 3. Advance simulation & 4. Extract state
            step_record = controller.step()
            sim_time = step_record["simulation_time"]
            act_vehs = step_record["active_vehicles"]
            warming_vehs = step_record["warming_up_vehicles"]
            infer_vehs = step_record["active_inference_vehicles"]
            cong_idx = step_record["congestion_index"]
            actions_cnt = step_record["actions_executed"]
            lat = step_record["latencies_ms"]

            logger.info(f"[Step 3-4] SUMO Advanced. Sim Time: {sim_time}s | Active Vehicles: {act_vehs}")
            logger.info(f"[Step 5-6] Buffer Updated. Warming-up: {warming_vehs} | Active for Inference: {infer_vehs}")
            logger.info(f"[Step 7] Hybrid Inference Latency: {lat['inference']:.2f} ms")
            logger.info(f"[Step 8] Traffic Intelligence: Speed={step_record['mean_speed_kmh']:.1f} km/h | Congestion Index={cong_idx:.3f}")

            # 9-10. Control policy & Validated TraCI action
            if controller.latest_actions:
                for act in controller.latest_actions:
                    logger.info(f"[Step 9-10] Validated Control Action Dispatched -> Type: {act.action_type} | Target: {act.target_id} | Rationale: {act.rationale}")
            else:
                logger.info("[Step 9-10] Control Policy Evaluated: System within safe operational margins. Action: NO_ACTION")

            logger.info(f"[Step 11-12] Cycle Latency: {lat['total']:.2f} ms (Effective Rate: {1000.0/max(1.0, lat['total']):.1f} FPS)")

            time.sleep(0.05)  # Smooth demo pacing

        print("\n" + "=" * 70)
        logger.info("[Step 13] Demonstration Complete. Compiling Summary Telemetry...")
        summary = controller.compile_simulation_results()
        t_met = summary["traffic_metrics"]
        l_met = summary["latency_metrics"]

        print("\nDEMONSTRATION RUN SUMMARY:")
        print(f"- Total Simulation Time: {summary['simulation_duration_seconds']}s ({summary['total_steps_executed']} steps)")
        print(f"- Total Departed Vehicles: {t_met['total_vehicles_departed']}")
        print(f"- Total Completed Vehicles: {t_met['total_vehicles_completed']}")
        print(f"- Average Corridor Speed: {t_met['average_speed_kmh']} km/h")
        print(f"- Actions Executed: {t_met['total_control_actions_executed']}")
        print(f"- Average Step Latency: {l_met['average_total_step_ms']} ms ({l_met['effective_simulation_fps']} FPS)")
        print("=" * 70 + "\n")

    finally:
        logger.info("Closing TraCI connection cleanly...")
        controller.stop()
        logger.info("TraCI Closed.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 9 Closed-Loop Demo")
    parser.add_argument("--steps", type=int, default=25, help="Number of simulation steps to run")
    args = parser.parse_args()
    run_demonstration(num_steps=args.steps)
