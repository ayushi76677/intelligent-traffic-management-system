"""
TraCI Bridge for Eclipse SUMO Co-Simulation
============================================
Handles process lifecycle, socket connection, state extraction, and validated
actuation commands for Eclipse SUMO 1.27.1 via the Python TraCI API.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set

logger = logging.getLogger("Simulation.TraciBridge")


class TraciBridge:
    """
    Robust TraCI integration bridge managing SUMO execution, telemetry extraction,
    and safe actuation.
    """

    def __init__(
        self,
        config_file: str = "sumo/simulation.sumocfg",
        workspace_dir: str = ".",
        gui: bool = False,
        step_length: float = 1.0,
        label: str = "emerge_phase9"
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.config_file = os.path.join(self.workspace_dir, config_file) if not os.path.isabs(config_file) else config_file
        self.gui = gui
        self.step_length = step_length
        self.label = label

        self.traci = None
        self.sumolib = None
        self.is_connected = False
        self.status = "DISCONNECTED"
        self.sumo_binary = None
        self.current_step = 0
        self.current_sim_time = 0.0

        # Cumulative telemetry counters
        self.total_departed_count = 0
        self.total_arrived_count = 0
        self.total_collisions_count = 0
        self.seen_vehicles: Set[str] = set()
        self.vehicle_depart_times: Dict[str, float] = {}
        self.completed_travel_times: List[float] = []

        self._initialize_environment()

    def _initialize_environment(self):
        """Locates SUMO binaries and imports traci."""
        try:
            import traci
            import sumolib
            self.traci = traci
            self.sumolib = sumolib
        except ImportError as e:
            logger.error(f"[TraciBridge] Failed to import traci/sumolib: {e}")
            self.status = "DEPENDENCY_ERROR"
            return

        # Locate binary
        bin_name = "sumo-gui" if self.gui else "sumo"
        try:
            self.sumo_binary = self.sumolib.checkBinary(bin_name)
        except Exception:
            # Fallback check in python roaming packages
            user_bin = os.path.expanduser(
                r"~\AppData\Roaming\Python\Python313\site-packages\sumo\bin"
            )
            candidate = os.path.join(user_bin, f"{bin_name}.exe")
            if os.path.exists(candidate):
                self.sumo_binary = candidate
            else:
                self.sumo_binary = None

        if self.sumo_binary and os.path.exists(self.sumo_binary):
            self.status = "INITIALIZED"
            logger.info(f"[TraciBridge] Found SUMO binary at: {self.sumo_binary}")
        else:
            self.status = "BINARY_NOT_FOUND"
            logger.error("[TraciBridge] SUMO binary not located.")

    def start(self, additional_args: Optional[List[str]] = None) -> bool:
        """Starts SUMO and establishes TraCI connection."""
        if not self.traci or not self.sumo_binary:
            logger.error("[TraciBridge] Cannot start: traci or sumo binary missing.")
            self.status = "START_FAILED"
            return False

        if not os.path.exists(self.config_file):
            logger.error(f"[TraciBridge] Config file missing: {self.config_file}")
            self.status = "CONFIG_MISSING"
            return False

        # Close existing connection if active
        if self.is_connected:
            self.close()

        cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--waiting-time-memory", "600",
            "--start", "true",
            "--quit-on-end", "true"
        ]
        if additional_args:
            cmd.extend(additional_args)

        try:
            self.traci.start(cmd, label=self.label)
            self.is_connected = True
            self.status = "RUNNING"
            self.current_step = 0
            self.current_sim_time = self.traci.simulation.getTime()
            self.seen_vehicles.clear()
            self.vehicle_depart_times.clear()
            self.completed_travel_times.clear()
            self.total_departed_count = 0
            self.total_arrived_count = 0
            logger.info(f"[TraciBridge] TraCI connected. Sim time: {self.current_sim_time}s")
            return True
        except Exception as e:
            logger.error(f"[TraciBridge] Connection error: {e}")
            self.is_connected = False
            self.status = f"ERROR: {str(e)}"
            return False

    def step(self) -> Dict[str, Any]:
        """Advances simulation by one timestep and collects raw state."""
        if not self.is_connected:
            raise RuntimeError("Cannot step: TraCI is not connected.")

        try:
            self.traci.simulationStep()
            self.current_step += 1
            self.current_sim_time = self.traci.simulation.getTime()

            # Track departed & arrived vehicles
            departed_now = self.traci.simulation.getDepartedIDList()
            arrived_now = self.traci.simulation.getArrivedIDList()

            for vid in departed_now:
                self.seen_vehicles.add(vid)
                self.vehicle_depart_times[vid] = self.current_sim_time
            self.total_departed_count += len(departed_now)

            for vid in arrived_now:
                dep_t = self.vehicle_depart_times.get(vid, self.current_sim_time)
                travel_time = max(1.0, self.current_sim_time - dep_t)
                self.completed_travel_times.append(float(travel_time))
            self.total_arrived_count += len(arrived_now)

            self.total_collisions_count += self.traci.simulation.getCollidingVehiclesNumber()

            return self.extract_simulation_state()
        except self.traci.FatalTraCIError as e:
            logger.warning(f"[TraciBridge] Simulation terminated or TraCI closed: {e}")
            self.is_connected = False
            self.status = "TERMINATED"
            return {"status": "TERMINATED", "simulation_time": self.current_sim_time, "vehicles": {}}
        except Exception as e:
            logger.error(f"[TraciBridge] Error during step: {e}")
            self.is_connected = False
            self.status = f"ERROR: {str(e)}"
            raise

    def extract_simulation_state(self) -> Dict[str, Any]:
        """Extracts complete raw vehicle, edge, lane, and junction states."""
        if not self.is_connected:
            return {}

        veh_ids = self.traci.vehicle.getIDList()
        vehicles = {}

        for vid in veh_ids:
            try:
                pos = self.traci.vehicle.getPosition(vid)
                spd = self.traci.vehicle.getSpeed(vid)
                ang = self.traci.vehicle.getAngle(vid)
                lane_id = self.traci.vehicle.getLaneID(vid)
                lane_idx = self.traci.vehicle.getLaneIndex(vid)
                road_id = self.traci.vehicle.getRoadID(vid)
                wait_time = self.traci.vehicle.getWaitingTime(vid)
                acc_wait = self.traci.vehicle.getAccumulatedWaitingTime(vid)
                length = self.traci.vehicle.getLength(vid)
                width = self.traci.vehicle.getWidth(vid)
                route_id = self.traci.vehicle.getRouteID(vid)
                route_edges = list(self.traci.vehicle.getRoute(vid))
                accel = self.traci.vehicle.getAcceleration(vid)

                vehicles[vid] = {
                    "vehicle_id": vid,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "speed": float(spd),
                    "speed_kmh": float(spd * 3.6),
                    "angle": float(ang),
                    "lane_id": lane_id,
                    "lane_index": int(lane_idx),
                    "edge_id": road_id,
                    "waiting_time": float(wait_time),
                    "accumulated_waiting_time": float(acc_wait),
                    "length": float(length),
                    "width": float(width),
                    "route_id": route_id,
                    "route_edges": route_edges,
                    "acceleration": float(accel),
                    "vehicle_class": self.traci.vehicle.getVehicleClass(vid)
                }
            except Exception:
                continue

        # Edge-level aggregation
        edge_ids = [e for e in self.traci.edge.getIDList() if not e.startswith(":")]
        edges = {}
        for eid in edge_ids:
            try:
                e_vehs = self.traci.edge.getLastStepVehicleIDs(eid)
                e_spd = self.traci.edge.getLastStepMeanSpeed(eid)
                e_occ = self.traci.edge.getLastStepOccupancy(eid)
                e_wait = self.traci.edge.getWaitingTime(eid)
                e_halted = self.traci.edge.getLastStepHaltingNumber(eid)

                edges[eid] = {
                    "edge_id": eid,
                    "vehicle_count": len(e_vehs),
                    "vehicle_ids": list(e_vehs),
                    "mean_speed": float(e_spd),
                    "mean_speed_kmh": float(e_spd * 3.6),
                    "occupancy": float(e_occ),
                    "waiting_time": float(e_wait),
                    "queue_length": int(e_halted)
                }
            except Exception:
                continue

        # Traffic light states (if any exist)
        tls_ids = self.traci.trafficlight.getIDList()
        traffic_lights = {}
        for tid in tls_ids:
            try:
                traffic_lights[tid] = {
                    "tls_id": tid,
                    "phase": self.traci.trafficlight.getPhase(tid),
                    "state": self.traci.trafficlight.getRedYellowGreenState(tid),
                    "phase_duration": self.traci.trafficlight.getPhaseDuration(tid)
                }
            except Exception:
                continue

        return {
            "status": self.status,
            "step": self.current_step,
            "simulation_time": float(self.current_sim_time),
            "active_vehicle_count": len(vehicles),
            "total_departed": self.total_departed_count,
            "total_arrived": self.total_arrived_count,
            "total_collisions": self.total_collisions_count,
            "vehicles": vehicles,
            "edges": edges,
            "traffic_lights": traffic_lights
        }

    # =========================================================================
    # Validated Actuation Primitives
    # =========================================================================

    def set_vehicle_route(self, veh_id: str, new_edges: List[str]) -> bool:
        """Assigns a new valid edge sequence route to a vehicle."""
        if not self.is_connected or veh_id not in self.traci.vehicle.getIDList():
            return False
        try:
            self.traci.vehicle.setRoute(veh_id, new_edges)
            return True
        except Exception as e:
            logger.warning(f"[TraciBridge] Failed to set route for {veh_id}: {e}")
            return False

    def reroute_vehicle(self, veh_id: str) -> bool:
        """Computes and assigns shortest travel-time detour using live edge weights."""
        if not self.is_connected or veh_id not in self.traci.vehicle.getIDList():
            return False
        try:
            self.traci.vehicle.rerouteTraveltime(veh_id)
            return True
        except Exception as e:
            logger.warning(f"[TraciBridge] Reroute failed for {veh_id}: {e}")
            return False

    def set_lane_max_speed(self, lane_id: str, speed_mps: float) -> bool:
        """Sets variable speed limit on a specific lane."""
        if not self.is_connected:
            return False
        try:
            self.traci.lane.setMaxSpeed(lane_id, max(1.0, float(speed_mps)))
            return True
        except Exception as e:
            logger.warning(f"[TraciBridge] Failed to set lane speed on {lane_id}: {e}")
            return False

    def set_edge_max_speed(self, edge_id: str, speed_mps: float) -> bool:
        """Sets variable speed limit across all lanes of an edge."""
        if not self.is_connected:
            return False
        try:
            lane_count = self.traci.edge.getLaneNumber(edge_id)
            for i in range(lane_count):
                lid = f"{edge_id}_{i}"
                self.traci.lane.setMaxSpeed(lid, max(1.0, float(speed_mps)))
            return True
        except Exception as e:
            logger.warning(f"[TraciBridge] Failed to set edge speed on {edge_id}: {e}")
            return False

    def set_traffic_light_phase(self, tls_id: str, phase_idx: int) -> bool:
        """Actuates traffic light to a specified phase index."""
        if not self.is_connected or tls_id not in self.traci.trafficlight.getIDList():
            return False
        try:
            self.traci.trafficlight.setPhase(tls_id, int(phase_idx))
            return True
        except Exception as e:
            logger.warning(f"[TraciBridge] TLS phase set failed for {tls_id}: {e}")
            return False

    def close(self):
        """Safely shuts down TraCI connection without lingering processes."""
        if self.is_connected and self.traci:
            try:
                self.traci.close()
            except Exception:
                pass
        self.is_connected = False
        self.status = "CLOSED"
        logger.info("[TraciBridge] Closed connection cleanly.")
