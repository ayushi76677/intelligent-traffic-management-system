"""
Simulation Integration Package (Phase 9)
========================================
Closed-loop traffic intelligence system coupling Eclipse SUMO 1.27.1 / TraCI
with the trained TCN-Transformer Gated Hybrid model and Traffic Intelligence Engine.
"""

from simulation.traci_bridge import TraciBridge
from simulation.traffic_state_adapter import TrafficStateAdapter
from simulation.control_policy import ClosedLoopControlPolicy
from simulation.sumo_controller import SumoClosedLoopController

__all__ = [
    "TraciBridge",
    "TrafficStateAdapter",
    "ClosedLoopControlPolicy",
    "SumoClosedLoopController",
]
