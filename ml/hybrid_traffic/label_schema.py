"""
Label Schema and Target Specification
======================================
Defines typed schemas, enums, class mappings, and metadata contracts
for multi-task hybrid traffic machine learning.

Enforces strict labeling provenance:
- label_source = "pseudo_rule_based" for all heuristic and rule-derived targets.
- label_source = "human" ONLY if manual annotations exist (verified: NONE currently exist).
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict


class LabelSource(str, Enum):
    """Provenance indicator for target labels."""
    PSEUDO_RULE_BASED = "pseudo_rule_based"
    HUMAN = "human"


class TaskCategory(str, Enum):
    """Functional categorization of machine learning tasks."""
    CONGESTION = "congestion_classification"
    RISK = "traffic_risk_prediction"
    BEHAVIOR = "vehicle_behavior_classification"
    VIOLATION = "violation_risk_classification"


class CongestionLevel(int, Enum):
    """Zone-level localized traffic congestion level."""
    LOW = 0       # <= 2 vehicles in zone
    MEDIUM = 1    # 3 to 5 vehicles in zone
    HIGH = 2      # >= 6 vehicles in zone


class RiskLevel(int, Enum):
    """Camera-relative approach threat / collision risk."""
    SAFE = 0      # Approach score < 60
    WARNING = 1   # Approach score 60 to 79
    HIGH = 2      # Approach score >= 80


class MotionState(int, Enum):
    """Instantaneous vehicle longitudinal motion behavior."""
    STOPPED = 0       # Speed < 5.0 px/s
    ACCELERATING = 1  # Accel > +5.0 px/s^2
    DECELERATING = 2  # Accel < -5.0 px/s^2
    CRUISING = 3      # Speed >= 5.0 px/s and |accel| <= 5.0 px/s^2


class ManeuverType(int, Enum):
    """Trajectory angular maneuver / steering intent."""
    STATIONARY = 0     # Vehicle is stopped
    TURNING_LEFT = 1   # Angular delta > +0.15 rad (+8.6 deg)
    TURNING_RIGHT = 2  # Angular delta < -0.15 rad (-8.6 deg)
    STRAIGHT = 3       # |Angular delta| <= 0.15 rad


class InfractionType(int, Enum):
    """Systematic traffic rule infraction category."""
    NONE = 0              # Normal compliant behavior
    OVERSPEEDING = 1      # Instantaneous speed > 120 px/s (~95th percentile)
    ILLEGAL_STOPPING = 2  # Stopped in active corridor > 3.0 seconds


@dataclass(frozen=True)
class TargetDefinition:
    """Specification of an individual target label."""
    target_id: str
    task_category: TaskCategory
    problem_type: str  # "multiclass", "binary", or "regression"
    label_source: LabelSource
    num_classes: Optional[int] = None
    class_names: Optional[Dict[int, str]] = None
    units: Optional[str] = None
    description: str = ""
    rule_formula: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["task_category"] = self.task_category.value
        d["label_source"] = self.label_source.value
        return d


# Canonical Target Registry
TARGET_REGISTRY: Dict[str, TargetDefinition] = {
    # -------------------------------------------------------------
    # Task 1: Congestion State Classification & Density Scoring
    # -------------------------------------------------------------
    "target_congestion_level": TargetDefinition(
        target_id="target_congestion_level",
        task_category=TaskCategory.CONGESTION,
        problem_type="multiclass",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=3,
        class_names={0: "LOW", 1: "MEDIUM", 2: "HIGH"},
        description="Zone-level traffic congestion state evaluated at the terminal frame of observation.",
        rule_formula="Evaluated from concurrent vehicle occupancy in active zone: count <= 2 -> LOW; 3..5 -> MEDIUM; >= 6 -> HIGH.",
        limitations="Derived from camera-observed zone occupancy thresholds from traffic_intelligence.py; uncalibrated to macroscopic Highway Capacity Manual (HCM) Level of Service (LOS).",
    ),
    "target_congestion_score": TargetDefinition(
        target_id="target_congestion_score",
        task_category=TaskCategory.CONGESTION,
        problem_type="regression",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        units="normalized_score_0_to_100",
        description="Continuous zone density stress score (0 to 100).",
        rule_formula="min(100.0, (zone_vehicle_count / 10.0) * 100.0).",
        limitations="Linear scaling bounded at 10 concurrent zone vehicles; represents proxy density rather than physical vehicle density per kilometer.",
    ),

    # -------------------------------------------------------------
    # Task 2: Traffic Risk & Collision/Approach Threat Prediction
    # -------------------------------------------------------------
    "target_risk_level": TargetDefinition(
        target_id="target_risk_level",
        task_category=TaskCategory.RISK,
        problem_type="multiclass",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=3,
        class_names={0: "SAFE", 1: "WARNING", 2: "HIGH"},
        description="Camera-relative vehicle approach threat category from approaching_vehicle_analysis.csv.",
        rule_formula="SAFE if score < 60; WARNING if 60 <= score < 80; HIGH if score >= 80, where score combines bounding box growth rate (>8%), downward motion (dy > 5px), and speed.",
        limitations="Optical perspective proxy: vehicles approaching camera view appear with growing bounding boxes; does not measure 3D LiDAR/radar distance.",
    ),
    "target_is_approaching": TargetDefinition(
        target_id="target_is_approaching",
        task_category=TaskCategory.RISK,
        problem_type="binary",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=2,
        class_names={0: "NON_APPROACHING", 1: "APPROACHING"},
        description="Binary indicator whether the vehicle is actively closing distance towards the camera viewpoint.",
        rule_formula="True if approach_score >= 60 (WARNING or HIGH risk); False otherwise.",
        limitations="Perspective dependent: vehicles moving away from or perpendicular to camera angle are classified as 0 even if moving rapidly.",
    ),
    "target_approach_score": TargetDefinition(
        target_id="target_approach_score",
        task_category=TaskCategory.RISK,
        problem_type="regression",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        units="threat_score_0_to_100",
        description="Continuous vehicle approach threat score (0 to 100) from approaching_vehicle_analysis.csv.",
        rule_formula="size_score (up to 50) + movement_score (up to 30) + speed_score (up to 20).",
        limitations="Derived from approaching_vehicle.py heuristics without radar ground truth.",
    ),

    # -------------------------------------------------------------
    # Task 3: Vehicle Motion Behavior & Maneuver Classification
    # -------------------------------------------------------------
    "target_motion_state": TargetDefinition(
        target_id="target_motion_state",
        task_category=TaskCategory.BEHAVIOR,
        problem_type="multiclass",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=4,
        class_names={0: "STOPPED", 1: "ACCELERATING", 2: "DECELERATING", 3: "CRUISING"},
        description="Instantaneous longitudinal dynamic behavior at the sequence horizon.",
        rule_formula="STOPPED if speed < 5 px/s; ACCELERATING if accel > 5 px/s^2; DECELERATING if accel < -5 px/s^2; else CRUISING.",
        limitations="Instantaneous pixel-acceleration at 30 FPS exhibits micro-fluctuations; CRUISING class is sparse due to dense intersection dynamics.",
    ),
    "target_maneuver_type": TargetDefinition(
        target_id="target_maneuver_type",
        task_category=TaskCategory.BEHAVIOR,
        problem_type="multiclass",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=4,
        class_names={0: "STATIONARY", 1: "TURNING_LEFT", 2: "TURNING_RIGHT", 3: "STRAIGHT"},
        description="Lateral trajectory intent and corridor progression state.",
        rule_formula="STATIONARY if speed < 5 px/s; TURNING_LEFT if heading_change_rad > 0.15 rad; TURNING_RIGHT if heading_change_rad < -0.15 rad; else STRAIGHT.",
        limitations="Based on smoothed 2D heading changes; complex S-curve swerving may fluctuate across window boundaries.",
    ),

    # -------------------------------------------------------------
    # Task 4: Violation & Incident Risk Flagging
    # -------------------------------------------------------------
    "target_has_infraction": TargetDefinition(
        target_id="target_has_infraction",
        task_category=TaskCategory.VIOLATION,
        problem_type="binary",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=2,
        class_names={0: "COMPLIANT", 1: "INFRACTION_DETECTED"},
        description="Systematic rule-based indicator of traffic infraction (overspeeding or illegal stopped obstruction).",
        rule_formula="1 if (speed_image_px_per_sec > 120.0 px/s [~95th percentile]) OR (is_stopped and stopped_duration > 3.0s); 0 otherwise.",
        limitations="Rule-based heuristic; does not represent legally certified police citations.",
    ),
    "target_infraction_type": TargetDefinition(
        target_id="target_infraction_type",
        task_category=TaskCategory.VIOLATION,
        problem_type="multiclass",
        label_source=LabelSource.PSEUDO_RULE_BASED,
        num_classes=3,
        class_names={0: "NONE", 1: "OVERSPEEDING", 2: "ILLEGAL_STOPPING"},
        description="Categorical breakdown of detected rule infraction.",
        rule_formula="1 if speed > 120 px/s; 2 if stopped in travel corridor > 3.0s; else 0.",
        limitations="Severe class imbalance with majority class (NONE) representing ~84% of data.",
    ),
}
