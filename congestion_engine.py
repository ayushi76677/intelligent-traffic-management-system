import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

ZONE_FILE = "zone_traffic_analysis.csv"
FLOW_FILE = "zone_traffic_flow.csv"

OUTPUT_FILE = "congestion_analysis.csv"

# These are initial project thresholds.
# They are NOT official traffic standards.

MAX_ZONE_VEHICLES = 200
MAX_INTERVAL_VEHICLES = 150

# Weight assigned to each signal
WEIGHT_DENSITY = 0.45
WEIGHT_FLOW = 0.35
WEIGHT_VARIATION = 0.20


# ============================================================
# LOAD DATA
# ============================================================

zone = pd.read_csv(ZONE_FILE)
flow = pd.read_csv(FLOW_FILE)

print("Zone data loaded.")
print("Flow data loaded.")


# ============================================================
# BASIC VALUES
# ============================================================

total_vehicles = int(
    zone["total_vehicles"].iloc[0]
)

average_speed = float(
    zone["average_clean_speed"].iloc[0]
)

traffic_level = str(
    zone["traffic_level"].iloc[0]
)


# ============================================================
# 1. VEHICLE DENSITY SCORE
# ============================================================

density_score = (
    total_vehicles /
    MAX_ZONE_VEHICLES
) * 100

density_score = np.clip(
    density_score,
    0,
    100
)


# ============================================================
# 2. FLOW SCORE
# ============================================================

max_flow = float(
    flow["unique_vehicles"].max()
)

average_flow = float(
    flow["unique_vehicles"].mean()
)

flow_score = (
    max_flow /
    MAX_INTERVAL_VEHICLES
) * 100

flow_score = np.clip(
    flow_score,
    0,
    100
)


# ============================================================
# 3. TRAFFIC VARIATION SCORE
# ============================================================
#
# Large changes in vehicle presence between intervals can
# indicate unstable traffic demand.
# ============================================================

flow_values = (
    flow["unique_vehicles"]
    .astype(float)
    .values
)

if len(flow_values) > 1:

    differences = np.abs(
        np.diff(flow_values)
    )

    average_change = differences.mean()

else:

    average_change = 0


variation_score = (
    average_change /
    MAX_INTERVAL_VEHICLES
) * 100

variation_score = np.clip(
    variation_score,
    0,
    100
)


# ============================================================
# 4. FINAL CONGESTION SCORE
# ============================================================

congestion_score = (
    density_score * WEIGHT_DENSITY +
    flow_score * WEIGHT_FLOW +
    variation_score * WEIGHT_VARIATION
)

congestion_score = np.clip(
    congestion_score,
    0,
    100
)


# ============================================================
# 5. CONGESTION CLASSIFICATION
# ============================================================

if congestion_score < 25:

    congestion_level = "LOW"

elif congestion_score < 50:

    congestion_level = "MODERATE"

elif congestion_score < 75:

    congestion_level = "HIGH"

else:

    congestion_level = "SEVERE"


# ============================================================
# 6. FIND PEAK INTERVAL
# ============================================================

peak_index = flow[
    "unique_vehicles"
].idxmax()

peak_interval = flow.loc[
    peak_index,
    "10_second_interval"
]

peak_vehicles = flow.loc[
    peak_index,
    "unique_vehicles"
]


# ============================================================
# 7. CREATE OUTPUT
# ============================================================

result = pd.DataFrame([
    {
        "zone": "ZONE_1",

        "total_vehicles":
            total_vehicles,

        "maximum_interval_vehicles":
            int(max_flow),

        "average_interval_vehicles":
            round(
                average_flow,
                2
            ),

        "density_score":
            round(
                density_score,
                2
            ),

        "flow_score":
            round(
                flow_score,
                2
            ),

        "variation_score":
            round(
                variation_score,
                2
            ),

        "congestion_score":
            round(
                congestion_score,
                2
            ),

        "congestion_level":
            congestion_level,

        "original_traffic_level":
            traffic_level,

        "peak_interval":
            int(peak_interval),

        "peak_interval_vehicles":
            int(peak_vehicles)
    }
])


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==========================================")
print("CONGESTION ENGINE")
print("==========================================")

print(
    result.to_string(index=False)
)

print("\nCreated:")
print(OUTPUT_FILE)

print("\nDone.")