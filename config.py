
from pathlib import Path


# ### ### ### EDIT THIS ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/leowelma/Downloads/UAH-DRIVESET-v1")  # <-- change this
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# driver metadata (from Romera et al. 2016)
DRIVER_META = {
    "D1": {"age": 34, "vehicle": "Citroën C4 (diesel)"},
    "D2": {"age": 20, "vehicle": "Opel Corsa (gasoline)"},
    "D3": {"age": 22, "vehicle": "Citroën C4 (diesel)"},
    "D4": {"age": 27, "vehicle": "Opel Astra (diesel)"},
    "D5": {"age": 30, "vehicle": "Volkswagen Golf (diesel)"},
    "D6": {"age": 46, "vehicle": "Nissan Juke (gasoline)"},}

# raw accelerometer columns
ACCEL_COLS = ["timestamp", "activation", "acc_x", "acc_y",
              "acc_z", "acc_x_kf", "acc_y_kf", "acc_z_kf",
              "roll", "pitch", "yaw", ]

# raw gps columns
GPS_COLS = ["timestamp", "speed_kmh", "latitude", "longitude",
            "altitude", "course", "gps_accuracy",]

# raw sensor channels
IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "roll", "pitch", "yaw"]

# DriveSafe maneuver types (from SEMANTIC_FINAL)
# DriveSafe scores are ruled-based threshold detections that are computed based on the presence of certain maneuvers in the driving data.
# Initial idea was to use them as an illustration for an "external criterion" as described in the Conclusion
# Provides counts but no timestamps, therefore not used for segmentation purposes
MANEUVER_TYPES = ["accelerations", "brakings", "turnings", "lane_weaving",
                  "lane_drifting", "overspeeding","car_following",]

BEHAVIOR_ORDER = ["NORMAL", "DROWSY", "AGGRESSIVE"]
ROAD_ORDER = ["MOTORWAY", "SECONDARY"]
