"""
Shared constants for the project.
Mirror the same folder IDs used in the data collection pipeline.
"""

from pathlib import Path

# PATHS
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
STATIC_DIR = DATA_DIR / "static"
DYNAMIC_DIR = DATA_DIR / "dynamic"


# GOOGLE DRIVE FOLDER IDS
DRIVE_FOLDER = {
    "stations": "1NUOONdY9Z7pssVZPrZio_kcuJpb8a-NQ",
    "lines": "1s0uIugowoVPJp9ebXOXPzZW1utzz7ck_",
    "trains": "1oTKkcW0tCNaSsmEshdZgkOq4n3nM07RI",
    "timetables": "1J_ZPRDahiL68AkWPazAnbtqkzRmv3PXH",
    "journeys": "14s6dncDhUzJ7pXjahfcMzduJHP_9J8nV",
    "weather": "1UzwckGP8K4NMOP5mUnIAu05RAKq2K5Cl",
}


# LINES TO ANALYZE
R1_LINE_ID = "R1"


# DATE FORMAT
DATE_FMT = "%Y_%m_%d"
