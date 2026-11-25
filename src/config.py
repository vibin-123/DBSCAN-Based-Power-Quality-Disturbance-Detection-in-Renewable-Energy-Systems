"""Central configuration: paths, dataset constants and analysis settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"
REPORTS = OUTPUTS / "reports"
RESULTS = OUTPUTS / "results"

# ---- SPARK (KIT) -----------------------------------------------------------
SPARK_DOI = "10.35097/bjdg3m3rg5jv3skk"
SPARK_TAR_URL = ("https://radar.kit.edu/radar-backend/archives/"
                 "bjdg3m3rg5jv3skk/versions/1/content")
SPARK_DIR = RAW / "spark"
SPARK_SMALL_ZIPS = ["EPI_PV.zip", "00meta_data.zip", "01exogenous_data.zip"]
PQ_METER = "EPI_TotalLoad"          # facility point of common coupling meter
PV_METER = "EPI_PV"
YEAR = 2024                          # only year in which PV data exists
PQ_MEASUREMENTS = [
    "U1", "U2", "U3", "U12", "U23", "U31", "Freq",
    "I1", "I2", "I3", "IN",
    "THD_U1", "THD_U2", "THD_U3", "THD_I1", "THD_I2", "THD_I3",
    "P_total", "Q_total", "S_total", "PF_total",
    "U1_h3", "U1_h5", "U1_h7", "U2_h5", "U3_h5",
]
PV_RATED_W = 10_720.0                # 10.72 kWp (SPARK machine metadata)

# ---- SMARD (Bundesnetzagentur) ----------------------------------------------
SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_REGION = "TransnetBW"
SMARD_FILTERS = {"solar_pv": 4068, "wind_onshore": 4067, "grid_load": 410}
SMARD_DIR = RAW / "smard"

# ---- Electrical constants ---------------------------------------------------
U_NOMINAL_LN = 230.0                 # V, EN 50160 nominal phase-to-neutral
F_NOMINAL = 50.0                     # Hz
SAG_PU, SWELL_PU = 0.90, 1.10        # EN 50160 dip / swell thresholds
SAMPLE_SECONDS = 5
WINDOW = "10min"                     # IEC 61000-4-30 aggregation interval
# The meter reports roughly every 6.5 s; mapped to SPARK's 5 s grid this leaves
# every fourth slot empty, so a complete window has ~76.7 % of slots filled
# (about 92 of 120). A window is accepted with >= 60 % filled slots, i.e. at
# least ~78 % of the samples the meter can actually deliver.
MIN_COVERAGE = 0.6

RANDOM_STATE = 42

for _p in (INTERIM, PROCESSED, PLOTS, REPORTS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
