import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
RAW_PATH = os.path.join("data", "raw", "adult24.csv")
CLEAN_PATH = os.path.join("data", "clean", "nhis_sleep_demo_clean.csv")

# -----------------------------
# Variables to select
# -----------------------------
SLEEP_PREFIX = "SLP"
DEMO_VARS = ["SEX_A", "AGEP_A", "EDUCP_A"]

# -----------------------------
# Codes representing invalid responses
# -----------------------------
INVALID_CODES = [7, 8, 9, 97, 98, 99]

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(RAW_PATH)

# -----------------------------
# Select sleep and demographic variables
# -----------------------------
sleep_vars = [col for col in df.columns if col.startswith(SLEEP_PREFIX)]
selected_vars = sleep_vars + DEMO_VARS
df_selected = df[selected_vars]

# -----------------------------
# Drop rows with invalid codes
# -----------------------------
for col in df_selected.columns:
    df_selected = df_selected[~df_selected[col].isin(INVALID_CODES)]

# -----------------------------
# Rename columns for clarity
# -----------------------------
rename_map = {
    "SLPMEDINTRO_A": "sleep_med_intro",
    "SLPHOURS_A": "sleep_hours_24h",
    "SLPREST_A": "wake_well_rested_30d",
    "SLPSTY_A": "trouble_staying_asleep_30d",
    "SLPMED1_A": "medication_prescribed_30d",
    "SLPMED2_A": "medication_otc_30d",
    "SLPMED3_A": "medication_mj_cbd_30d",
    "SEX_A": "sex",
    "AGEP_A": "age",
    "EDUCP_A": "education"
}
df_selected = df_selected.rename(columns=rename_map)

# -----------------------------
# Save cleaned dataset
# -----------------------------
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df_selected.to_csv(CLEAN_PATH, index=False)

print(f"✅ Cleaned file saved to {CLEAN_PATH} with {df_selected.shape[0]} rows.")
