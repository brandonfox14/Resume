import os
import mne
import pandas as pd

RAW_DIR = "data/raw/eeg"
EXPORT_DIR = "data/eeg_csv"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Sessions and tasks to scan
sessions = ["ses-1", "ses-2"]
tasks = ["eyesopen", "eyesclosed"]

print("📤 Exporting EEG recordings to CSV (10 seconds, filtered 1–40 Hz)...")
print("------------------------------------------------------------")

n_exported = 0
n_skipped = 0

for subj in sorted(os.listdir(RAW_DIR)):
    if not subj.startswith("sub-"):
        continue

    subj_path = os.path.join(RAW_DIR, subj)
    for ses in sessions:
        for task in tasks:
            set_fname = f"{subj}_{ses}_task-{task}_eeg.set"
            set_path = os.path.join(subj_path, ses, "eeg", set_fname)

            if not os.path.exists(set_path):
                print(f"❌ Skipped (not found): {set_fname}")
                n_skipped += 1
                continue

            try:
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
                raw.pick_types(eeg=True)
                raw.crop(tmax=10)
                raw.filter(1.0, 40.0, verbose=False)

                data, times = raw.get_data(return_times=True)
                df = pd.DataFrame(data.T, columns=raw.ch_names)
                df["Time"] = times

                out_fname = f"{subj}_{ses}_{task}.csv"
                out_path = os.path.join(EXPORT_DIR, out_fname)
                df.to_csv(out_path, index=False)
                print(f"✅ Exported: {out_fname}")
                n_exported += 1

            except Exception as e:
                print(f"❌ Failed: {set_fname} ({e})")
                n_skipped += 1

print("------------------------------------------------------------")
print(f"✅ Done. Exported: {n_exported} CSVs. Skipped: {n_skipped}.")
