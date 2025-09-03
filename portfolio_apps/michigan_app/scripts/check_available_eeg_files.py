import os
from collections import defaultdict

RAW_DIR = "data/raw/eeg"

# Expected sessions and tasks per subject
expected_sessions = ["ses-1", "ses-2"]
expected_tasks = ["eyesopen", "eyesclosed"]

# Organize findings
found = defaultdict(lambda: defaultdict(set))
all_files = []

for root, dirs, files in os.walk(RAW_DIR):
    for fname in files:
        if fname.endswith("_eeg.set"):
            all_files.append(fname)
            try:
                parts = fname.replace("_eeg.set", "").split("_")
                participant_id = parts[0]
                session = parts[1]
                task = parts[2].replace("task-", "")

                found[participant_id][session].add(task)
            except Exception as e:
                print(f"⚠️ Could not parse filename: {fname} ({e})")

# --- Summary Report ---
print("📋 EEG Dry Run Summary")
print("-" * 40)

total_found = 0
total_expected = 0
subjects = sorted(found.keys())

for subj in subjects:
    print(f"Subject {subj}:")
    for sess in expected_sessions:
        for task in expected_tasks:
            total_expected += 1
            label = f"{sess}, {task}"
            if task in found[subj].get(sess, set()):
                print(f"  ✅ {label}")
                total_found += 1
            else:
                print(f"  ❌ {label} (missing)")
    print()

# --- Totals ---
print("-" * 40)
print(f"Subjects found: {len(subjects)}")
print(f"Files scanned: {len(all_files)}")
print(f"Expected session-task combos: {total_expected}")
print(f"Available .set combos: {total_found}")
print(f"Missing: {total_expected - total_found}")
