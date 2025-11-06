import csv
import os
from tqdm import tqdm

input_path = "/home/hail/Desktop/lichess_db_standard_rated_2019-02.csv"

output_dirs = {
    "beginner": "/media/hail/HDD/beginner_e",
    "intermediate": "/media/hail/HDD/intermediate_e",
    "expert": "/media/hail/HDD/expert_e"
}

for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

counters = {k: 0 for k in output_dirs.keys()}

def classify_elo(elo):
    elo = int(elo)
    if elo <= 1367:
        return "beginner", -1
    elif elo < 1739:
        return "intermediate", 0
    else:
        return "expert", 1

with open(input_path, newline='', encoding="utf-8") as f:
    total_lines = sum(1 for _ in f) - 1

valid = 0
prev_game_id = None
current_rows = []

with open(input_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in tqdm(reader, total=total_lines, desc="Processing", ncols=100):
        # classical + normal ??? ??
        if row["type"].strip().lower() != "classical":
            continue
        if row["termination"].strip().lower() != "normal":
            continue

        game_id = row["game_id"]

        if prev_game_id is not None and game_id != prev_game_id:
            white_elo = current_rows[0]["white_elo"]
            group, label = classify_elo(white_elo)
            for r in current_rows:
                r["labels"] = label

            out_dir = output_dirs[group]
            out_idx = counters[group]
            out_path = os.path.join(out_dir, f"{out_idx}.csv")

            with open(out_path, "w", newline='', encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=current_rows[0].keys())
                writer.writeheader()
                writer.writerows(current_rows)

            counters[group] += 1
            current_rows = []

        current_rows.append(row)
        prev_game_id = game_id
        valid += 1

    if current_rows:
        white_elo = current_rows[0]["white_elo"]
        group, label = classify_elo(white_elo)
        for r in current_rows:
            r["labels"] = label

        out_dir = output_dirs[group]
        out_idx = counters[group]
        out_path = os.path.join(out_dir, f"{out_idx}.csv")

        with open(out_path, "w", newline='', encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=current_rows[0].keys())
            writer.writeheader()
            writer.writerows(current_rows)
        counters[group] += 1

print("Done")
print(f"Processed total {total_lines:,} lines, {valid:,} valid classical games.")
print("Saved counts:", counters)
