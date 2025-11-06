import csv
import os

input_path = "/home/hail/Desktop/lichess_db_standard_rated_2019-07.csv"
output_dir = "/media/hail/HDD/all_trn"

os.makedirs(output_dir, exist_ok=True)

processed = 0
valid = 0
counter = 0

with open(input_path, newline='', encoding="utf-8") as f:
    total_lines = sum(1 for _ in f) - 1

with open(input_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    current_game_id = None
    current_rows = []

    for row in reader:
        processed += 1

        if row["type"].strip().lower() != "classical":
            continue
        if row["termination"].strip().lower() != "normal":
            continue

        valid += 1
        row["labels"] = 1

        game_id = row["game_id"]

        # ? ???? ?? ?? ??
        if current_game_id is not None and game_id != current_game_id:
            out_path = os.path.join(output_dir, f"{counter}.csv")
            with open(out_path, "w", newline='', encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=current_rows[0].keys())
                writer.writeheader()
                writer.writerows(current_rows)
            counter += 1
            current_rows = []

        current_game_id = game_id
        current_rows.append(row)

    # ??? ?? ??
    if current_rows:
        out_path = os.path.join(output_dir, f"{counter}.csv")
        with open(out_path, "w", newline='', encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=current_rows[0].keys())
            writer.writeheader()
            writer.writerows(current_rows)
        counter += 1

print(f"\n? Done! Processed {processed} lines, saved {valid} valid rows.")
print(f"? Total {counter} games saved to folder: {output_dir}")
