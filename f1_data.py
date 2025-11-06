import os
import pandas as pd
from pathlib import Path

src_folder = Path("/media/hail/HDD/chess_data_/f_score/f1_all_c/")
dst_root = Path("/media/hail/HDD/chess_data_/")

counters = {"f1": 0, "f2": 0, "f3": 0}

for sub in ["f1", "f2", "f3"]:
    (dst_root / sub).mkdir(parents=True, exist_ok=True)

for file_name in os.listdir(src_folder):
    if not file_name.endswith(".csv"):
        continue

    file_path = src_folder / file_name
    df = pd.read_csv(file_path)

    if "white_elo" not in df.columns or df["white_elo"].dropna().empty:
        print(f"?? {file_name}: white_elo ??, ??")
        continue

    elo = df["white_elo"].iloc[0]

    if elo <= 1367:
        target = "f3"
    elif elo >= 1739:
        target = "f1"
    else:
        target = "f2"

    new_name = f"{counters[target]}.csv"
    counters[target] += 1

    save_path = dst_root / target /new_name
    df.to_csv(save_path, index=False)
