import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def split_and_save_games(input_csv, output_base=".", threshold_high=1739, threshold_low=1367, seed=42):
    df = pd.read_csv(input_csv)

    condition = (df["white_elo"] >= threshold_high) | (df["white_elo"] <= threshold_low)
    filtered_df = df[condition]

    games = list(filtered_df.groupby("game_id"))

    np.random.seed(seed)
    np.random.shuffle(games)

    half = len(games) // 2
    games_train1 = games[:half]
    games_train2 = games[half:]

    dirs = {"train1": games_train1, "train2": games_train2}

    for dir_name in dirs.keys():
        os.makedirs(os.path.join(output_base, dir_name), exist_ok=True)

    for dir_name, games in dirs.items():
        save_dir = os.path.join(output_base, dir_name)
        for idx, (_, game_data) in enumerate(tqdm(games, desc=f"Saving to {dir_name}")):
            game_data.to_csv(os.path.join(save_dir, f"{idx}.csv"), index=False)

    for dir_name in dirs.keys():
        save_dir = os.path.join(output_base, dir_name)
        file_count = len([f for f in os.listdir(save_dir) if f.endswith(".csv")])
        print(f"{dir_name}: {file_count} files saved")

if __name__ == "__main__":
    input_csv = "/home/hail/chess_pt/train_data.csv"
    output_base = "/home/hail/chess_pt/data_"
    split_and_save_games(input_csv, output_base)
