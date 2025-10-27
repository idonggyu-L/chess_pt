
import pandas as pd
import os

def split_and_save_games(input_csv, output_base=".", threshold_high=1739, threshold_low=1367):
    df = pd.read_csv(input_csv)
    output_dirs = {
        "eval1": [],
        "eval2": [],
        "eval3": []
    }

    for dir_name in output_dirs.keys():
        os.makedirs(os.path.join(output_base, dir_name), exist_ok=True)

    for game_id, game_data in df.groupby("game_id"):
        white_elo = game_data["white_elo"].iloc[0]
        if white_elo >= threshold_high:
            output_dirs["eval1"].append((game_id, game_data))
        elif white_elo <= threshold_low:
            output_dirs["eval3"].append((game_id, game_data))
        else:
            output_dirs["eval2"].append((game_id, game_data))

    for dir_name, games in output_dirs.items():
        save_dir = os.path.join(output_base, dir_name)
        for idx, (_, game_data) in enumerate(games):
            game_data.to_csv(os.path.join(save_dir, f"{idx}.csv"), index=False)

if __name__ == "__main__":
    input_csv = "/home/hail/chess_pt/eval_data.csv"
    output_base = "/home/hail/chess_pt/data"
    split_and_save_games(input_csv, output_base)
