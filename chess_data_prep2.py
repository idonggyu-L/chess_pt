import pandas as pd
import os

input_csv = "/home/hail/chess_pt/train_data.csv"
df = pd.read_csv(input_csv)

output_dir = "train"
os.makedirs(output_dir, exist_ok=True)

for idx, (game_id, game_data) in enumerate(df.groupby("game_id")):
    game_data.to_csv(os.path.join(output_dir, f"{idx}.csv"), index=False)