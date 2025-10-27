import pandas as pd

file_path = '/media/hail/HDD/chess_data/raws/07_classical.csv'

df = pd.read_csv(file_path)

df['label'] = df['white_elo'].apply(
    lambda x: -1 if x <= 1367 else (1 if x >= 1739 else 0)
)

df.to_csv(file_path, index=False)
print(f"Label added and saved to the same file: {file_path}")
