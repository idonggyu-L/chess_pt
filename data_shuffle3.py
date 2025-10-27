import os
import random
import pandas as pd

folder1 = "/media/hail/HDD/chess_data/classical_g/train1"
folder2 = "/media/hail/HDD/chess_data/classical_g/train2"


files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
common_files = sorted([f for f in files1 & files2 if f.endswith(".csv")])


indices = list(range(len(common_files)))
shuffled_indices = indices[:]
random.shuffle(shuffled_indices)

for i, j in zip(indices, shuffled_indices):
    if i == j:
        continue

    file_i = common_files[i]
    file_j = common_files[j]

    path1_i = os.path.join(folder1, file_i)
    path2_i = os.path.join(folder2, file_i)

    path1_j = os.path.join(folder1, file_j)
    path2_j = os.path.join(folder2, file_j)


    df1 = pd.read_csv(path1_i)
    df2 = pd.read_csv(path2_i)


    df2.to_csv(path1_i, index=False)
    df1.to_csv(path2_i, index=False)

print("done")
