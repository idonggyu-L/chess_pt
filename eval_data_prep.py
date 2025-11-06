import os
import shutil
import random

# Source folders
tr1 = "/media/hail/HDD/chess_data_/classical/train1"
tr1_ = "/media/hail/HDD/chess_data_/classical/train1_"
tr2 = "/media/hail/HDD/chess_data_/classical/train2"
tr3 = "/media/hail/HDD/chess_data_/classical/train3"

# Output folders
out_A = "/media/hail/HDD/chess_data_/classical/new_A"
out_B = "/media/hail/HDD/chess_data_/classical/new_B"
os.makedirs(out_A, exist_ok=True)
os.makedirs(out_B, exist_ok=True)


def merge_and_shuffle(folder_list, out_folder):
    # Collect all CSV file paths from the given folders
    all_files = []
    for f in folder_list:
        all_files.extend([os.path.join(f, x) for x in os.listdir(f) if x.endswith(".csv")])

    # Shuffle the file order
    random.shuffle(all_files)

    # Copy files to the new folder with new names
    for i, src_path in enumerate(all_files):
        dst_path = os.path.join(out_folder, f"{i}.csv")
        shutil.copy(src_path, dst_path)


# Run the function
merge_and_shuffle([tr1, tr1_], out_A)
merge_and_shuffle([tr2, tr3], out_B)

print("Done: new_A and new_B folders created.")
