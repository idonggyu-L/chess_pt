import os
import random
import shutil

base_dir = "/media/hail/HDD"
expert_dir = os.path.join(base_dir, "expert_f")
intermediate_dir = os.path.join(base_dir, "intermediate_f")
beginner_dir = os.path.join(base_dir, "beginner_f")

mixed_dir = os.path.join(base_dir, "mixed")
mix_pairs = {
    "exp_int_f": [expert_dir, intermediate_dir],
    "int_beg_f": [intermediate_dir, beginner_dir],
    "exp_beg_f": [expert_dir, beginner_dir]
}

os.makedirs(mixed_dir, exist_ok=True)
for name in mix_pairs.keys():
    os.makedirs(os.path.join(mixed_dir, name), exist_ok=True)


def mix_and_copy(pair_dirs, save_dir):
    files = []
    for d in pair_dirs:
        files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]

    random.shuffle(files)

    for i, src in enumerate(files):
        dst = os.path.join(save_dir, f"{i}.csv")
        shutil.copy2(src, dst)

for name, dirs in mix_pairs.items():
    print(f"Processing {name}...")
    save_path = os.path.join(mixed_dir, name)
    mix_and_copy(dirs, save_path)

print("done")
