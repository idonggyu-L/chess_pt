import os
import random
import shutil

base_dir = "/media/hail/HDD"

expert_dir = os.path.join(base_dir, "expert_e")
inter_dir = os.path.join(base_dir, "intermediate_e")
begin_dir = os.path.join(base_dir, "beginner_e")

mix_exp = os.path.join(base_dir, "mixed", "exp_beg_e")  # expert+beginner
mix_int = os.path.join(base_dir, "mixed", "int_beg_e")  # inter+beginner
mix_beg = os.path.join(base_dir, "mixed", "exp_int_e")  # expert+intermediate

output_dirs = {
    "expert": [os.path.join(base_dir, "expert_swap_Ae"), os.path.join(base_dir, "expert_swap_Be")],
    "inter": [os.path.join(base_dir, "inter_swap_Ae"), os.path.join(base_dir, "inter_swap_Be")],
    "begin": [os.path.join(base_dir, "begin_swap_Ae"), os.path.join(base_dir, "begin_swap_Be")]
}

for pair in output_dirs.values():
    for d in pair:
        os.makedirs(d, exist_ok=True)


def swap_random_half(src1, src2, saveA, saveB):
    files1 = sorted([f for f in os.listdir(src1) if f.endswith(".csv")])
    files2 = sorted([f for f in os.listdir(src2) if f.endswith(".csv")])

    n = min(len(files1), len(files2))
    half_n = n // 2
    indices = random.sample(range(n), half_n)

    for i in range(n):
        f1 = os.path.join(src1, files1[i])
        f2 = os.path.join(src2, files2[i])

        if i in indices:

            shutil.copy2(f2, os.path.join(saveA, f"{i}.csv"))
            shutil.copy2(f1, os.path.join(saveB, f"{i}.csv"))
        else:

            shutil.copy2(f1, os.path.join(saveA, f"{i}.csv"))
            shutil.copy2(f2, os.path.join(saveB, f"{i}.csv"))

    print(f"{os.path.basename(saveA)} & {os.path.basename(saveB)} 완료 (교환 {len(indices)}개)")


# 실행
swap_random_half(expert_dir, mix_int, *output_dirs["expert"])
swap_random_half(inter_dir, mix_beg, *output_dirs["inter"])
swap_random_half(begin_dir, mix_exp, *output_dirs["begin"])
