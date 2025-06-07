import os
import csv
import math
import argparse
import subprocess
import numpy as np


def pass_at_k(result_list, n, k):
    prob_list = []
    for c in result_list:
        prob = 1 - math.comb(n - c, k) / math.comb(n, k)
        prob_list.append(prob)
    return float(np.mean(prob_list))


path = "build_generation"
files = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

parser = argparse.ArgumentParser(description="Analyze Verilog test results.")
parser.add_argument("--end", type=int, required=True)
args = parser.parse_args()
end = args.end

total_samples = len(files) * end
progress = 0
correct_list = []
for n, file in enumerate(files):
    correct = 0

    for i in range(1, end + 1):
        progress += 1
        path2file = os.path.join(path, file)
        excuted_file = os.path.join(path2file, f"{file}_sample{i:02d}")

        try:
            compile_out = subprocess.run(
                [
                    "iverilog",
                    "-Wall",
                    "-Winfloop",
                    "-Wno-timescale",
                    "-g2012",
                    "-s",
                    "tb",
                    "-o",
                    excuted_file,
                    f"{path2file}/{file}_sample{i:02d}.sv",
                    f"{path2file}/{file}_test.sv",
                    f"{path2file}/{file}_ref.sv",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Compile failed for {file} sample{i:02d}\n")
            print(e.stderr.decode())  # 印出 stderr 內容來看發生什麼事
            continue

        print(f"[{progress}/{total_samples}]")
        print(f"Running {excuted_file}...")
        try:
            out = subprocess.run(
                [excuted_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Execution failed for {file} sample{i:02d}\n")
            print(e.stderr.decode())
            continue
        except subprocess.TimeoutExpired:
            print(f"❌ Execution timed out for {file} sample{i:02d}\n")
            continue
        except Exception as e:
            print(f"❌ An unexpected error occurred for {file} sample{i:02d}: {e}\n")
            continue

        result = out.stdout.decode("utf-8")
        print(result)

        if "Mismatches: 0" in result:
            correct += 1

    correct_list.append(correct)

print(correct_list)

with open("eval/output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(correct_list)


print("Pass@1:", pass_at_k(correct_list, end, 1))
print("Pass@5:", pass_at_k(correct_list, end, 5))
print("Pass@10:", pass_at_k(correct_list, end, 10))
