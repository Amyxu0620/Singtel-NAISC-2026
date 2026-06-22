import os
import traceback
import subprocess


def run_case(train_path, test_path, case_name, log_path):
    cmd = [
        "python", "./main.py",
        "--train_data_filepath", train_path,
        "--test_data_filepath", test_path,
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"CASE: {case_name}\n")
        f.write("=" * 80 + "\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            f.write("STDOUT:\n")
            f.write(result.stdout + "\n")
            f.write("\nSTDERR:\n")
            f.write(result.stderr + "\n")
            f.write(f"\nRETURN CODE: {result.returncode}\n")
        except Exception:
            f.write("PIPELINE CRASHED\n")
            f.write(traceback.format_exc())
