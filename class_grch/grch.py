# class_grch/grch.py
# Run CLASS with GRCH module
import os
import subprocess

def run_class():
    print("Running CLASS with GRCH module...")
    cmd = [
        "class", "grch.ini"
    ]
    result = subprocess.run(cmd, cwd="class_grch", capture_output=True, text=True)
    if result.returncode == 0:
        print("CLASS completed successfully.")
        print("Output: results/class_output/grch_cl.dat")
    else:
        print("CLASS failed:", result.stderr)

if __name__ == "__main__":
    run_class()
