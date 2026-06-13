"""End-to-End Demo Script for RADAR Anomaly Detection.

This orchestrator script generates synthetic chest X-rays, runs self-supervised
training on normal radiographs, and executes evaluation over the test partition.
"""

import os
import subprocess
import sys


def run_command(command: str) -> int:
    """Executes a terminal command and streams its output to stdout in real-time.

    Args:
        command: The shell command to run.

    Returns:
        The integer return code from the subprocess execution.
    """
    print(f"Executing command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture everything in sequence
        text=True,
    )

    # Stream output dynamically
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line.strip())
        process.stdout.close()

    rc = process.wait()
    if rc != 0:
        print(f"Warning: Command failed with exit code {rc}")
    return rc


def main() -> None:
    """Orchestrates the entire data-generation, training, and evaluation demo."""
    print("=" * 60)
    print("RADAR: Single-Step Medical Anomaly Detection Orchestrator")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable or "python3"

    # Step 1: Generate Data
    print("\n[Step 1/3] Generating Synthetic Lung Dataset...")
    cmd_data = f"{python_exec} {os.path.join(base_dir, 'create_lung_dataset.py')}"
    data_rc = run_command(cmd_data)
    if data_rc != 0:
        print("Error: Synthetic data generation failed.")
        sys.exit(data_rc)

    # Step 2: Train Model
    print("\n[Step 2/3] Training RADAR Model (Self-Supervised)...")
    cmd_train = f"{python_exec} {os.path.join(base_dir, 'scripts/train.py')}"
    train_rc = run_command(cmd_train)
    if train_rc != 0:
        print("Error: Model training phase failed.")
        sys.exit(train_rc)

    # Step 3: Run Inference & Visualization
    print("\n[Step 3/3] Running Batch Inference & Visualizations...")
    cmd_infer = f"{python_exec} {os.path.join(base_dir, 'scripts/inference.py')}"
    infer_rc = run_command(cmd_infer)
    if infer_rc != 0:
        print("Error: Inference execution failed.")
        sys.exit(infer_rc)

    print("\n" + "=" * 60)
    print("DEMO ORCHESTRATION COMPLETED SUCCESSFULLY")
    print(f"Visual results saved in: {os.path.join(base_dir, 'outputs/visualizations')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

