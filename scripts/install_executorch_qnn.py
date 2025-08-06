#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys

# Configuration
COMMIT_HASH = "b29a627c958eca2d4ff89db90fa7206ee8d94d37"
REPO_URL = "https://github.com/pytorch/executorch.git"
TARGET_DIR = "/tmp/executorch"
INSTALL_SCRIPT = "install_executorch.py"
INSTALL_MARKER = ".installed_successfully"
QNN_INSTALL_SCRIPT = "backends/qualcomm/scripts/install_qnn_sdk.sh"
BUILD_SCRIPT = "backends/qualcomm/scripts/build.sh"


def run_command(cmd, cwd=None):
    """Run a shell command with real-time output streaming"""
    print(f"$ {' '.join(cmd)}")
    if cwd:
        print(f"  [in {cwd}]")

    # Start the process
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end="", flush=True)

    # Wait for process to finish and get exit code
    return_code = process.wait()

    if return_code != 0:
        print(f"\nERROR: Command failed with exit code {return_code}: {' '.join(cmd)}")
        sys.exit(1)


def main():
    print(f"Cloning ExecuTorch repository to {TARGET_DIR}")
    print(f"Using commit: {COMMIT_HASH}")

    # Clean up existing directory
    if os.path.exists(TARGET_DIR):
        print(f"Removing existing directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    # Clone repository (shallow clone)
    run_command(["git", "clone", "--depth", "1", REPO_URL, TARGET_DIR])

    # Checkout specific commit
    run_command(["git", "fetch", "--depth=1", "origin", COMMIT_HASH], cwd=TARGET_DIR)
    run_command(["git", "checkout", COMMIT_HASH], cwd=TARGET_DIR)

    # Check if installation has already completed
    install_marker_path = os.path.join(TARGET_DIR, INSTALL_MARKER)
    install_script_path = os.path.join(TARGET_DIR, INSTALL_SCRIPT)

    if not os.path.exists(install_marker_path):
        # Run installation script
        print(f"Running installation script: {install_script_path}")
        print("=" * 50)
        run_command([sys.executable, install_script_path], cwd=TARGET_DIR)
        print("=" * 50)

        # Create success marker
        open(install_marker_path, "w").close()
        print("Installation completed successfully!")
    else:
        print("Installation already completed - skipping install_executorch.py")

    # Run Qualcomm SDK installation
    qnn_script_path = os.path.join(TARGET_DIR, QNN_INSTALL_SCRIPT)
    print(f"Running Qualcomm SDK installation: {qnn_script_path}")
    print("=" * 50)
    run_command(["bash", QNN_INSTALL_SCRIPT], cwd=TARGET_DIR)
    print("=" * 50)

    # Run Qualcomm build script
    build_script_path = os.path.join(TARGET_DIR, BUILD_SCRIPT)
    print(f"Running build script: {build_script_path}")
    print("=" * 50)
    run_command(["bash", BUILD_SCRIPT], cwd=TARGET_DIR)
    print("=" * 50)

    print("\nAll steps completed successfully!")
    print(f"ExecuTorch installed at: {TARGET_DIR}")
    print(f"Commit: {COMMIT_HASH}")


if __name__ == "__main__":
    main()
