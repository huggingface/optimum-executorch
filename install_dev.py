import argparse
import subprocess
import sys


def install_torch_nightly_deps():
    """Install torch related dependencies from pinned nightly"""
    NIGHTLY_VERSION = "dev20250620"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"executorch==0.7.0.{NIGHTLY_VERSION}",
            f"torch==2.8.0.{NIGHTLY_VERSION}",
            f"torchvision==0.23.0.{NIGHTLY_VERSION}",
            f"torchaudio==2.8.0.{NIGHTLY_VERSION}",
            "torchao==0.12.0.dev20250528",
            "--extra-index-url",
            "https://download.pytorch.org/whl/nightly/cpu",
        ]
    )


def install_dep_from_source():
    """Install deps from source at pinned commits"""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/huggingface/transformers@51f94ea06d19a6308c61bbb4dc97c40aabd12bad#egg=transformers",  # v4.52.4
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/pytorch-labs/tokenizers@fc32028858020c4fcafe37aaaeaf5d1b480336a2#egg=pytorch-tokenizers",
        ]
    )


def main():
    """Install optimum-executorch in dev mode with nightly dependencies"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_override_torch",
        action="store_true",
        help="Skip installation of nightly executorch and torch dependencies",
    )
    args = parser.parse_args()

    # Install package with dev extras
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".[dev]"])

    # Install nightly dependencies
    if not args.skip_override_torch:
        install_torch_nightly_deps()

    # Install source dependencies
    install_dep_from_source()


if __name__ == "__main__":
    main()
