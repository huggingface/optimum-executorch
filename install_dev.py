import argparse
import subprocess
import sys


def install_torch_nightly_deps():
    """Install torch related dependencies from pinned nightly"""
    EXECUTORCH_NIGHTLY_VERSION = "dev20250915"
    TORCHAO_NIGHTLY_VERSION = "dev20250915"
    # Torch nightly is aligned with pinned nightly in https://github.com/pytorch/executorch/blob/main/install_requirements.py#L74
    TORCH_NIGHTLY_VERSION = "dev20250915"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"executorch==0.8.0.{EXECUTORCH_NIGHTLY_VERSION}",
            f"torch==2.10.0.{TORCH_NIGHTLY_VERSION}",
            f"torchvision==0.25.0.{TORCH_NIGHTLY_VERSION}",
            f"torchaudio==2.8.0.{TORCH_NIGHTLY_VERSION}",
            f"torchao==0.14.0.{TORCHAO_NIGHTLY_VERSION}",
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
            "git+https://github.com/huggingface/transformers@91393fe4cc3266a05bc0d129e34ff5f761bb46e2#egg=transformers",  # 4.56.1
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
