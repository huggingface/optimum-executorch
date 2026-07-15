import argparse
import subprocess
import sys


STABLE_TORCH_DEPS = [
    "executorch==1.3.1+cpu",
    "torch==2.12.0+cpu",
    "torchvision==0.27.0+cpu",
    "torchao==0.17.0+cpu",
]

NIGHTLY_TORCH_DEPS = [
    "executorch==1.4.0.dev20260714+cpu",
    # Keep torch aligned with the published torchvision nightly dependency.
    "torch==2.14.0.dev20260713+cpu",
    "torchvision==0.29.0.dev20260714+cpu",
    "torchaudio==2.11.0.dev20260714+cpu",
    "torchao==0.18.0.dev20260714+cpu",
]


def install_torch_deps(dependency_stack: str):
    """Install torch related dependencies from pinned CPU wheels."""
    torch_deps = STABLE_TORCH_DEPS if dependency_stack == "stable" else NIGHTLY_TORCH_DEPS
    index_url = (
        "https://download.pytorch.org/whl/cpu"
        if dependency_stack == "stable"
        else "https://download.pytorch.org/whl/nightly/cpu"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",  # Prevent cached CUDA packages
            *torch_deps,
            "--extra-index-url",
            index_url,
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
            "git+https://github.com/huggingface/transformers@bdc85cb85c8772d37aa29ce447860b44d7fad6ef#egg=transformers",  # v5.0.0rc0
        ]
    )


def main():
    """Install optimum-executorch in dev mode with pinned dependencies."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_override_torch",
        action="store_true",
        help="Skip installation of pinned executorch and torch dependencies",
    )
    parser.add_argument(
        "--dependency_stack",
        choices=["stable", "nightly"],
        default="stable",
        help="Pinned dependency stack to install",
    )
    args = parser.parse_args()

    # Install pinned torch dependencies FIRST to avoid pulling CUDA versions.
    if not args.skip_override_torch:
        install_torch_deps(args.dependency_stack)

    # Install package with dev extras
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".[dev]"])

    # Install source dependencies
    install_dep_from_source()


if __name__ == "__main__":
    main()
