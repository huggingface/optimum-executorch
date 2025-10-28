import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/executorch/version.py
filepath = "optimum/executorch/version.py"
try:
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRE = [
    "optimum~=1.24",
    "executorch>=1.0.0",
    "transformers==4.56.1",
]

TESTS_REQUIRE = [
    "accelerate>=0.26.0",
    "coremltools>=8.2.0",
    "datasets==3.6.0",  # Locked to 3.6.0 due to https://github.com/huggingface/datasets/issues/7707
    "parameterized",
    "pytest",
    "safetensors",
    "sentencepiece",
    "numba!=0.58.0",  # Due to the bug https://github.com/numba/numba/issues/9209
    "librosa",
    "soundfile",
    "tiktoken",
]


QUALITY_REQUIRE = ["black~=23.1", "ruff==0.4.4"]


EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
}


setup(
    name="optimum-executorch",
    version=__version__,
    description="Optimum Executorch is an interface between the Hugging Face libraries and ExecuTorch",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, inference, executorch",
    url="https://github.com/huggingface/optimum",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.10.0",
    include_package_data=True,
    zip_safe=False,
)
