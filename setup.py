"""Setup script for LoRA-Diffusion package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lora-diffusion",
    version="0.1.0",
    author="Iman Khazrak, Robert Green",
    author_email="ikhazra@bgsu.edu",
    description="Parameter-Efficient Fine-Tuning for Diffusion Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/lora-diffusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lora-diffusion-train=scripts.train:main",
            "lora-diffusion-eval=scripts.evaluate:main",
        ],
    },
)
