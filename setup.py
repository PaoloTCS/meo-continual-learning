"""
Setup script for MEO: Mask Evolution Operators for Continual Learning Stability.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meo-continual-learning",
    version="1.0.0",
    author="Paolo Pignatelli di Montecalvo",
    author_email="paolo.pignatelli@verbumtechnologies.com",
    description="Mask Evolution Operators for Continual Learning Stability",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PaoloTCS/meo-continual-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords=[
        "continual learning",
        "catastrophic forgetting",
        "activation-space",
        "mask evolution operators",
        "deep learning",
        "neural networks",
        "machine learning",
        "artificial intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/PaoloTCS/meo-continual-learning/issues",
        "Source": "https://github.com/PaoloTCS/meo-continual-learning",
        "Documentation": "https://github.com/PaoloTCS/meo-continual-learning#readme",
    },
)
