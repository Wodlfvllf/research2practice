from setuptools import setup, find_packages

setup(
    name="geolora",
    version="0.1.0",
    description="Geometric Low-Rank Adaptation (GeoLoRA) implementation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)