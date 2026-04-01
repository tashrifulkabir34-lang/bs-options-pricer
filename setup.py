"""
setup.py — Black-Scholes Option Pricer & Greeks Dashboard
Author: tashrifulkabir34-lang
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bs-options-pricer",
    version="1.0.0",
    author="tashrifulkabir34-lang",
    description="Black-Scholes Option Pricer & Greeks Dashboard with IV solver and strategy payoffs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tashrifulkabir34-lang/bs-options-pricer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "bs-dashboard=app:main",
        ],
    },
)
