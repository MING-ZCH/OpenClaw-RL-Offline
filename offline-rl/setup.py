"""Setup for OpenClaw-RL Offline RL Extension."""

from setuptools import setup, find_packages

setup(
    name="openclaw-offline-rl",
    version="0.1.0",
    description="Offline RL extension for OpenClaw-RL: IQL, CQL, AWAC, Off-Policy GRPO "
                "with environment adapters for OSWorld, AndroidWorld, WebArena, and AlfWorld.",
    author="OpenClaw-RL Extended Team",
    python_requires=">=3.7",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
        "osworld": [
            "Pillow>=9.0.0",
            # osworld: install from https://github.com/xlang-ai/OSWorld
        ],
        "androidworld": [
            # android_world: install from https://github.com/google-research/android_world
        ],
        "webarena": [
            "playwright>=1.30.0",
        ],
        "alfworld": [
            # alfworld: install from https://github.com/alfworld/alfworld
        ],
    },
    package_data={
        "": ["configs/*.json"],
    },
)
