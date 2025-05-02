#!/usr/bin/env python
"""
Setup script for development environment of NeoJAMS.
This script installs the package in development mode and sets up pre-commit hooks.
"""

import subprocess
import sys


def run_command(command):
    """Run a command and return its output."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=True)
    return process


def main():
    """Main function to set up the development environment."""
    print("Setting up NeoJAMS development environment...")

    # Install the package in development mode
    print("\n1. Installing NeoJAMS in development mode with all extras...")
    run_command(f"{sys.executable} -m pip install -e '.[dev,tests,display]'")

    # Install pre-commit hooks
    print("\n2. Installing pre-commit hooks...")
    run_command(f"{sys.executable} -m pre_commit install")

    # Run pre-commit on all files
    print("\n3. Running pre-commit on all files (this may take a while)...")
    run_command(f"{sys.executable} -m pre_commit run --all-files")

    print("\nSetup complete! You're ready to develop NeoJAMS.")
    print("\nTo run tests: pytest")
    print("To run linting: pre-commit run --all-files")


if __name__ == "__main__":
    main()
