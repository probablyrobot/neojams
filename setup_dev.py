#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script for NeoJAMS development environment."""

import os
import platform
import subprocess
import sys


def setup_dev_environment():
    """Set up the development environment for NeoJAMS."""
    print("Setting up NeoJAMS development environment...")
    
    # Check if Poetry is installed
    try:
        subprocess.run(["poetry", "--version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Poetry not found. Please install Poetry first:")
        print("https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Install dependencies with Poetry
    print("Installing dependencies with Poetry...")
    subprocess.run(["poetry", "install", "--with", "dev"], check=True)
    
    # Set up pre-commit hooks
    print("Setting up pre-commit hooks...")
    subprocess.run(["poetry", "run", "pre-commit", "install"], check=True)
    
    print("\nDevelopment environment setup complete!")
    print("\nTo activate the environment, run:")
    print("  poetry shell")
    print("\nTo run tests:")
    print("  poetry run pytest")
    print("\nTo run pre-commit checks:")
    print("  poetry run pre-commit run --all-files")


if __name__ == "__main__":
    setup_dev_environment()
