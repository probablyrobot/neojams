#!/bin/bash

# This script sets up the development environment for NeoJAMS
# It installs the package in development mode and sets up pre-commit hooks

set -e  # Exit on any error

echo "Setting up NeoJAMS development environment..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install the package in development mode with all extras
echo "Installing NeoJAMS in development mode with all extras..."
pip install -e ".[dev,tests,display]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files
echo "Running pre-commit on all files (this may take a while)..."
pre-commit run --all-files

echo ""
echo "Setup complete! Your development environment is ready."
echo ""
echo "To activate the environment: source venv/bin/activate"
echo "To run tests: pytest"
echo "To run linting: pre-commit run --all-files"
