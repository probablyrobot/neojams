#!/bin/bash

# This script sets up the development environment for NeoJAMS
# It installs the package in development mode and sets up pre-commit hooks

set -e  # Exit on any error

echo "Setting up NeoJAMS development environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first:"
    echo "https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies with Poetry
echo "Installing dependencies with Poetry..."
poetry install --with dev

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
poetry run pre-commit install

echo -e "\nDevelopment environment setup complete!"
echo -e "\nTo activate the environment, run:"
echo "  poetry shell"
echo -e "\nTo run tests:"
echo "  poetry run pytest"
echo -e "\nTo run pre-commit checks:"
echo "  poetry run pre-commit run --all-files"
