@echo off
:: This script sets up the development environment for NeoJAMS on Windows
:: It installs the package in development mode and sets up pre-commit hooks

echo Setting up NeoJAMS development environment...

:: Create a virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install the package in development mode with all extras
echo Installing NeoJAMS in development mode with all extras...
pip install -e ".[dev,tests,display]"

:: Install pre-commit hooks
echo Installing pre-commit hooks...
pre-commit install

:: Run pre-commit on all files
echo Running pre-commit on all files (this may take a while)...
pre-commit run --all-files

echo.
echo Setup complete! Your development environment is ready.
echo.
echo To activate the environment: venv\Scripts\activate
echo To run tests: pytest
echo To run linting: pre-commit run --all-files

:: Keep the window open
pause
