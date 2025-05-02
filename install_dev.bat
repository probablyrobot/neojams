@echo off
REM Script to set up a development environment for NeoJAMS on Windows

echo Setting up NeoJAMS development environment...

REM Check if Poetry is installed
where poetry >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Poetry not found. Please install Poetry first:
    echo https://python-poetry.org/docs/#installation
    exit /b 1
)

REM Install dependencies with Poetry
echo Installing dependencies with Poetry...
poetry install --with dev

REM Set up pre-commit hooks
echo Setting up pre-commit hooks...
poetry run pre-commit install

echo.
echo Development environment setup complete!
echo.
echo To activate the environment, run:
echo   poetry shell
echo.
echo To run tests:
echo   poetry run pytest
echo.
echo To run pre-commit checks:
echo   poetry run pre-commit run --all-files
