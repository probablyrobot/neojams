# Development Guide for NeoJAMS

This document provides guidelines for developing the NeoJAMS package.

## Development Environment

### Python Version Support

NeoJAMS requires **Python 3.12 or later**. All development work should use Python 3.12+ features and avoid deprecated features or backward compatibility hacks.

### Setting Up a Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/marl/jams.git
   cd jams
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

   Alternatively, you can use the convenience scripts:
   - On Unix/macOS: `./install_dev.sh`
   - On Windows: `install_dev.bat`
   - Python script: `python setup_dev.py`

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Quality

NeoJAMS uses several tools to maintain code quality:

1. **Black**: For automatic code formatting with a line length of 120 characters.

2. **Ruff**: For linting and code quality checks.

3. **Pre-commit**: To run these checks automatically before each commit.

### Ruff Rules

The following Ruff rules are enabled:
- E: pycodestyle errors
- F: pyflakes
- I: isort
- B: flake8-bugbear
- C4: flake8-comprehensions
- UP: pyupgrade

### Coding Standards

When writing code for NeoJAMS:

1. Use Python 3.12+ features without worrying about backward compatibility:
   - Modern string formatting with f-strings
   - Type annotations where appropriate
   - Modern exception handling with `raise ... from ...`

2. Avoid unnecessary dictionary calls:
   - Use `{}` instead of `dict()`
   - Use `[]` instead of `list()`

3. Use proper exception handling:
   - Always include `from err` or `from None` in exception handling
   - Include `stacklevel=2` in all warning calls

4. Use modern import practices:
   - Avoid wildcard imports (`from module import *`)
   - Explicitly import only what is needed
   - Use `importlib.resources` instead of `pkg_resources`

5. Use modern class syntax:
   - Use `super()` instead of `super(__class__, self)`
   - No need to inherit from `object` explicitly

## Testing

Tests should be run with:

```bash
pytest
```

## Package Structure

The package is organized as follows:

- `neojams/`: Main package code
- `tests/`: Test files
- `docs/`: Documentation
- `scripts/`: Utility scripts

## Release Process

1. Update version number in `neojams/version.py`
2. Update HISTORY.md
3. Create a release commit
4. Tag the release
5. Build and upload to PyPI:
   ```bash
   python setup.py sdist
   twine upload dist/*
   ```

## Documentation

Documentation is built with Sphinx. To build the docs:

```bash
cd docs
make html
```
