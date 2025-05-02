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

2. Install Poetry if you don't have it:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Set up the development environment:
   ```bash
   poetry install --with dev
   ```

   Alternatively, you can use the convenience scripts:
   - On Unix/macOS: `./install_dev.sh`
   - On Windows: `install_dev.bat`
   - Python script: `python setup_dev.py`

4. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

5. Activate the Poetry environment:
   ```bash
   poetry shell
   ```

## Code Quality

NeoJAMS uses several tools to maintain code quality:

1. **Black**: For automatic code formatting with a line length of 120 characters.

2. **Ruff**: For linting and code quality checks.

3. **Pre-commit**: To run these checks automatically before each commit.

4. **Pydantic**: For data validation and type checking.

### Type Checking with Pydantic

NeoJAMS uses Pydantic for runtime type checking and data validation. The models are defined in `neojams/models.py` and provide typed alternatives to the original classes. Benefits include:

- Runtime validation of input data
- Automatic type conversion where appropriate
- Clear definitions of data structures with type annotations
- IDE support for type hints and autocompletion

Example usage:

```python
from neojams import JAMSModel, AnnotationModel, ObservationModel

# Create an observation with validation
obs = ObservationModel(time=3.0, duration=2.0, value="C:maj", confidence=0.9)

# Create an annotation with validation
ann = AnnotationModel(
    namespace="chord",
    data=[obs],
    time=0.0,
    duration=10.0
)

# Create a complete JAMS object
jams = JAMSModel(annotations=[ann])

# Validate and convert to dict
jams_dict = jams.model_dump()
```

When developing new features, consider adding appropriate type annotations and leveraging the Pydantic models for validation.

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
poetry run pytest
```

## Package Structure

The package is organized as follows:

- `neojams/`: Main package code
- `tests/`: Test files
- `docs/`: Documentation
- `examples/`: Example scripts

## Release Process

1. Update version number in `pyproject.toml`
2. Update HISTORY.md
3. Create a release commit
4. Tag the release
5. Build and upload to PyPI:
   ```bash
   poetry build
   poetry publish
   ```

## Documentation

Documentation is built with Sphinx. To build the docs:

```bash
poetry install --with docs
poetry run sphinx-build -b html docs docs/_build/html
```
