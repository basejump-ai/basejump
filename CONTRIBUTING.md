# Contributing to Basejump

Thank you for your interest in contributing to Basejump! We welcome contributions from the community.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux; on Windows: .venv\Scripts\activate
```
3. Install development dependencies:
```bash
   pip install basejump[dev]
```
4. Set up pre-commit hooks:
```bash
   pre-commit install
```

## Code Style

- We use [Black](https://github.com/psf/black) for code formatting
- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Docstrings follow the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html)

Pre-commit hooks will automatically check formatting and linting before each commit.

## Testing

Run tests with:
```bash
pytest tests
```

Type checking with:
```bash
mypy basejump-core
```

Please ensure all tests pass and add tests for new features.

## Submitting Changes

1. Create a new branch for your feature or bugfix
2. Make your changes and commit them with clear, descriptive messages
3. Push to your fork and submit a pull request
4. Ensure all checks pass

## Questions?

Join our [Discord community](https://discord.gg/Dhgz5ekRCF) or open an issue for discussion.