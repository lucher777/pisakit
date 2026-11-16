# Contributing to pisakit

Thank you for your interest in improving `pisakit`. Contributions covering bug fixes, new international assessment presets, estimator optimizations, documentation improvements, and unit tests are welcome.

## Development Setup

```bash
git clone https://github.com/peacekange/pisakit.git
cd pisakit
pip install -e .[dev]
```

## Running Tests and Verification Checks

Before opening a pull request, verify that the full unit test suite, CLI smoke tests, and cross-package numerical equivalence checks pass locally:

```bash
pytest -v
python examples/quickstart.py
pisakit --help
pisakit equiv
pisakit demo-real
```

## Code Standards

1. Keep core computational dependencies minimal (`numpy`, `scipy`, `pandas`).
2. Add NumPy/Google-style docstrings and full Python type annotations (`from __future__ import annotations`) to all public classes and functions.
3. Any mathematical or numerical change must pass `verify_numerical_equivalence()` (`< 1e-10` against SciPy WLS, from-scratch OECD Fay BRR + Rubin pooling, and explicit dense block-diagonal REML).
4. Include unit tests in `tests/` covering normal inputs, edge cases, and invalid parameter errors.

## Reporting Bugs and Feature Requests

Please open an issue on the project repository containing a minimal reproducible example, `pisakit` version, Python version, and operating system.

## Community Conduct

Participants are expected to maintain constructive, respectful, and inclusive technical discussions in issues and pull requests.
