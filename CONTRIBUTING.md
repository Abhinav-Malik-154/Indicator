# Contributing

Thank you for helping improve the BTC Price-Behaviour Indicator. Contributions
are welcome when they make the research more reproducible, the evaluation more
honest, or the dashboard more useful.

## Before You Start

- Read the [README](README.md) and [phase notes](PHASES.md) to understand the
  project boundaries and current findings.
- Check existing issues and pull requests before starting larger changes.
- For changes to metrics, labels, features, splits, or backtests, explain how
  leakage and out-of-sample behavior are protected.

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Development Workflow

1. Create a focused branch from the current default branch.
2. Make the smallest change that addresses the issue.
3. Add or update tests for behavioral changes.
4. Run the test suite and linter locally:

```bash
pytest
ruff check .
```

5. Update the relevant documentation when commands, outputs, or methodology
   change.
6. Open a pull request describing the motivation, validation performed, and
   any limitations or remaining uncertainty.

## Research Standards

- Never introduce look-ahead leakage, random splits, or full-series fitting
  into a time-series evaluation path.
- Keep training, validation, and test data chronologically separated.
- Report fees, slippage, trade counts, and the base-rate comparison where they
  affect a result.
- Do not remove a failed experiment or negative result merely because it is
  inconvenient.
- Treat live signals as append-only records; do not rewrite history after the
  outcome is known.
- Do not commit secrets, local environment files, generated model artifacts, or
  raw data that is already covered by `.gitignore`.

## Pull Requests

Please include:

- a concise problem statement and summary of the change;
- tests or a clear reason tests are not applicable;
- commands used to validate the change;
- before/after screenshots for dashboard UI changes;
- notes about changed assumptions, metrics, or reproducibility.

Pull requests may be revised when a result is not reproducible, a metric is
ambiguous, or a change makes the evaluation less conservative.

## Code Style

The project targets Python 3.11+, uses Ruff for linting, and follows the
existing module and test structure. Keep functions small, prefer explicit data
boundaries, and avoid unrelated formatting changes in the same pull request.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE.md).