# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pears** is a Python library for statistical ranking models with uncertainty quantification. It provides tools to infer rankings from comparison data.

## Common Development Commands

### Setup & Installation

```bash
poetry install          # Install dependencies in editable mode (development)
```

### Running Tests

```bash
poetry run pytest                    # Run all tests
poetry run pytest tests/test_file.py # Run specific test file
poetry run pytest -v                 # Verbose output
poetry run pytest --cov              # With coverage report
```

### Code Quality & Linting

```bash
poetry run ruff check .              # Run linter
poetry run ruff format .             # Auto-format code
poetry run ruff check --fix .        # Lint with auto-fixes
poetry run mypy src/                 # Type checking
poetry run pre-commit run --all-files # Run all pre-commit hooks
```

### Version Management

Uses [bump-my-version](https://github.com/callowayproject/bump-my-version) for semantic versioning:

```bash
poetry run bump-my-version bump patch  # 0.1.0 → 0.1.1 (bug fixes)
poetry run bump-my-version bump minor  # 0.1.0 → 0.2.0 (new features)
poetry run bump-my-version bump major  # 0.1.0 → 1.0.0 (breaking changes)
poetry run bump-my-version bump --dry-run --verbose patch  # Preview changes
```

Each bump command creates a git commit, git tag (`vX.Y.Z`), and updates version in `pyproject.toml` and `src/pears/__init__.py`.

## Architecture & Code Organization

### Core Module Structure

**Models** (`src/pears/models/`):

- `bradley_terry.py`: Main `BradleyTerryModel` class that fits pairwise comparison data using iterative scaling algorithm. The algorithm is based on Newman (2023) JMLR and uses the `iterative_scaling_bt()` function for MLE estimation. Models are fitted to `PairwiseComparisonData` objects and expose a `scores()` method (requires `@require_fit` decorator).
- `base.py`: Contains `@require_fit` decorator that validates model is fitted before calling methods.

**Data** (`src/pears/data/`):

- `base.py`: `PairwiseComparisonData` class that encapsulates pairwise comparison observations. Validates that observations are exactly 2 items and all items exist in the provided items list. Provides `encoded_observations` property (integer-encoded tuples) and uses an internal `SequentialEncoder`.

**Encoders** (`src/pears/encoders/`):

- `base.py`: `SequentialEncoder` maps item labels (strings) to contiguous integers starting at 0. Used for internal computation and provides bidirectional mapping via `encode()` and `decode()`.

**Ranking** (`src/pears/ranking/`):

- Currently minimal (only module docstring). Placeholder for ranking strategies to convert model estimates to rankings.

### Data Flow

1. User creates `PairwiseComparisonData` with observations (`list[list[str]]` as [winner, loser]) and items list
2. `PairwiseComparisonData` internally encodes observations via `SequentialEncoder`
3. `BradleyTerryModel.fit(data)` receives the `PairwiseComparisonData` and:
   - Extracts `data.encoded_observations` as list of integer tuples
   - Runs `iterative_scaling_bt()` for MLE estimation
   - Stores encoder reference for later decoding
4. `model.scores()` returns decoded results (string labels → float skill scores)

### Key Design Patterns

**Encoder** (`SequentialEncoder`):

- Maps item labels (strings) to contiguous integers (0 to N-1) for efficient internal computation
- Bidirectional: `encode(label)` → int and `decode(int)` → label
- Enables fast array-based computations while maintaining human-readable results

**Dataset Class** (`PairwiseComparisonData`):

- Encapsulates observations and validation logic in a single place
- Validates at the boundary: checks observation format (2 items), type correctness (strings), and membership in items list
- Transparently handles encoding via internal `SequentialEncoder` while exposing both raw and encoded observations
- Provides `encoded_observations` property for model consumption

**Models** (`BradleyTerryModel`):

- Stateful object that separates fitting (`fit()`) from inference (`scores()`)
- Uses `@require_fit` decorator to enforce preconditions—methods can only be called after fitting
- Stores both fitted parameters (`params_`) and the encoder for decoding results back to item labels

**Ranking Functions** (`src/pears/ranking/`):

- Placeholder module for strategies to convert model estimates into rankings

## Code Quality Standards

### Linting & Formatting

- **Ruff** (replaces black, isort, flake8) with line length 100
- **MyPy** type checking (moderate strictness: `disallow_incomplete_defs=true`, but not full strictness)
- Pre-commit hooks auto-run ruff and mypy on commits

### Special Ruff Exceptions

- `N802`, `N806`: Disabled to allow scientific variable names (W_i, N_ij, etc.)
- `ARG001`, `ARG002`: Unused function/method arguments (common in abstract methods)
- `RET504`: Unnecessary variable assignments sometimes kept for readability
- `__init__.py` files: Unused imports allowed (F401) since they're re-exported

### MyPy Overrides

- `scipy`, `matplotlib`, `seaborn`, `sklearn`: Missing imports ignored (third-party stubs unavailable)

## Testing

Unit tests should contain the **minimal amount of tests to verify required behavior and only the required behavior**. This encourages:
- Concise, focused test cases
- Avoiding redundant or "nice-to-have" test coverage
- Clear intent: each test documents a specific requirement
- Faster iteration and maintenance

When writing tests, ask: "Is this test necessary to verify the required behavior?" If not, omit it.

## Git & Pre-commit

Pre-commit hooks exclude `src/bt.py` and `src/pears.py` from checks (likely experimental/analysis scripts).

Standard hooks run:

1. Basic file hygiene (trailing whitespace, EOF fixes, YAML/TOML validation)
2. Ruff linting with auto-fixes
3. Ruff formatting
4. MyPy type checking (only on `src/pears/` files)
