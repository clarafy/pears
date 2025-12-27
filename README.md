# pears

Statistical ranking models with uncertainty quantification.

To analyze data from Thanksgiving 2025, run

`poetry run python src/pears.py`

## Installation

For development (installs package + all dependencies in editable mode):

```bash
git clone https://github.com/clarafy/pears.git
cd pears
poetry install
```

For users installing from GitHub:

```bash
pip install git+https://github.com/clarafy/pears.git
```

To install a specific tagged version:

```bash
pip install git+https://github.com/clarafy/pears.git@v1.0.0
```

Replace `v1.0.0` with the desired version tag.

### Development Workflow

```bash
# Run tests
poetry run pytest

# Activate virtual environment
poetry shell

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Run pre-commit hooks manually
poetry run pre-commit run --all-files

# Update pre-commit hook versions
poetry run pre-commit autoupdate
```

### Version Management

This project uses [bump-my-version](https://github.com/callowayproject/bump-my-version) for semantic versioning.

```bash
# Bump patch version (0.1.0 → 0.1.1) - bug fixes
poetry run bump-my-version bump patch

# Bump minor version (0.1.0 → 0.2.0) - new features
poetry run bump-my-version bump minor

# Bump major version (0.1.0 → 1.0.0) - breaking changes
poetry run bump-my-version bump major

# Preview changes without committing (dry-run)
poetry run bump-my-version bump --dry-run --verbose patch
```

Each bump command will:
1. Update version in `pyproject.toml` and `src/pears/__init__.py`
2. Create a git commit with message "Bump version: X → Y"
3. Create a git tag `vX.Y.Z`
4. Leave you ready to push: `git push && git push --tags`

## Quick Start

```python
from pears import BradleyTerryModel, ConfidenceIntervalRankingStrategy

model = BradleyTerryModel()
model.fit(comparisons)

strategy = ConfidenceIntervalRankingStrategy(alpha=0.05)
ranking = strategy.rank(model)
```
