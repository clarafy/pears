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

## Quick Start

```python
from pears import BradleyTerryModel, ConfidenceIntervalRankingStrategy

model = BradleyTerryModel()
model.fit(comparisons)

strategy = ConfidenceIntervalRankingStrategy(alpha=0.05)
ranking = strategy.rank(model)
```
