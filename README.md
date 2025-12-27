# pears

Statistical ranking models with uncertainty quantification.

## Installation

For development (installs package + all dependencies in editable mode):

```bash
git clone https://github.com/domfj/pears.git
cd pears
poetry install
```

For users installing from GitHub:

```bash
pip install git+https://github.com/domfj/pears.git
```

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
```

## Quick Start

```python
from pears import BradleyTerryModel, ConfidenceIntervalRankingStrategy

model = BradleyTerryModel()
model.fit(comparisons)

strategy = ConfidenceIntervalRankingStrategy(alpha=0.05)
ranking = strategy.rank(model)
```