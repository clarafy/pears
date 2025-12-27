# pears

Statistical ranking models with uncertainty quantification.

## Installation

From the repository:

```bash
pip install -e /path/to/pears
```

Or from GitHub:

```bash
pip install git+https://github.com/domfj/pears.git
```

## Quick Start

```python
from pears import BradleyTerryModel, ConfidenceIntervalRankingStrategy

model = BradleyTerryModel()
model.fit(comparisons)

strategy = ConfidenceIntervalRankingStrategy(alpha=0.05)
ranking = strategy.rank(model)
```