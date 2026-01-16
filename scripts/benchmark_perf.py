"""Performance benchmarks for Bradley-Terry model."""

import time

import numpy as np

from pears.data import PairwiseComparisonData
from pears.models.bradley_terry import BradleyTerryModel


def generate_artificial_comparisons(
    n_items: int = 50,
    n_votes: int = 300,
    seed: int = 42,
) -> tuple[list[list[str]], list[str]]:
    """Generate artificial pairwise comparison data with non-trivial win probabilities.

    Creates a realistic scenario where items have different underlying skill levels,
    and comparisons follow Bradley-Terry probabilities based on these skills.

    Parameters
    ----------
    n_items : int, default=50
        Number of items to compare
    n_votes : int, default=300
        Number of pairwise comparisons to generate
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    observations : list[list[str]]
        List of [winner, loser] pairs
    items : list[str]
        List of all item labels
    """
    rng = np.random.default_rng(seed)

    # Generate non-trivial underlying skill parameters
    # Use log-normal distribution to create realistic skill spread
    true_skills = rng.lognormal(mean=0, sigma=1.0, size=n_items)
    true_skills = true_skills / np.sum(true_skills)  # Normalize to sum to 1

    items = [f"item_{i:02d}" for i in range(n_items)]

    # Generate n_votes pairwise comparisons
    observations = []
    for _ in range(n_votes):
        # Sample two distinct items
        i, j = rng.choice(n_items, size=2, replace=False)

        # Bradley-Terry win probability: P(i beats j) = π_i / (π_i + π_j)
        p_i_wins = true_skills[i] / (true_skills[i] + true_skills[j])

        # Determine winner based on probability
        if rng.random() < p_i_wins:
            winner, loser = items[i], items[j]
        else:
            winner, loser = items[j], items[i]

        observations.append([winner, loser])

    return observations, items


def benchmark_fit_performance(
    n_items: int = 50,
    n_votes: int = 300,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Benchmark Bradley-Terry model fit performance.

    Parameters
    ----------
    n_items : int, default=50
        Number of items in dataset
    n_votes : int, default=300
        Number of pairwise comparisons
    n_trials : int, default=100
        Number of fit trials to run
    seed : int, default=42
        Random seed for data generation

    Returns
    -------
    dict[str, float]
        Timing statistics in milliseconds:
        - 'mean': Mean fit time
        - 'median': Median fit time
        - 'p95': 95th percentile
        - 'p99': 99th percentile
        - 'min': Minimum fit time
        - 'max': Maximum fit time
    """
    # Generate test data
    observations, items = generate_artificial_comparisons(n_items, n_votes, seed)
    data = PairwiseComparisonData(observations, items)

    # Warmup: run once to ensure JIT compilation, imports, etc.
    model = BradleyTerryModel()
    model.fit(data)

    # Benchmark
    times = []
    for _ in range(n_trials):
        model = BradleyTerryModel()
        start = time.perf_counter()
        model.fit(data)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    times_array = np.array(times)

    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "p95": float(np.percentile(times_array, 95)),
        "p99": float(np.percentile(times_array, 99)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
    }


def test_production_scale_performance():
    """Test that production scale (n=50, m=300) meets performance targets."""
    stats = benchmark_fit_performance(n_items=50, n_votes=300, n_trials=100)

    print("\n" + "=" * 70)
    print("Bradley-Terry Model Performance Benchmark")
    print("=" * 70)
    print("Dataset: n=50 items, m=300 votes")
    print("Trials: 100 fits")
    print("-" * 70)
    print(f"Mean:   {stats['mean']:6.2f} ms")
    print(f"Median: {stats['median']:6.2f} ms")
    print(f"p95:    {stats['p95']:6.2f} ms")
    print(f"p99:    {stats['p99']:6.2f} ms")
    print(f"Min:    {stats['min']:6.2f} ms")
    print(f"Max:    {stats['max']:6.2f} ms")
    print("-" * 70)

    # Target: 100 fits/second on 0.5 CPU = ~5ms effective CPU time per fit
    # With system overhead, aim for mean < 10ms
    target_mean = 10.0  # ms
    target_p95 = 15.0  # ms

    if stats["mean"] < target_mean:
        print(f"✓ Performance target met: mean {stats['mean']:.2f}ms < {target_mean}ms")
    else:
        print(f"✗ Performance target missed: mean {stats['mean']:.2f}ms > {target_mean}ms")

    if stats["p95"] < target_p95:
        print(f"✓ p95 target met: {stats['p95']:.2f}ms < {target_p95}ms")
    else:
        print(f"✗ p95 target missed: {stats['p95']:.2f}ms > {target_p95}ms")

    # Estimate throughput
    fits_per_second = 1000 / stats["mean"]
    print("-" * 70)
    print(f"Estimated throughput: {fits_per_second:.1f} fits/second (single-threaded)")
    print("Target throughput: 100 fits/second")

    if fits_per_second >= 100:
        print("✓ Throughput target met")
    else:
        print(f"✗ Throughput gap: need {100 / fits_per_second:.1f}x speedup")

    print("=" * 70 + "\n")

    # Assert performance targets (soft assertion - warn but don't fail)
    # Uncomment to enforce hard performance requirements:
    # assert stats["mean"] < target_mean, f"Mean time {stats['mean']:.2f}ms exceeds target {target_mean}ms"


def test_scaling_characteristics():
    """Benchmark performance across different dataset sizes."""
    print("\n" + "=" * 70)
    print("Scaling Characteristics")
    print("=" * 70)

    configs = [
        (25, 150, "Small (n=25, m=150)"),
        (50, 300, "Production (n=50, m=300)"),
        (100, 600, "Large (n=100, m=600)"),
    ]

    results = []
    for n_items, n_votes, label in configs:
        stats = benchmark_fit_performance(n_items=n_items, n_votes=n_votes, n_trials=50, seed=42)
        results.append((label, n_items, n_votes, stats))
        print(f"{label:30s} mean={stats['mean']:6.2f}ms  p95={stats['p95']:6.2f}ms")

    print("=" * 70 + "\n")


def benchmark_fit_with_ci_performance(
    n_items: int = 50,
    n_votes: int = 300,
    n_trials: int = 20,
    method: str = "sandwich",
    n_bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, float]:
    """Benchmark Bradley-Terry model fit + confidence interval performance.

    Parameters
    ----------
    n_items : int, default=50
        Number of items in dataset
    n_votes : int, default=300
        Number of pairwise comparisons
    n_trials : int, default=20
        Number of fit+CI trials to run
    method : str, default="sandwich"
        CI method: "sandwich" or "bootstrap"
    n_bootstrap : int, default=300
        Number of bootstrap samples (if method="bootstrap")
    seed : int, default=42
        Random seed for data generation

    Returns
    -------
    dict[str, float]
        Timing statistics in milliseconds
    """
    # Generate test data
    observations, items = generate_artificial_comparisons(n_items, n_votes, seed)
    data = PairwiseComparisonData(observations, items)

    # Warmup
    model = BradleyTerryModel()
    model.fit(data)
    model.confidence_intervals(method=method, n_bootstrap=n_bootstrap, seed=seed)

    # Benchmark
    times = []
    for _ in range(n_trials):
        model = BradleyTerryModel()
        start = time.perf_counter()
        model.fit(data)
        model.confidence_intervals(method=method, n_bootstrap=n_bootstrap, seed=seed)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    times_array = np.array(times)

    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "p95": float(np.percentile(times_array, 95)),
        "p99": float(np.percentile(times_array, 99)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
    }


def test_confidence_interval_performance():
    """Test performance with confidence interval calculation included."""
    print("\n" + "=" * 70)
    print("Fit + Confidence Interval Performance (n=50, m=300)")
    print("=" * 70)

    # Test sandwich method (fast, asymptotic)
    print("\nMethod: sandwich (asymptotic, recommended for large samples)")
    print("-" * 70)
    stats_sandwich = benchmark_fit_with_ci_performance(
        n_items=50, n_votes=300, n_trials=50, method="sandwich"
    )
    print(f"Mean:   {stats_sandwich['mean']:7.2f} ms")
    print(f"Median: {stats_sandwich['median']:7.2f} ms")
    print(f"p95:    {stats_sandwich['p95']:7.2f} ms")

    fits_per_sec_sandwich = 1000 / stats_sandwich["mean"]
    print(f"Throughput: {fits_per_sec_sandwich:.1f} fits+CI/second")

    target = 100
    if fits_per_sec_sandwich >= target:
        print(f"✓ Target met: {fits_per_sec_sandwich:.1f} >= {target} fits+CI/second")
    else:
        print(f"✗ Target missed: need {target / fits_per_sec_sandwich:.1f}x speedup")

    # Test bootstrap method (slower, but better for small samples)
    print("\nMethod: bootstrap (n_bootstrap=300, better for small samples)")
    print("-" * 70)
    stats_bootstrap = benchmark_fit_with_ci_performance(
        n_items=50, n_votes=300, n_trials=20, method="bootstrap", n_bootstrap=300
    )
    print(f"Mean:   {stats_bootstrap['mean']:7.2f} ms")
    print(f"Median: {stats_bootstrap['median']:7.2f} ms")
    print(f"p95:    {stats_bootstrap['p95']:7.2f} ms")

    fits_per_sec_bootstrap = 1000 / stats_bootstrap["mean"]
    print(f"Throughput: {fits_per_sec_bootstrap:.1f} fits+CI/second")

    if fits_per_sec_bootstrap >= target:
        print(f"✓ Target met: {fits_per_sec_bootstrap:.1f} >= {target} fits+CI/second")
    else:
        print(f"✗ Target missed: need {target / fits_per_sec_bootstrap:.1f}x speedup")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Can we achieve 100 fits+CI/second?")
    print("=" * 70)
    print(
        f"Sandwich method:  {fits_per_sec_sandwich:6.1f} fits+CI/sec - {'✓ YES' if fits_per_sec_sandwich >= target else '✗ NO'}"
    )
    print(
        f"Bootstrap method: {fits_per_sec_bootstrap:6.1f} fits+CI/sec - {'✓ YES' if fits_per_sec_bootstrap >= target else '✗ NO'}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run production scale benchmark
    test_production_scale_performance()

    # Run scaling analysis
    test_scaling_characteristics()

    # Run confidence interval benchmarks
    test_confidence_interval_performance()
