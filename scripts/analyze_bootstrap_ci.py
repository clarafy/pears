"""
Bootstrap confidence interval analysis for Bradley-Terry model.

Analyzes how confidence interval properties vary with sample size:
- CI width vs sample size
- Pairwise overlap rate vs sample size
"""

import matplotlib.pyplot as plt
import numpy as np

from pears.data import PairwiseComparisonData
from pears.models.bradley_terry import BradleyTerryModel

# Configuration
SEED = 42
N_ITEMS = 10
LOGNORMAL_MU = 0
LOGNORMAL_SIGMA = 2
SAMPLE_SIZES = [50, 100, 150, 200]
N_BOOTSTRAP = 300
CI_ALPHA = 0.05  # 95% CI


def generate_true_scores(n_items, mu, sigma, seed):
    """Generate ground truth skill scores from lognormal distribution.

    Parameters
    ----------
    n_items : int
        Number of items
    mu : float
        Mean parameter for underlying normal distribution
    sigma : float
        Standard deviation for underlying normal distribution
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict[str, float]
        Mapping from item label (e.g., "Item_0") to true skill score.
        Scores are normalized to sum to 1.
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Generate lognormal scores
    raw_scores = rng.lognormal(mean=mu, sigma=sigma, size=n_items)

    # Normalize to sum to 1 (Bradley-Terry constraint)
    normalized_scores = raw_scores / np.sum(raw_scores)

    # Create labeled dictionary
    item_labels = [f"Item_{i}" for i in range(n_items)]
    true_scores = dict(zip(item_labels, normalized_scores, strict=True))

    return true_scores


def generate_synthetic_comparisons(true_scores, n_comparisons, seed):
    """Generate synthetic pairwise comparison data using Bradley-Terry model.

    Parameters
    ----------
    true_scores : dict[str, float]
        True skill scores for each item
    n_comparisons : int
        Number of pairwise comparisons to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list[list[str]]
        List of [winner, loser] pairs
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    items = list(true_scores.keys())
    n_items = len(items)
    observations = []

    for _ in range(n_comparisons):
        # Randomly sample two different items
        i, j = rng.choice(n_items, size=2, replace=False)
        item_i, item_j = items[i], items[j]

        # Bradley-Terry win probability: P(i beats j) = pi_i / (pi_i + pi_j)
        pi_i = true_scores[item_i]
        pi_j = true_scores[item_j]
        p_i_beats_j = pi_i / (pi_i + pi_j)

        # Bernoulli trial to determine winner
        if rng.random() < p_i_beats_j:
            winner, loser = item_i, item_j
        else:
            winner, loser = item_j, item_i

        observations.append([winner, loser])

    return observations


def calculate_ci_width(cis):
    """Calculate confidence interval width for each item.

    Parameters
    ----------
    cis : dict[str, tuple[float, float]]
        Confidence intervals from model.confidence_intervals()

    Returns
    -------
    dict[str, float]
        Mapping from item label to CI width (upper - lower)
    """
    return {label: upper - lower for label, (lower, upper) in cis.items()}


def calculate_pairwise_overlap(cis):
    """Calculate pairwise overlap rate among all item pairs.

    Two items have overlapping CIs if:
    max(lower_i, lower_j) < min(upper_i, upper_j)

    Parameters
    ----------
    cis : dict[str, tuple[float, float]]
        Confidence intervals for each item

    Returns
    -------
    tuple[int, float]
        (overlap_count, overlap_rate)
        - overlap_count: number of pairs with overlapping CIs
        - overlap_rate: fraction of all pairs with overlapping CIs
    """
    items = list(cis.keys())
    n_items = len(items)
    total_pairs = n_items * (n_items - 1) // 2  # C(n, 2)

    overlap_count = 0

    for i in range(n_items):
        for j in range(i + 1, n_items):
            item_i, item_j = items[i], items[j]
            lower_i, upper_i = cis[item_i]
            lower_j, upper_j = cis[item_j]

            # Check for overlap: [lower_i, upper_i] ∩ [lower_j, upper_j] ≠ ∅
            if max(lower_i, lower_j) < min(upper_i, upper_j):
                overlap_count += 1

    overlap_rate = overlap_count / total_pairs

    return overlap_count, overlap_rate


def analyze_sample_size(true_scores, sample_size, n_bootstrap, alpha, seed):
    """Run full analysis pipeline for a given sample size.

    Parameters
    ----------
    true_scores : dict[str, float]
        True skill scores
    sample_size : int
        Number of comparisons to generate
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    seed : int
        Random seed for this sample size

    Returns
    -------
    dict
        Analysis results containing:
        - sample_size: int
        - estimated_scores: dict[str, float]
        - cis: dict[str, tuple[float, float]]
        - ci_widths: dict[str, float]
        - mean_ci_width: float
        - median_ci_width: float
        - pairwise_overlap_count: int
        - pairwise_overlap_rate: float
    """
    # Generate synthetic data
    items = list(true_scores.keys())
    observations = generate_synthetic_comparisons(true_scores, sample_size, seed)

    # Create PairwiseComparisonData
    data = PairwiseComparisonData(observations, items)

    # Fit Bradley-Terry model
    model = BradleyTerryModel()
    model.fit(
        data,
        ci_method="sandwich",
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed + 1000,  # Offset to ensure different seed from data generation
    )

    # Get point estimates
    estimated_scores = model.scores()

    # Run bootstrap
    cis = model.confidence_intervals()

    # Calculate CI widths
    ci_widths = calculate_ci_width(cis)

    # Calculate pairwise overlap
    overlap_count, overlap_rate = calculate_pairwise_overlap(cis)

    return {
        "sample_size": sample_size,
        "estimated_scores": estimated_scores,
        "cis": cis,
        "ci_widths": ci_widths,
        "mean_ci_width": np.mean(list(ci_widths.values())),
        "median_ci_width": np.median(list(ci_widths.values())),
        "pairwise_overlap_count": overlap_count,
        "pairwise_overlap_rate": overlap_rate,
    }


def print_results_table(results):
    """Print formatted results table to console.

    Parameters
    ----------
    results : list[dict]
        List of analysis results from analyze_sample_size()
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print(
        f"Configuration: {N_ITEMS} items, {N_BOOTSTRAP} bootstrap samples, "
        f"{100 * (1 - CI_ALPHA):.0f}% CI"
    )
    print("=" * 80 + "\n")

    # Summary table
    print(f"{'Sample Size':<15} {'Mean CI Width':<18} {'Median CI Width':<18} {'Overlap Rate':<15}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['sample_size']:<15} "
            f"{result['mean_ci_width']:<18.6f} "
            f"{result['median_ci_width']:<18.6f} "
            f"{result['pairwise_overlap_rate']:<15.4f}"
        )

    print("\n" + "=" * 80)

    # Detailed breakdown for each sample size
    for result in results:
        print(f"\nSample Size: {result['sample_size']}")
        print(
            f"  Pairwise overlap: {result['pairwise_overlap_count']}/45 pairs "
            f"({result['pairwise_overlap_rate']:.2%})"
        )

        # Show CI width distribution
        widths = sorted(result["ci_widths"].values())
        print("  CI width distribution:")
        print(f"    Min:  {np.min(widths):.6f}")
        print(f"    Q25:  {np.percentile(widths, 25):.6f}")
        print(f"    Q50:  {np.median(widths):.6f}")
        print(f"    Q75:  {np.percentile(widths, 75):.6f}")
        print(f"    Max:  {np.max(widths):.6f}")


def plot_results(results):
    """Generate visualization plots for analysis results.

    Parameters
    ----------
    results : list[dict]
        List of analysis results from analyze_sample_size()
    """
    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    sample_sizes = [r["sample_size"] for r in results]

    # Plot 1: Mean CI width vs sample size
    ax = axes[0, 0]
    mean_widths = [r["mean_ci_width"] for r in results]
    ax.plot(sample_sizes, mean_widths, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Sample Size (Number of Comparisons)", fontsize=11)
    ax.set_ylabel("Mean CI Width", fontsize=11)
    ax.set_title("Mean Confidence Interval Width vs Sample Size", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 2: Median CI width vs sample size
    ax = axes[0, 1]
    median_widths = [r["median_ci_width"] for r in results]
    ax.plot(sample_sizes, median_widths, marker="s", linewidth=2, markersize=8, color="coral")
    ax.set_xlabel("Sample Size (Number of Comparisons)", fontsize=11)
    ax.set_ylabel("Median CI Width", fontsize=11)
    ax.set_title("Median Confidence Interval Width vs Sample Size", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 3: Pairwise overlap rate vs sample size
    ax = axes[1, 0]
    overlap_rates = [r["pairwise_overlap_rate"] for r in results]
    ax.plot(sample_sizes, overlap_rates, marker="^", linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Sample Size (Number of Comparisons)", fontsize=11)
    ax.set_ylabel("Pairwise Overlap Rate", fontsize=11)
    ax.set_title("Pairwise CI Overlap Rate vs Sample Size", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Plot 4: CI width distribution (bar chart for largest sample size)
    ax = axes[1, 1]
    largest_result = results[-1]
    ci_widths_list = list(largest_result["ci_widths"].values())
    items_list = list(largest_result["ci_widths"].keys())

    # Sort by CI width for better visualization
    sorted_items_widths = sorted(zip(items_list, ci_widths_list, strict=True), key=lambda x: x[1])
    sorted_items, sorted_widths = zip(*sorted_items_widths, strict=True)

    ax.barh(range(len(sorted_items)), sorted_widths, color="skyblue", edgecolor="navy")
    ax.set_yticks(range(len(sorted_items)))
    ax.set_yticklabels(sorted_items, fontsize=9)
    ax.set_xlabel("CI Width", fontsize=11)
    ax.set_title(
        f"CI Width per Item (n={largest_result['sample_size']})", fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = "/Users/domfj/pears/scripts/bootstrap_ci_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.show()


def main():
    """Main execution function."""
    print("Starting bootstrap confidence interval analysis...")
    print(f"Random seed: {SEED}")

    # Generate true scores (ground truth)
    true_scores = generate_true_scores(N_ITEMS, LOGNORMAL_MU, LOGNORMAL_SIGMA, SEED)

    print(f"\nGenerated {N_ITEMS} items with lognormal true scores")
    print("True scores (normalized):")
    for label, score in sorted(true_scores.items()):
        print(f"  {label}: {score:.6f}")

    # Run analysis for each sample size
    results = []
    for sample_size in SAMPLE_SIZES:
        print(f"\nAnalyzing sample size: {sample_size}...")
        result = analyze_sample_size(
            true_scores,
            sample_size,
            N_BOOTSTRAP,
            CI_ALPHA,
            seed=SEED + sample_size,  # Unique seed per sample size
        )
        results.append(result)

    # Print results
    print_results_table(results)

    # Generate plots
    print("\nGenerating visualization plots...")
    plot_results(results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
