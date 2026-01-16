"""Ranking strategies based on confidence interval comparisons."""


def rank_conservative(
    scores: dict[str, float],
    confidence_intervals: dict[str, tuple[float, float]],
) -> list[str]:
    """
    Rank items conservatively based on confidence interval bounds.

    Conservative ranking: R̃_m = 1 + Σ_{m'∈[M]} 1{inf C_{m'} > sup C_m}

    Only counts item j as "definitely better" than item i when j's lower
    confidence bound is strictly greater than i's upper confidence bound.
    This ensures no model's performance is understated.

    Parameters
    ----------
    scores : dict[str, float]
        Mapping from item labels to point estimates (used for tie-breaking).
    confidence_intervals : dict[str, tuple[float, float]]
        Mapping from item labels to (lower_bound, upper_bound) CI tuples.

    Returns
    -------
    list[str]
        Items sorted by approximate rank (best first). Items with better
        approximate ranks appear first. Ties are broken by score (higher
        first), then alphabetically.

    Raises
    ------
    ValueError
        If scores and confidence_intervals have different keys or are empty.

    References
    ----------
    Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.
    https://arxiv.org/abs/2403.04132v1, Section 5, page 5.
    """
    # Validate inputs
    if not scores:
        return []

    if set(scores.keys()) != set(confidence_intervals.keys()):
        raise ValueError("scores and confidence_intervals must have the same item keys")

    # Compute approximate rank for each item
    ranks = {}
    for item in scores:
        _, upper_i = confidence_intervals[item]

        # Count how many items are definitely better (lower bound > this upper bound)
        count_better = 0
        for other_item in scores:
            if item == other_item:
                continue

            lower_j, _ = confidence_intervals[other_item]

            # Conservative: j is better if lower_j > upper_i
            if lower_j > upper_i:
                count_better += 1

        ranks[item] = 1 + count_better

    # Sort by: rank (ascending), score (descending), name (alphabetical)
    sorted_items = sorted(scores.keys(), key=lambda x: (ranks[x], -scores[x], x))

    return sorted_items


def rank_liberal(
    scores: dict[str, float],
    confidence_intervals: dict[str, tuple[float, float]],
) -> list[str]:
    """
    Rank items liberally based on confidence interval bounds.

    Liberal ranking: R̃_m = 1 + Σ_{m'∈[M]} 1{sup C_{m'} > inf C_m}

    Counts item j as better than item i when j's upper confidence bound is
    strictly greater than i's lower confidence bound (allowing overlaps).
    This ensures no model's performance is overstated.

    Parameters
    ----------
    scores : dict[str, float]
        Mapping from item labels to point estimates (used for tie-breaking).
    confidence_intervals : dict[str, tuple[float, float]]
        Mapping from item labels to (lower_bound, upper_bound) CI tuples.

    Returns
    -------
    list[str]
        Items sorted by approximate rank (best first). Items with better
        approximate ranks appear first. Ties are broken by score (higher
        first), then alphabetically.

    Raises
    ------
    ValueError
        If scores and confidence_intervals have different keys or are empty.

    References
    ----------
    Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.
    https://arxiv.org/abs/2403.04132v1, Section 5, page 5.
    """
    # Validate inputs
    if not scores:
        return []

    if set(scores.keys()) != set(confidence_intervals.keys()):
        raise ValueError("scores and confidence_intervals must have the same item keys")

    # Compute approximate rank for each item
    ranks = {}
    for item in scores:
        lower_i, _ = confidence_intervals[item]

        # Count how many items are possibly better (upper bound > this lower bound)
        count_better = 0
        for other_item in scores:
            if item == other_item:
                continue

            _, upper_j = confidence_intervals[other_item]

            # Liberal: j is better if upper_j > lower_i
            if upper_j > lower_i:
                count_better += 1

        ranks[item] = 1 + count_better

    # Sort by: rank (ascending), score (descending), name (alphabetical)
    sorted_items = sorted(scores.keys(), key=lambda x: (ranks[x], -scores[x], x))

    return sorted_items
