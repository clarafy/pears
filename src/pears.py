import numpy as np
import pandas as pd
from bt import iterative_scaling_bt
import argparse

def load_key(filepath):
    """
    Load the key CSV file into a dictionary mapping letters to names.
    Assumes columns: letter, name
    """
    key_df = pd.read_csv(filepath)
    return dict(zip(key_df['label'], key_df['identity']))

def extract_comparisons(df):
    """
    Extract paired comparisons from the DataFrame.
    Returns a list of (winner, loser) tuples, and optionally weights.
    Skips rows where preference is None or strength is 0.
    """
    print('Extracting comparisons from DataFrame...')
    comparisons = []
    weights = []
    forecasts = {}
    for _, row in df.iterrows():
        # Extract percentage from format like "61-80%"
        range_str = row.iloc[-1]  # Last column
        if pd.isna(range_str):
            continue
        # Parse range and use midpoint
        range_str = str(range_str).replace('%', '').strip()
        parts = range_str.split('-')
        if len(parts) == 2:
            lb = round(int(parts[0]) / 10) * 10
            ub = round(int(parts[1]) / 10) * 10
            midpoint = float((lb + ub) / 2 / 100)
        else:
            continue

        if row['strength'] == 0:  # TODO: incorporate ties
            continue
        pair = row['pair'].split()
        if len(pair) != 2:
            continue
        item1, item2 = pair
        preferred = row['preference']
        if preferred == item1:
            winner, loser = item1, item2
        elif preferred == item2:
            winner, loser = item2, item1
        else:
            continue  # Invalid preference
        comparisons.append((winner, loser))
        weights.append(row['strength'])
        if midpoint not in forecasts:
            forecasts[midpoint] = [(winner, loser)]
        else:
            forecasts[midpoint].append((winner, loser))
    return comparisons, weights, forecasts

def fit_bradley_terry(comparisons, weights=None): # TODO: incorporate strength
    """
    Fit a Bradley-Terry model to the comparisons.
    Returns the fitted parameters (strengths for each item).
    """
    # Get unique items
    items = set()
    for w, l in comparisons:
        items.add(w)
        items.add(l)
    items = sorted(list(items))
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    
    # Convert to indices
    indexed_comparisons = [(item_to_idx[w], item_to_idx[l]) for w, l in comparisons]

    # Compute win fraction for each item
    win_counts = {idx: 0 for idx in range(len(items))}
    total_counts = {idx: 0 for idx in range(len(items))}

    for w, l in indexed_comparisons:
        win_counts[w] += 1
        total_counts[w] += 1
        total_counts[l] += 1

    win_rates = {item: win_counts[item_to_idx[item]] / total_counts[item_to_idx[item]] 
                     for item in items if total_counts[item_to_idx[item]] > 0}
    
    # Fit the model
    print("Fitting Bradley-Terry model...")
    params = iterative_scaling_bt(indexed_comparisons, iterations=10000)
    
    return win_rates, {item: params[idx] for item, idx in item_to_idx.items()}

def get_rankings(params):
    """
    Get rankings from the fitted parameters.
    Returns a list of items sorted by strength (descending).
    """
    return sorted(params.keys(), key=lambda x: params[x], reverse=True)

def compute_log_likelihood(comparisons, params):
    """
    Compute the log-likelihood of the global params w.r.t. participant-specific comparisons.
    L(p) = sum_{ij} [w_ij * ln(p_i) - w_ij * ln(p_i + p_j)]
    """
    ll = 0.0
    for idx, (winner, loser) in enumerate(comparisons):
        # w_ij = weights[idx]
        p_i = params[winner]
        p_j = params[loser]
        ll += np.log(p_i) - np.log(p_i + p_j)
    return ll

def compute_qq_plot(forecasts, comparisons):
    """
    Compute data for QQ plot comparing forecasted probabilities to actual outcomes.
    Returns two lists: forecasted probabilities and actual outcomes (0 or 1).
    """
    forecasted_probs = []
    actual_probs = []
    n_per_bin = []
    
    for prob, pairs in forecasts.items():
        winner_count = 0
        loser_count = 0
        for winner, loser in pairs:
            winner_count += comparisons.count((winner, loser)) # TODO: not accounting for ties/0!
            loser_count += comparisons.count((loser, winner))

        forecasted_probs.append(prob)
        actual_probs.append(winner_count / (winner_count + loser_count))
        n_per_bin.append(winner_count + loser_count)

        # Sort by forecasted probabilities
        sorted_indices = np.argsort(forecasted_probs)
        forecasted_probs = [forecasted_probs[i] for i in sorted_indices]
        actual_probs = [actual_probs[i] for i in sorted_indices]
        n_per_bin

    return forecasted_probs, actual_probs, n_per_bin


# Example usage (can be removed or used in main)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bradley-Terry paired comparison analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to the main CSV data file")
    parser.add_argument("--key", type=str, required=True, help="Path to the key CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    key = load_key(args.key)
    
    # Overall rankings
    global_comparisons, weights, _ = extract_comparisons(df)
    win_rates, global_params = fit_bradley_terry(global_comparisons, weights)
    rankings = get_rankings(global_params)

    print()
    print("Global fitted parameters:")
    for item, val in global_params.items():
        print(f"{item}: {val}")
    print()

    print("----- Global Ranking -----")
    for r in rankings:
        resort_name = key.get(r, r)  # Use real name if available, else letter
        print(f"{rankings.index(r) + 1}. {resort_name}: {global_params[r]:.4f} (win rate: {win_rates.get(r, 0):.3f})")

    print("\n--- Rankings for each Participant---")
    
    # Rankings per unique name
    grouped = df.groupby('name')
    participant_lls = []
    participant_eces = []
    
    for name, group_df in grouped:
        comparisons, weights, forecasts = extract_comparisons(group_df)
        if not comparisons:
            continue
        ll = compute_log_likelihood(comparisons, global_params) # participant-specific log-likelihood under global
        forecasted_probs, actual_probs, n_per_bin = compute_qq_plot(forecasts, global_comparisons)
        # assert(np.sum(n_per_bin) == len(global_comparisons)) # participants may not forecast all possible prob values
        ece = np.sum(n_per_bin * np.abs(forecasted_probs)) / np.sum(n_per_bin)  # HERE: normalization?
        win_rates, params = fit_bradley_terry(comparisons, weights)

        rankings = get_rankings(params)
        
        print("\nRankings for {} (global log-likelihood: {:.4f}):".format(name, ll))
        for r in rankings:
            resort_name = key.get(r, r)
            print(f"{resort_name}: {params[r]:.4f} (win rate: {win_rates.get(r, 0):.3f})")
        print()

        print("QQ plot data (ECE = {:.4f}):".format(ece))
        for fp, ap in zip(forecasted_probs, actual_probs):
            print(f"Forecasted: {fp:.3f}, Actual: {ap:.3f}")
        print()
        
        participant_lls.append((name, ll))
        participant_eces.append((name, ece))
    
    # Sort and print participants by log-likelihood
    participant_lls.sort(key=lambda x: x[1])
    print("\n--- Participants by Log-Likelihood under Global Parameters (increasing) ---")
    for i, (name, ll) in enumerate(participant_lls, 1):
        print(f"{i}. {name}: {ll:.4f}")

    # Sort and print participants by ECE (increasing)
    participant_eces.sort(key=lambda x: x[1])
    print("\n--- Participants by Expected Calibration Error (increasing) ---")
    for i, (name, ece) in enumerate(participant_eces, 1):
        print(f"{i}. {name}: {ece:.4f}")