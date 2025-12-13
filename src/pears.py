import pandas as pd
from bt import iterative_scaling_bt

def load_csv(filepath):
    """
    Load the CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)

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
    for _, row in df.iterrows():
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
    return comparisons, weights

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
    params = iterative_scaling_bt(indexed_comparisons, iterations=1000)
    
    return win_rates, {item: params[idx] for item, idx in item_to_idx.items()}

def get_rankings(params):
    """
    Get rankings from the fitted parameters.
    Returns a list of items sorted by strength (descending).
    """
    return sorted(params.keys(), key=lambda x: params[x], reverse=True)

# Example usage (can be removed or used in main)
if __name__ == "__main__":
    df = load_csv("/Users/clarafy/code/pears/data/112725.csv")
    key = load_key("/Users/clarafy/code/pears/data/112725-key.csv")
    comparisons, weights = extract_comparisons(df)
    win_rates, params = fit_bradley_terry(comparisons, weights)
    rankings = get_rankings(params)

    print()
    print("Fitted parameters:")
    for item, val in params.items():
        print(f"{item}: {val}")
    print()

    print("Rankings:")
    for r in rankings:
        resort_name = key.get(r, r)  # Use real name if available, else letter
        rank = rankings.index(r) + 1
        print(f"{rank}. {resort_name}: {params[r]:.4f} (win rate: {win_rates.get(r, 0):.3f})")