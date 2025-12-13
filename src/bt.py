import numpy as np
from collections import defaultdict

def iterative_scaling_bt(match_results: list[tuple[int, int]], 
                         initial_theta: dict[int, float] = None, 
                         iterations: int = 20, 
                         tolerance: float = 1e-6) -> dict[int, float]:
    """
    Implements the Iterative Scaling Algorithm (ISA) for the Bradley-Terry model 
    MLE based on the update rule in Equation (4) from:
    
    M. E. J. Newman. Efficient Computation of Rankings from Pairwise Comparisons. JMLR, 24(238):1−25, 2023.
    https://jmlr.org/papers/v24/22-1086.html

    Args:
        match_results: A list of tuples (i, j), where i won over j.
        initial_theta: A dictionary mapping item ID (int) to its starting 
                       positive skill parameter (float).
        iterations: Maximum number of iterations to run.
        tolerance: Stopping condition based on the maximum change in parameters.

    Returns:
        A dictionary mapping item ID to its final estimated skill parameter (pi_i).
    """
    if initial_theta is None:
        # Initialize all items with equal skill if not provided
        unique_items = set()
        for winner, loser in match_results:
            unique_items.add(winner)
            unique_items.add(loser)
        initial_theta = {item: 1.0 for item in unique_items}
    
    # 1. Initialization and Data Pre-processing
    
    # Get all unique items
    item_ids = sorted(initial_theta.keys())
    
    # Calculate W_i (Total Wins for item i)
    W = defaultdict(int)
    for winner, _ in match_results:
        W[winner] += 1
        
    # Calculate n_ij (Total comparisons between i and j)
    N = defaultdict(lambda: defaultdict(int))
    for winner, loser in match_results:
        N[winner][loser] += 1
        N[loser][winner] += 1
        
    # Convert initial_theta to an array/list for easier iteration
    pi = np.array([initial_theta[i] for i in item_ids], dtype=float)
    
    # Map item IDs to their index in the pi array
    item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
    
    # Main Iterative Loop
    print(f"Starting Iterative Scaling for {len(item_ids)} items.")
    
    for t in range(iterations):
        pi_prev = pi.copy()
        
        # Calculate the new pi_i for all items
        for i_idx, i in enumerate(item_ids):
            W_i = W[i]
            
            # The sum in the denominator
            denominator_sum = 0.0
            
            for j_idx, j in enumerate(item_ids):
                if i == j:
                    continue
                
                n_ij = N[i][j]
                
                # Only include pairs that were compared (n_ij > 0)
                if n_ij > 0:
                    # n_ij / (pi_i + pi_j)
                    denominator_sum += n_ij / (pi_prev[i_idx] + pi_prev[j_idx])
            
            # Update rule: pi_i^(t+1) = W_i / denominator_sum
            if W_i > 0 and denominator_sum > 0:
                pi[i_idx] = W_i / denominator_sum
            elif W_i == 0:
                # If an item never won, its score will approach zero.
                pi[i_idx] = 0.0
            # Note: W_i must be > 0 or the item is non-estimable (typically handled by setting score to 0
            # and removing it, but here we set to 0.0 as a safe numerical approx)
        
        # Normalization Step (for identifiability, sum(pi) = 1 is often used)
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        
        # Check for convergence
        max_change = np.max(np.abs(pi - pi_prev))
        
        # Optional: Print iteration status
        if (t + 1) % 1000 == 0 or t == 0: 
            print(f"  Iteration {t+1}: Max parameter change = {max_change:.6f}")
        
        if max_change < tolerance:
            print(f"Converged after {t+1} iterations.")
            break
        
    # Final results mapping index back to item ID
    final_pi = {item_ids[i]: float(pi[i]) for i in range(len(item_ids))}
    
    return final_pi

# --- Example Usage (Using the same data from the previous step) ---

# initial_theta = {
#     1: 1.0,
#     2: 1.0,
#     3: 1.0
# }

# sample_results = [
#     (1, 2),  # 1 wins over 2
#     (1, 3),  # 1 wins over 3
#     (2, 3),  # 2 wins over 3
#     (3, 2),  # 3 wins over 2
#     (2, 1)   # 2 wins over 1
# ]

# estimated_pi = iterative_scaling_bt(sample_results, iterations=50)

# print("\n--- Final Estimated Skill Parameters (pi) ---")
# # The final ranking is determined by the magnitude of these pi values.
# sorted_pi = sorted(estimated_pi.items(), key=lambda item: item[1], reverse=True)

# print("Final Ranking:")
# for item_id, pi_val in sorted_pi:
#     print(f"Item {item_id}: pi = {pi_val:.6f}")