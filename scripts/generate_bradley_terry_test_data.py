"""
One-time script to generate expected values for Bradley-Terry test.

This script runs the iterative_scaling_bt function on the test dataset
and outputs the expected scores for hardcoding in the unit test.
"""

from pears.models.bradley_terry import iterative_scaling_bt

# Define the test dataset
items = ["A", "B", "C", "D", "E"]
observations = [
    ["A", "B"],
    ["A", "C"],
    ["A", "D"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
    ["D", "E"],
    ["A", "E"],
    ["B", "E"],
    ["C", "E"],
]

# Manually encode observations (since we're testing the model)
# A=0, B=1, C=2, D=3, E=4 (alphabetical order matches SequentialEncoder)
encoder_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
encoded_observations = [
    (encoder_mapping[winner], encoder_mapping[loser]) for winner, loser in observations
]

print("Encoded observations:", encoded_observations)
print()

# Run iterative scaling
result = iterative_scaling_bt(encoded_observations)

print("Raw result (integer IDs to scores):")
for item_id, score in sorted(result.items()):
    print(f"  {item_id}: {score}")
print()

# Decode to string labels
reverse_mapping = {v: k for k, v in encoder_mapping.items()}
decoded_result = {reverse_mapping[item_id]: score for item_id, score in result.items()}

print("Expected scores for test (string labels to scores):")
print("{")
for label in sorted(decoded_result.keys()):
    print(f'    "{label}": {decoded_result[label]},')
print("}")
