
# 6. Write python functions to compute the power set of a crisp set.

from itertools import chain, combinations

def power_set(original_set):
    """
    Compute the power set of a given crisp set.

    :param original_set: Input set
    :return: A set containing all subsets of the original set
    """
    s = list(original_set)  # Convert set to list for indexing
    return {frozenset(subset) for subset in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))}

# Example usage
A = {1, 2, 3}
power_set_A = power_set(A)

print("Power Set of A:")
for subset in power_set_A:
    print(set(subset))  # Convert frozenset back to normal set for display
