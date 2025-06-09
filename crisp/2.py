# 2. Write python functions to generate the n-sampling sets, each of size k, from a given set.

import random

def generate_sampling_sets(original_set, n, k):
    """
    Generate n sampling sets, each of size k, from a given set.

    :param original_set: The original set from which to sample.
    :param n: The number of sampling sets to generate.
    :param k: The size of each sampling set.
    :return: A list of sampling sets.
    """
    if k > len(original_set):
        raise ValueError("Sample size k cannot be larger than the original set size")
    
    original_list = list(original_set)  # Convert to list for indexing
    sampling_sets = [set(random.sample(original_list, k)) for _ in range(n)]
    return sampling_sets

# Example usage
universal_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
n = 3  # Number of sets
k = 4  # Size of each set

samples = generate_sampling_sets(universal_set, n, k)
for i, s in enumerate(samples, 1):
    print(f"Sample Set {i}: {s}")
