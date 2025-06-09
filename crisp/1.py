# Soft Computing Laboratory Assignment I

# Crisp Sets and Basic Operations

# 1. Write python functions to generate the n-population from a given set.

import random

def generate_population(given_set, n):
    """
    Generate an n-sized population (random sample) from the given set.

    Parameters:
    given_set (set or list): The set from which to sample.
    n (int): The number of elements in the generated population.

    Returns:
    list: A randomly selected population of size n.
    """
    if n > len(given_set):
        raise ValueError("n cannot be larger than the size of the given set")

    return random.sample(list(given_set), n)

# Example usage
given_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
n = 5
population = generate_population(given_set, n)
print("Generated Population:", population)
