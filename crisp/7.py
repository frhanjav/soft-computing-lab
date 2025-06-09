
# 7. Write python functions to check whether a crisp set is a subset of another set.

def is_subset(set1, set2):
    """
    Check if set1 is a subset of set2.

    :param set1: The first set (potential subset)
    :param set2: The second set (potential superset)
    :return: True if set1 is a subset of set2, otherwise False
    """
    return set1 <= set2  # Equivalent to set1.issubset(set2)

# Example usage
A = {1, 2}
B = {1, 2, 3, 4}

print("Is A a subset of B?", is_subset(A, B))  # Output: True
print("Is B a subset of A?", is_subset(B, A))  # Output: False
