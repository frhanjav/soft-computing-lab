
# 8. Write python functions to check whether a crisp set is a superset of another set.

def is_superset(set1, set2):
    """
    Check if set1 is a superset of set2.

    :param set1: The first set (potential superset)
    :param set2: The second set (potential subset)
    :return: True if set1 is a superset of set2, otherwise False
    """
    return set1 >= set2  # Equivalent to set1.issuperset(set2)

# Example usage
A = {1, 2, 3, 4}
B = {1, 2}

print("Is A a superset of B?", is_superset(A, B))  # Output: True
print("Is B a superset of A?", is_superset(B, A))  # Output: False
