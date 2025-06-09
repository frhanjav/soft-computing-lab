# 9. Write python functions to check whether two input crisp sets is disjoint or not.

def are_disjoint(set1, set2):
    """
    Check if two sets are disjoint (have no common elements).

    :param set1: First set
    :param set2: Second set
    :return: True if sets are disjoint, False otherwise
    """
    return set1.isdisjoint(set2)

# Example usage
A = {1, 2, 3}
B = {4, 5, 6}
C = {3, 4, 5}

print("Are A and B disjoint?", are_disjoint(A, B))  # Output: True
print("Are A and C disjoint?", are_disjoint(A, C))  # Output: False
