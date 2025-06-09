# 3. Write python functions to compute the union of two crisp sets.

def crisp_union(set1, set2):
    """
    Compute the union of two crisp sets.

    :param set1: First set
    :param set2: Second set
    :return: Union of set1 and set2
    """
    return set1 | set2  # Equivalent to set1.union(set2)

# Example usage
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

union_set = crisp_union(A, B)
print("Union of A and B:", union_set)
