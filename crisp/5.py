# 5. Write python functions to compute the symmetric difference of two crisp sets.

def crisp_symmetric_difference(set1, set2):
    """
    Compute the symmetric difference of two crisp sets.

    :param set1: First set
    :param set2: Second set
    :return: Symmetric difference of set1 and set2
    """
    return set1 ^ set2  # Equivalent to set1.symmetric_difference(set2)

# Example usage
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

sym_diff_set = crisp_symmetric_difference(A, B)
print("Symmetric Difference of A and B:", sym_diff_set)
