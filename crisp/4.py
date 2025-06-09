# 4. Write python functions to compute the intersection of two crisp sets.

def crisp_intersection(set1, set2):
    """
    Compute the intersection of two crisp sets.

    :param set1: First set
    :param set2: Second set
    :return: Intersection of set1 and set2
    """
    return set1 & set2  # Equivalent to set1.intersection(set2)

# Example usage
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

intersection_set = crisp_intersection(A, B)
print("Intersection of A and B:", intersection_set)
