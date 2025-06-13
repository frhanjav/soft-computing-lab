import numpy as np

def max_min_composition(R, S):
    """
    Perform max-min composition of two fuzzy relations R and S.
    
    For relations R: X -> Y and S: Y -> Z, the composition R ∘ S: X -> Z
    is computed as: (R ∘ S)(x,z) = max_y(min(R(x,y), S(y,z)))
    
    Args:
        R (numpy.ndarray): First fuzzy relation matrix (m x n)
        S (numpy.ndarray): Second fuzzy relation matrix (n x p)
    
    Returns:
        numpy.ndarray: Composed relation matrix (m x p)
    
    Raises:
        ValueError: If matrices dimensions are incompatible for composition
    """
    R = np.array(R)
    S = np.array(S)
    
    # Check dimension compatibility
    if R.shape[1] != S.shape[0]:
        raise ValueError(f"Cannot compose: R has {R.shape[1]} columns "
                        f"but S has {S.shape[0]} rows")
    
    # Get dimensions
    m, n = R.shape
    n, p = S.shape
    result = np.zeros((m, p))
    
    # Perform max-min composition
    for i in range(m):
        for j in range(p):
            # For each (i,j), find max over all intermediate connections
            min_values = [min(R[i, k], S[k, j]) for k in range(n)]
            result[i, j] = max(min_values)
    
    return result

def print_matrix(matrix, name):
    """Helper function to print matrices."""
    print(f"\n{name}:")
    for row in matrix:
        print([f"{val:.2f}" for val in row])

def main():
    """Demonstrate max-min composition."""
    
    print("Max-Min Composition of Fuzzy Relations")
    print("=" * 40)
    
    # Example: Person-Food-Health relationship
    # R: How much each person likes each food
    R = [
        [0.8, 0.3, 0.6],  # Person 1: loves pizza, dislikes salad, likes pasta
        [0.2, 0.9, 0.4]   # Person 2: dislikes pizza, loves salad, somewhat likes pasta
    ]
    
    # S: How much each food helps with health goals
    S = [
        [0.1, 0.8],  # Pizza: bad for weight loss, good for energy
        [0.9, 0.3],  # Salad: great for weight loss, poor for energy  
        [0.4, 0.7]   # Pasta: okay for weight loss, good for energy
    ]
    
    print_matrix(R, "R: Person-Food preferences")
    print_matrix(S, "S: Food-Health benefits")
    
    # Compute composition: Person-Health relationship
    result = max_min_composition(R, S)
    print_matrix(result, "R ∘ S: Person-Health outcomes")
    
    print("\nInterpretation:")
    print("Person 1 -> Weight Loss: 0.60 (mainly through pasta)")
    print("Person 1 -> Energy:      0.80 (through pizza)")
    print("Person 2 -> Weight Loss: 0.90 (through salad)")  
    print("Person 2 -> Energy:      0.70 (through pasta)")

if __name__ == "__main__":
    main()