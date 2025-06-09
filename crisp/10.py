# 10. Write python functions to implement the following:
# (a) Add an element in a given set.
# (b) Update the set. 
# (c) Remove an element from the set.
# (d) Discard an element from the set. 
# (e) Pop the element from the set.
# (f) Clear the set.
# (g) Distinguish between remove() and discard() functions in python.

def add_element(s, element):
    """Add an element to the set."""
    s.add(element)

def update_set(s, elements):
    """Update the set with multiple elements."""
    s.update(elements)

def remove_element(s, element):
    """Remove an element from the set (raises an error if not found)."""
    s.remove(element)  # Raises KeyError if element is not found

def discard_element(s, element):
    """Discard an element from the set (does not raise an error if not found)."""
    s.discard(element)  # No error if element is not found

def pop_element(s):
    """Pop and return an arbitrary element from the set."""
    return s.pop()  # Raises KeyError if the set is empty

def clear_set(s):
    """Clear all elements from the set."""
    s.clear()

# Example usage
A = {1, 2, 3, 4}

add_element(A, 5)
print("After adding 5:", A)

update_set(A, [6, 7, 8])
print("After updating with [6, 7, 8]:", A)

remove_element(A, 3)
print("After removing 3:", A)

discard_element(A, 10)  # No error if 10 is not present
print("After discarding 10 (not present):", A)

popped = pop_element(A)
print("Popped element:", popped)
print("After popping an element:", A)

clear_set(A)
print("After clearing the set:", A)
