import numpy as np
from quick_lcs import length_sum

def test_length_sum():
    arr1 = np.array(["apple", "banana", "cherry"], dtype=object)
    arr2 = np.array(["pie", "split", "tart"], dtype=object)

    # Create the result array as a NumPy float64 array
    result = np.zeros(len(arr1), dtype=np.float64)  # Pre-allocate result array

    length_sum(arr1, arr2, result)  # Call the modified function

    expected = np.array([8.0, 11.0, 10.0], dtype=np.float64)  # Lengths of "applepie", "bananasplit", "cherrytart"
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

if __name__ == "__main__":
    test_length_sum()
    print("Test passed!")

