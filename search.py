def linear_search(lst, val):
    for idx, x in enumerate(lst):
        if x == val:
            return idx
    return -1

def binary_search_iterative(lst, val, key):
    # while loop
    left, right = 0, len(lst) - 1

    while left <= right:
        idx  = (left + right)/2
        middle = key(lst[idx])

        if middle == val:
            return middle
        
        if middle < val:
            left = middle + 1
        
        elif middle > val:
            right = middle - 1
    return -1

def binary_search_recursive(lst, val):
    #log n
    # if left < right, check if its the middle, if not, try with half the list
    #recursion
    left, right = 0, len(lst) - 1

    if left <= right:
        middle = (left + right) / 2

        if val == middle:
            return middle
        if val < middle:
            return binary_search_recursive(lst[:middle], val)
        elif val > middle:
            return binary_search_recursive(lst[middle+1:], val)
    return -1