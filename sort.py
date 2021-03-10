from random import randint

def bubble_sort(l):
    # average, worst-case O(n^2)
    # best-case O(n) (already sorted)

    # For each index, swap with the index to the right if greater.
    n = len(l)
    for i in range(n):
        yes = True #already sorted
        for j in range(n-i-1): # the end should be sorted each time
            if l[j] > l[j+1]: # if consecutive arent sorted, sort them
                l[j], l[j+1] = l[j+1], l[j]
                yes = False
            
        if yes:
            break
    return l

def insertion_sort(l, left=0, right=None):
    # average, worst-case O(n^2)
    # best-case O(n) (already sorted)

    # For each index starting from 1, push it to the left until it is bigger than the left
    if right is None:
        right = len(l) - 1
    for i in range(left+ 1, right+1):
        key = l[i]
        j = i - 1

        while j >= left and l[j] > key:
            l[j+1] = l[j]
            j -= 1
        l[j+1] = key

    return l

def merge(left, right):
    # Merge both halves
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    
    # Merge the two lists, sorted
    result = []
    index_left = index_right = 0
    while len(result) < len(left) + len(right):
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
        
        if index_right == len(right):
            result += left[index_left:]
            break
        if index_left == len(left):
            result += right[index_right:]
            break
    return result

def merge_sort(l):
    # Recursively split input in half
    # Merge both halves

    # O(nlogn)
    if len(l) < 2:
        return l
    
    midpoint = len(l) // 2

    return merge(
        left=merge_sort(l[:midpoint]),
        right=merge_sort(l[midpoint:]))

def quicksort(l):
    # Select random pivot
    # Make low, same, high lists from all x
    # quicksort low and high

    # O(n) when pivot is median
    # O(n^2) when pivot is smallest or largest
    # Trades memory space for speed; 
    if len(l) < 2:
        return l
    
    low, same, high = [], [], []
    pivot = l[randint(0, len(l) - 1)]
    for x in l:
        if x < pivot:
            low.append(x)
        if x == pivot:
            same.append(x)
        if x > pivot:
            high.append(x)
        
    return quicksort(low) + same + quicksort(high)

def timsort(l):
    # Create small slices, sort them using insertion sort (fast with small lists)
    # Merge runs recursively
    min_run = 32
    n = len(l)

    # Slice and sort small portions of input -> either min_run or size
    for i in range(0, n, min_run):
        insertion_sort(l, i, min((i+ min_run - 1), n-1))

    # Merge sorted slices
    size = min_run
    while size < n:
        # merge 2 lists:
        for start in range(0, n, size*2):
            midpoint = start + size - 1
            end = min((start+ size*2-1), (n-1))

            merged = merge(left=l[start:midpoint+1], right=l[midpoint+1: end+1])
            l[start:start+len(merged)] = merged

        size *= 2