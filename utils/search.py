from typing import List


def strict_binary_search(lst: List[int], val: int) -> int:
    """
    Strict binary search, val should be exist in lst
    Return the index of val in lst
    """
    left, right = 0, len(lst) - 1

    while left < right:
        mid = (left + right) // 2

        if lst[mid] < val:
            left = mid + 1
        else:
            right = mid

    return left
