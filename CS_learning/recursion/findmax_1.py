def FindMax(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    return two_max([arr[-1], FindMax(arr[:-1])])