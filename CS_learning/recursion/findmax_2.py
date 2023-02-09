def FindMax(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    if n == 2:
        return two_max(arr)
    m = int(n/2)
    left = arr[:m]
    right = arr[m:]
    return FindMax([FindMax(left), FindMax(right)])