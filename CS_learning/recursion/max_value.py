def FindMax(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    if n == 2:
        return two_max(arr)
    m = int(n/2)
    return FindMax([FindMax(arr[:m]),FindMax(arr[m:])])

def FindMax2(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    return two_max([arr[-1], FindMax2(arr[:-1])])

def two_max(arr):
    if arr[0] >= arr[1]:
        return arr[0]
    else:
        return arr[1]



if __name__ == "__main__":
    a = list(range(100))
    print(FindMax(a))