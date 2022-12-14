for x in range(0, n):
    for y in range(x+1, n):
        if a[x] + a[y] == 0:
            res = res + 1