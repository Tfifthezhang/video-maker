def Sum(a):
    if len(a) == 1:
        return a[0]
    return a[-1] + Sum(a[:-1])