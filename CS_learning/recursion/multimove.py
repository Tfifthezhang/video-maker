def MultiMove(n, a, b, c):
    if n >= 1:
        MultiMove(n - 1, a, c, b)
        SingleMove(a, c)
        MultiMove(n - 1, b, a, c)
