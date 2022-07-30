def main():
    m,k = list(map(int,input().split(' ')))
    mod = 100000007
    res = 1
    for i in range(k):
        res = (res*m) % mod
        m = (m*m) % mod 
    print(res)