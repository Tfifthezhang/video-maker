def Loop_binary(a, b):
    res = 1
    while b > 0:
        if (b & 1): # 按位与运算，判断是否为1
            res = res * a
        a = a * a
        b >>= 1 # 二进制位右移一位
    return res