def Recur_binary(a, b):
    if b == 0:
        return 1
    res = Recur_binary(a, b // 2) # 递归获取二进制的位
    if (b % 2) == 1: # 判断是否为1
        return res * res * a
    else:
        return res * res