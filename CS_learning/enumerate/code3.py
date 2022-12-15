hash_map = {}
for x in a:
    if hash_map.get(0 - x, None):
        res += 1
    else:
        hash_map[x] = True
