import numpy as np
import itertools

# ??
def pairwise(iterable): # "ABCDE" -> AB BC CD DE ..
    a,b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

result = np.linspace(0, 10, num = 4)
print(result)

for start, end in pairwise(result):
    print("start : ", start, "end : ", end)
    value = np.random.uniform(start, end)
    print("value", value)

