import numpy as np

a = np.arange(3)
# b = np.arange(6)
c = np.arange(3).reshape(1, 3)
d = np.arange(6).reshape(2, 3)
e = np.arange(3).reshape(3, 1)


c = np.int32([5, 2, 3, 6])
print(c)


print(type(np.sort(c)))
print(type(c))
print((c.sort().shape))
print(c)
