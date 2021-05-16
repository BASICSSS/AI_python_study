# Day_19_01_NumpyAdvanced.py
import numpy as np

# scalar + scalar : 3 + 7
# scalar + array  : 3 + [1, 3]          broadcast
# array  + array  : [1, 3] + [2, 5]     vector

a = np.arange(3)      # (1, 3)      # [0, 1, 2]
b = np.arange(6)      # (1, 6)      # [0, 1, 2, 3, 4, 5]
c = np.arange(3).reshape(1, 3)      # [[0, 1, 2]]
d = np.arange(6).reshape(2, 3)      # [[0, 1, 2], [3, 4, 5]]
e = np.arange(3).reshape(3, 1)      # [[0], [1], [2]]

# print(a + b)          # 에러
print(a + c)            # scalar + vector
print((a + c).shape)    # (1, 3)
print(a + d)            # broadcast + vector
print((a + d).shape)    # (2, 3)
print(a + e)            # broadcast + broadcast

# 문제
# 나머지 변수들에 대해서도 연산이 성립하는지 확인하세요 (6개의 연산)
# print(b + c)          # 에러
# print(b + d)          # 에러
print(b + e)            # broadcast + broadcast
print((b + e).shape)    # (3, 6)

print(c + d)            # broadcast + vector
print((c + d).shape)    # (2, 3)

# print(d + e)          # 에러
print('-' * 30)

np.random.seed(17)

print(np.random.random([2, 3]))         # 0 ~ 1
print(np.random.randn(2, 3))            # 표준 정규분포
print(np.random.uniform(size=[2, 3]))   # 균등분포
print(np.random.rand(2, 3))             # uniform 단순 버전
print()

print(np.random.choice(range(5), 10))

a = np.random.choice([1, 3, 7, 9], 12)
print(a)
print()

print(np.max(a))
print(np.sum(a))

print(a.max())
print(a.sum())
print('-' * 30)

# 문제
# np.reshape 함수를 사용해서 a를 2차원으로 변환하세요
# b = a.reshape(3, 4)
b = np.reshape(a, newshape=[3, 4])
print(b)
print()

print(b.max())
print(b.max(axis=0))            # 수직(열)
print(b.max(axis=1))            # 수평(행)
# print(b.max(axis=2))          # 에러
print('-' * 30)

c = np.int32([5, 2, 9, 6])
print(c)
print(np.sort(c))       # 정렬 결과 반환(파이썬: sorted)

print(c)
print(c.sort())         # 자기자신 정렬(파이썬: sort)

print(c)
