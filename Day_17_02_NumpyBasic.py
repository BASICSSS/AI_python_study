# Day_17_02_NumpyBasic.py
import numpy as np


print(np.arange(10))
print(np.arange(0, 10))
print(np.arange(0, 10, 1))

# 배열: 같은 공간(메모리), 같은 자료형만 저장
# 리스트: 같은 공간, 다른 자료형도 가능
# ndarray: N-dimensional array(다차원 배열)
print(type(np.arange(10)))      # <class 'numpy.ndarray'>

print([1, 3, 3.1, 'python'])    # 다른 자료형을 묶었다
print(list(range(10)))
print('-' * 30)

a = np.arange(12)
print(a)
# a.shape = 99                  # 변경 불가
print(a.shape, a.dtype, a.size, a.ndim)     # (12,) int64 12 1
print(type(a.shape), type(a[0]))            # <class 'tuple'> <class 'numpy.int64'>
print()

# b = a.reshape(3, 4)
# b = a.reshape(4, 3)
# b = a.reshape(4, -1)        # 3
b = a.reshape(3, -1)        # 4
# b = a.reshape(2, 6)
# b = a.reshape(1, 12)
# b = a.reshape(5, 7)       # 에러. 약수만 가능
print(b)
print(b.shape, b.dtype, b.size, b.ndim)     # (3, 4) int64 12 2
print()

# 문제
# 3차원으로 변환하세요 (3가지 코드로 구현)
c = a.reshape(3, 2, 2)
# c = a.reshape(3, 1, 4)
# c = a.reshape(3, -1, 2)
# c = a.reshape(-1, -1, 2)  # 에러. 나누어 떨어지지 않음
print(c)
print(c.shape, c.dtype, c.size, c.ndim)     # (3, 2, 2) int64 12 3
print()

# 문제
# 이차원 배열 변수인 b를 1차원으로 변환하는 3가지 코드를 만드세요
print(b.reshape(12))
print(b.reshape(-1))
print(b.reshape(b.size))
