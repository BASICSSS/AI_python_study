# Day_18_01_NumpyBasic.py
import numpy as np

print([1, 3, 5])
print(np.array([1, 3, 5]))

# 문제
# 넘파이 배열을 리스트로 변환하세요
print(list(np.array([1, 3, 5])))
print(list(np.int32([1, 3, 5])))
print()

# 문제
# 리스트를 사용해서 2행 3열의 넘파이 배열을 만드세요 (3가지 방법)
# (코드에 리스트만 들어있으면 됩니다)
print(np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3))
# print(np.array([[1, 3, 5], [0, 2, 4, 8]]))      # [list([1, 3, 5]) list([0, 2, 4, 8])]
print(np.array([[1, 3, 5], [0, 2, 4]]))
print(np.array([np.arange(3), np.arange(4, 7)]))
print(np.array([range(3), range(4, 7)]))

print(np.array(range(6)))
print("-" * 30)

print(np.zeros([2, 5]).dtype)  # float64
print(np.zeros([2, 5]))
print(np.ones([2, 5]))
print(np.full([2, 5], -1))
print()

# 문제
# a와 동일한 모양의 넘파이 배열을 만드세요 (zeros 사용)
a = [[1, 2, 3], [4, 5, 6]]
print(np.zeros_like(a))
print(np.zeros(np.array(a).shape, dtype=np.int32))
print(np.ones_like(a))
print(np.full_like(a, -1))
print("-" * 30)

print(np.arange(0, 3, 1))
print(np.arange(0, 1, 0.1))
print(np.arange(0, 1.1, 0.1))

print(np.linspace(0, 1, 10))
print(np.linspace(0, 1, 11))

# 0 2 4 6
#  ^ ^ ^
print("-" * 30)

b = np.arange(10)
print(b)

# for i in b:
#     print(i + 1)

# for i in range(len(b)):
#     b[i] += 7
# print(b)

# broadcast 연산
# 장점: 입력이 줄어든다. 가독성이 좋다. 성능이 좋다. 어려운 코드를 대신해 준다.
print(b + 7)
print(b * 7)
print(b ** 2)
print(b > 5)
print(np.logical_and(b > 3, b < 7))
print()

c = b.reshape(2, 5)
print(c + 7)
print(c * 7)
print(c ** 2)
print(c > 5)
print(np.logical_and(c > 3, c < 7))
print("-" * 30)

# vector 연산
print(b + b)
print(b * b)
print(b > b)
print()

print(c + c)
print(c * c)
print(c > c)

# print(b + [1, 2, 3])              # 에러
# print(b + np.array([1, 2, 3]))    # 에러
print(b + list(range(10)))  # 성공
print(b + range(10))  # 성공
print("-" * 30)

# universal function
print(np.sin(b))
print(np.cos(b))
