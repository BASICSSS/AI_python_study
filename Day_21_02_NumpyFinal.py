# Day_21_02_NumpyFinal.py
import numpy as np


a = np.arange(12).reshape(3, 4)
print(a)

# 문제
# 첫 번째 int와 마지막 int를 출력하세요
print(a[0], a[0][0])
print(a[-1], a[-1][-1])
print(a[0, 0], a[-1, -1])           # fancy indexing
print('-' * 30)

# 문제
# 첫 번째 행을 출력하세요
# 첫 번째 열을 출력하세요
for i in a:
    print(i[0], end=' ')
print()

for i in range(len(a)):
    print(a[i, 0], end=' ')
print()

print(a[0, 0], a[1, 0], a[2, 0])

print(a[:, 0])              # 첫 번째 열 출력
print(a[0])
print(a[0, :])              # 첫 번째 행 출력
print('-' * 30)

# 문제
# 2차원 배열을 행과 열을 뒤집어서 출력하세요
print(a[::-1])
print(a[::-1, ::-1])
print()

print(a[::-1, :])
print(a[:, ::-1])
