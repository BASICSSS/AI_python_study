# Day_12_02_PythonList.py
import random

# 차원
# 1. 반복문 사용 횟수
# 2. 최종 데이터에 접근할 때 필요한 인덱스 갯수

a = [[0, 1],
     [2, 3, 4],
     [5, 6, 7, 8]]

print(a)
print(a[0], a[1], a[2])
print(a[0][0], a[1][0], a[2][0])

print(len(a))
print(len(a[0]), len(a[1]), len(a[2]))
print()

# 문제
# 2차원 리스트에 포함된 요소 갯수를 구하세요
s = 0
for i in range(len(a)):
    # print(len(a[i]))
    s += len(a[i])

print(s)

s = 0
for i in a:
    # print(len(i))
    s += len(i)

print(s)

s = 0
for i in a:
    for _ in i:
        s += 1

print(s)
print('-' * 30)

print(random.randrange(0, 100, 1))
print(random.randrange(0, 100))
print(random.randrange(100))

# 문제
# 앞에서 만든 2차원 리스트를 100보다 작은 홀수 난수로 채우세요
# for row in a:
#     for col in row:
#         print(col)
#         col = 99
#     # row[0] = random.randrange(1, 100, 2)

# 첫 번째
# for row in a:
#     for j in range(len(row)):
#         row[j] = random.randrange(1, 100, 2)

# 두 번째
for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] = random.randrange(1, 100, 2)

print(a)
print('-' * 30)

# 리스트(a, row), 정수(col)
# col = 99
# row = 99        # 리스트 => 정수
# row[0] = 99

# 문제
# 2차원 리스트에서 가장 큰 값을 찾으세요
m = a[0][0]
for row in a:
    for col in row:
        # print(col)
        if col > m:
            m = col

print()
print(m)
print('-' * 30)

# 문제
# 2차원 리스트를 새로운 1차원 리스트에 치환(생성)하세요
# [[25, 95], [77, 95, 3], [75, 71, 83, 77]]
# => [25, 95, 77, 95, 3, 75, 71, 83, 77]

b = []
# for row in a:
#     for col in row:
#         b.append(col)

# for row in a:
#     b.append(row)         # wrong

for row in a:
    # b.extend(row)         # right
    b += row

print(b)

# b.append([10, 20, 30])
# print(b)
#
# b.extend([10, 20, 30])
# print(b)
#
# b += [10, 20, 30]
# print(b)

import itertools
print(list(itertools.chain(*a)))
print(list(itertools.chain(a[0], a[1], a[2])))

# print([10, 20, 30])
# print(*[10, 20, 30])        # unpack (force)
# print(10, 20, 30)
#
# a1, a2, a3 = [10, 20, 30]   # unpack
# print(a1, a2, a3)
