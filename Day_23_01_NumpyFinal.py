# Day_23_01_NumpyFinal.py
import numpy as np

a = np.arange(12) * 2
print(a)
print()

print(a[0], a[3], a[7])     # 0 6 14
b = [0, 3, 7]
print(a[b])                 # 인덱스 배열, [ 0  6 14]
print(a[[0, 3, 7]])
print()

c = [[2, 9], [3, 5]]
# print(a[c])               # 에러
d = np.int32(c)
print(a[d])                 # [[ 4 18] [ 6 10]]
print(a[d.reshape(-1)])
print(a[d.reshape(-1)].reshape(2, 2))
print('-' * 30)

e = a.reshape(3, 4)
print(e)
print()

print(e[0], e[2], e[1])
print(e[[0, 2, 1]])
print()

f = [[0, 1], [2, 3]]
# print(e[f])               # [ 4 14], deprecated
print(e[[0, 1], [2, 3]])    # [ 4 14], fancy indexing
print(e[(0, 1), (2, 3)])    # [ 4 14]
print()

# 팬시 인덱싱: 정수, 리스트(배열), 슬라이싱
print(e[0, (2, 3)])         # [4 6]
print(e[1:, (2, 3)])        # [[12 14] [20 22]]
print(e[1:, 2:])            # [[12 14] [20 22]]
print('-' * 30)

# 문제
# 테두리는 1로, 속은 0으로 채워진 5x5 배열을 만드세요 (zeros 사용)
g = np.zeros([5, 5], dtype=np.int32)

# g[0], g[4] = 1, 1
# g[(0, -1)] = 1        # 0행 -1열을 가리킴
g[[0, -1]] = 1          # 반드시 [] 사용할 것

# g[:, 0], g[:, -1] = 1, 1
g[:, [0, -1]] = 1

print(g)
print()

# 문제
# 앞에서 만든 5x5 2차원 배열에 대해 대각선 양쪽(x자 형태)으로 3을 넣어주세요
# g[0, 0], g[1, 1] = 3, 3
# g[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 3
# g[[0, 1, 2, 3, 4], [4-0, 4-1, 4-2, 4-3, 4-4]] = 3
# g[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0] = 3

# h = np.arange(g.shape[0])
# g[h, h] = 3
# # g[h, 4-h] = 3
# g[h, h[::-1]] = 3

g[range(5), range(5)] = 3
# g[range(5), reversed(range(5))] = 3       # 에러
# g[range(5), list(reversed(range(5)))] = 3
g[range(5), range(5-1, -1, -1)] = 3
print()

print(np.eye(5, dtype=np.int32))
print(np.identity(5, dtype=np.int32))
print(np.identity(5, dtype=np.int32)[:, ::-1])
exit()
print(g)
print('-' * 30)

print(a)        # [ 0  2  4  6  8 10 12 14 16 18 20 22]
print(a > 7)
print(a[a > 7])     # 참인 경우만 필터링

print(e)
# [[ 0  2  4  6]
#  [ 8 10 12 14]
#  [16 18 20 22]]

print(e > 7)
print(e[e > 7])     # [ 8 10 12 14 16 18 20 22]
print()

e[e > 7] = 99
print(e)
print('-' * 30)

np.random.seed(1)

# k = np.random.random_integers(0, 100, 10)     # deprecated (100 포함)
k = np.random.randint(0, 100, 12)
print(k)
print(np.max(k))            # 79
print(np.argmax(k))         # 6, 가장 큰 값의 위치
print(k[np.argmax(k)])      # 79
print()

m = k.reshape(3, 4)
print(m)

print(np.argmax(m))
print(np.argmax(m, axis=0))
print(np.argmax(m, axis=1))
print()

print(np.sort(k))

# 문제
# np.argsort 함수를 사용해서 정렬된 결과를 출력하세요
print(np.argsort(k))
print(k[np.argsort(k)])
print('-' * 30)

t = [1, 0, 2, 3, 0, 2]
print(np.nonzero(t))        # (array([0, 2, 3, 5]),)

w = np.int32(t)
print(w[np.nonzero(w)])     # [1 2 3 2]
print()


v = np.reshape(t, [2, 3])
print(v)                    # [[1 0 2] [3 0 2]]
print(np.nonzero(v))        # (array([0, 0, 1, 1]), array([0, 2, 0, 2]))
print(v[np.nonzero(v)])     # [1 2 3 2]
print()
