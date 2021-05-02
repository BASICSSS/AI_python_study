# Day_19_03_ComprehensionAdvanced.py


for i in range(2):
    for j in range(3):
        print(i, j)


# 문제
# 앞에서 만든 2차원 반복문을 컴프리헨션으로 변환하세요 (리스트 요소는 튜플)
print([(i, j) for i in range(2) for j in range(3)])

# 문제
# 2차원 리스트를 1차원 리스트로 변환하세요
a = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]

for i in a:
    # print(i)
    for j in i:
        print(j, end=" ")
print()
print([j for i in a for j in i])
print()

# 문제
# 2차원 배열의 홀수 합계를 구하세요 (2가지)
for i in a:
    for j in i:
        if j % 2:
            print(j, end=" ")
print()

print([j for i in a for j in i if j % 2])
print(sum([j for i in a for j in i if j % 2]))

print([sum(i) for i in a])
print([[0] for i in a])
print([[0 for j in i] for i in a])
print([[j for j in i] for i in a])
print([[j for j in i if j % 2] for i in a])
print([sum([j for j in i if j % 2]) for i in a])
print(sum([sum([j for j in i if j % 2]) for i in a]))


class Fake:
    def __init__(self):
        print("ss")


# print(a)
