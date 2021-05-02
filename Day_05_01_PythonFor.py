# Day_05_01_PythonFor.py

# print('sunshine')
# print('sunshine')
# print('sunshine')
# print('sunshine')
# print('sunshine')

# 0, 1, 2, 3, 4
for i in range(0, 5, 1):        # 시작, 종료, 증감
    print('sunshine', i)
print()

for i in range(0, 5):           # 시작, 종료, 증감(1)
    print('sunshine', i)
print()

for i in range(5):              # 시작(0), 종료, 증감(1)
    print('sunshine', i)
print()

# 문제
# 1. 100보다 작은 양수에서 짝수만 골라서 출력하세요
#    2, 4, 6, ... 98
for i in range(2, 10, 2):
    print(i, end=' ')
print()

# 2. 5, 4, 3, 2, 1의 순서로 출력하세요
for i in range(5, 0, -1):
    print(i, end=' ')
print()
