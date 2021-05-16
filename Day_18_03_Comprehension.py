# Day_18_03_Comprehension.py
import random

# 컴프리헨션: (함수 매개 변수로 전달할) 컬렉션을 구성하는 한 줄짜리 반복문


for i in range(5):
    i

for _ in range(10):
    random.randrange(100)

print([i for i in range(5)])
print((i for i in range(5)))
print(tuple(i for i in range(5)))
print({i for i in range(5)})

# 문제
# 20보다 작은 양수 중에서 짝수로 구성된 리스트를 만드세요
print([i for i in range(2, 20, 2)])

# 문제
# 0~100 사이의 난수 10개로 구성된 리스트를 만드세요
numbers = [random.randrange(100) for _ in range(10)]
print(numbers)

# 문제
# 난수 리스트로부터 홀수로만 구성된 리스트를 만드세요
for i in numbers:
    if i % 2:
        print(i)

print([i for i in numbers if i % 2])
