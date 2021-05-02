# Day_04_01_PythonOperator.py

# 연산 : 산술, 관계, 논리

# 산술 : +  -  *  //  %  **  /
a, b = 7, 3

print(a + b)
print(a - b)
print(a * b)
print(a // b)       # 나눗셈 (정수) : 몫 연산
print(a % b)        # 나머지
print(a ** b)       # 지수
print(a / b)        # 나눗셈 (실수)
print()

#     2             # //
#   +---
# 3 | 7
#     6
#    ---
#     1             # %

# 문제
# 변수 c를 반올림한 값을 구하세요
c = 3.14    # 3
c = 2.71    # 3

print(c // 1)
print((c + 0.5) // 1)

print(int(c))
print(int(c + 0.5))         # str, int, float, bool
print()

# int('hello')              # 에러
# int('123')

# 관계 : >  >=  <  <=  ==  !=

print(a > b)        # 7 > 3
print(a >= b)
print(a < b)
print(a <= b)       # 7 <= 3
print(a == b)
print(a != b)

print(int(a != b))
print()

# 논리 : and  or  not

# T and T = T
# T and F = F
# F and T = F
# F and F = F

print(True and True)
print(True and False)
print(False and True)
print(False and False)
print()

print(not True)
