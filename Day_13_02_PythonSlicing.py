# Day_13_02_PythonSlicing.py

a = [0, 1, 2, 3, 4, 5, 6]
print(a[len(a) - 1], a[len(a) - 2])
print(a[-1], a[-2])

print(a[0:3:1])         # 시작, 종료, 증감
print(a[:100])          # 범위를 벗어나도 안전(이렇게 사용하면 안됨)

# 문제
# 1. 리스트에서 짝수 번째 요소만 출력하세요
print(a[0:7:2])
print(a[::2])

# 2. 리스트에서 홀수 번째 요소만 출력하세요
print(a[1:7:2])
print(a[1::2])
print()

# 문제
# 1차원 리스트를 거꾸로 출력하세요
print(a[3])
print(a[3:4])
print(a[3:3])           # empty
print(a[7-1], a[-1])

print(a[7-1:0:-1])
print(a[7-1:-1:-1])
print(a[-1:-1:-1])
print(a[::-1])          # 증감(음수) : 마지막 --> 처음
print(a[::1])           # 증감(양수) : 처음 --> 마지막
