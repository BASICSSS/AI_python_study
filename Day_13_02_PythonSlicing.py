# Day_13_02_PythonSlicing.py

a = [0, 1, 2, 3, 4, 5, 6]
print(a[len(a) - 1], a[len(a) - 2])
print(a[-1], a[-2])

print(a[0:3:1])  # 시작, 종료, 증감
print(a[:100])  # 범위를 벗어나도 안전(이렇게 사용하면 안됨)

# 문제
# 1. 리스트에서 짝수 번째 요소만 출력하세요
print(a[0:7:2])
print(a[::1])
