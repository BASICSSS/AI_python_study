# Day_02_01_PythonBasic.py

# ctrl + shift + f10 / ctrl + shift + r
# alt + 1 / cmd + 1
# alt + 4 / cmd + 4

# 문제
# 아래쪽에 있는 코드 한 줄을 5개로 복사해서 붙여 넣으세요
# ctrl + c + v + v + v + v + v + v

# 문제
# 한 줄만 남기고 나머지 줄을 삭제하세요
# ctrl + x + x + x + x

# 프로그래밍 : 코드(함수), 데이터(변수, 상수)
print('hello, python!!')        # 문자열(str)
print(123, 3.14, True)          # 정수(int), 실수(float), 논리값(bool)

a = 3.14
print(a)

a = 2.71
print(a)

# 에러. 상수는 변경할 수 없다 (대입 연산 왼쪽에 올 수 없다)
# 3.14 = 2.71

# 문제
# 문자열, 정수 변수를 1개씩 만드세요
b, c = 'hello', 456
print(b, c)

# 변수 -> 연산자 -> 제어문 -> 반복문 -> 컬렉션(리스트) -> 함수

# 문제
# 변수 i와 j에 들어있는 값을 교환하세요
i, j = 3, 7
print(i, j)

# 코드가 들어갈 위치
# i, j = 7, 3           # bug
i, j = j, i             # error

# t = i
# i = j
# j = t

print(i, j)
