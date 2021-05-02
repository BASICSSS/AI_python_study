# Day_04_02_PythonIf.py

a = 85.7

if a > 70.0:
    print('비가 온다')
else:
    print('비가 오지 않는다')

# 산술 > 관계 > 논리

# 문제
# 변수 b에 들어있는 값이 홀수면 홀수라고, 짝수면 짝수라고 출력하세요
b = 31
if b % 2 == 0:
    print('짝수')
else:
    print('홀수')

if b % 2 == 1:
    print('홀수')
else:
    print('짝수')

if b % 2:
    print('홀수')
else:
    print('짝수')

# 참 : 거짓이 아닌 모든 값, 1(대표 참값)
# 거짓 : 0

# 문제
# 아래 코드가 짝수를 출력하도록 b의 값을 수정하세요
# b = 0
# if b:
#     print('홀수')
# else:
#     print('짝수')

print(type("hello"))
print(type(3.14))

c = input('정수를 입력하세요 : ')
print(c)
print(type(c))

c = int(c)

# 문제
# input 함수로 입력 받은 정수가 음수인지 양수인지 0인지 알려주세요
if c > 0:
    print('양수')
else:
    # print('음수, 제로')
    if c < 0:
        print('음수')
    else:
        print('제로')

if c > 0:
    print('양수')
elif c < 0:
    print('음수')
else:
    print('제로')
