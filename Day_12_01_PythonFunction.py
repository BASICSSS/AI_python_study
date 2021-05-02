# Day_12_01_PythonFunction.py


# a = (3, 7)
a = 3, 7
print(a)
print(a[0], a[1])

# a[0] = 99                 # 튜플은 수정 불가

b, c = 3, 7
print(b, c)

a = 3, 7
b, c = 3, 7
b, c = a
# b, c, d = a               # 변수 2개를 3개에 치환 불가


def f_6(a, b):
    return a + b, a * b


d = f_6(4, 9)
print(d)

d = 4 + 9, 4 * 9
print(d)

e, f = f_6(4, 9)
print(e, f)
print('-' * 30)


def f_7(a, b, c):
    print(a, b, c)


f_7(1, 2, 3)                # positional argument
f_7(a=1, b=2, c=3)          # keyword argument
# f_7(name='kim', user_id='abc12', order=True)

f_7(c=3, a=1, b=2)
f_7(1, 2, c=3)
# f_7(a=1, 2, c=3)          # positional은 keyword 앞에.
print('-' * 30)


def f_8(a=0, b=0, c=0):     # default argument
    print(a, b, c)


# 문제
# f_8 함수를 호출하는 3가지 코드를 만드세요
f_8()
f_8(1)
f_8(1, 2)
f_8(1, 2, 3)
f_8(a=1)
f_8(c=3)
f_8(a=1, c=3)
f_8(1, c=3)
f_8(1, b=2)
