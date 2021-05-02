# Day_11_02_PythonFunction.py


# 교수 => 데이터 => 학생 : 매개 변수(parameter, argument)
# 교수 <= 데이터 <= 학생 : 반환값


# 매개 변수 없고, 반환값 없고.
def f_1():
    print('f_1')


f_1()


# 매개 변수 있고, 반환값 없고.
def f_2(a, b):
    print('f_2', a + b)


# 문제
# f_2를 호출하는 두 가지 방법을 구현하세요
f_2(12, 34)
f_2('12', '34')


# 매개 변수 없고, 반환값 있고.
def f_3():
    a, b = 12, 34
    print('f_3', a + b)
    # 기본 반환값: None


c = f_3()
print(c)


def f_4():
    a, b = 12, 34
    print('f_4', a + b)

    return a + b


c = f_4()
print(c)
print(f_4())


# 매개 변수 있고, 반환값 있고.
def f_5(a, b):
    c = a + b
    # print('f_5', c)

    return c


print(f_5(3, 5))
print(f_5(34, 57))
print('-' * 30)


# 문제
# 2개의 정수로부터 큰 숫자를 찾는 함수를 만드세요
def max_2(a, b):
    # if a > b:
    #     return a
    # else:
    #     return b

    if a > b:
        return a

    return b

    # if a > b:
    #     b = a
    #
    # return b


print(max_2(3, 7))
print(max_2(7, 3))
print(max_2(7, 7))
print()


# 문제
# 4개의 정수로부터 큰 숫자를 찾는 함수를 만드세요
def max_4(a, b, c, d):
    # if a >= b and a >= c and a >= d: return a
    # if b >= a and b >= c and b >= d: return b
    # if c >= a and c >= b and c >= d: return c
    #
    # return d

    # if a < b:   a = b
    # if a < c:   a = c
    # if a < d:   a = d
    #
    # return a

    # 추가 문제
    # 앞에서 만들었던 max_2 함수를 사용해서 다시 풀어보세요
    # 복면가왕
    # return max_2(max_2(a, b), max_2(c, d))

    # 한국시리즈
    return max_2(max_2(max_2(a, b), c), d)


print(max_4(1, 3, 5, 7))
print(max_4(3, 5, 7, 1))
print(max_4(5, 7, 1, 3))
print(max_4(7, 1, 3, 5))

# print(max([7, 1, 3, 5]))
