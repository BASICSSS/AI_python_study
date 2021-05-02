# Day_13_05_PythonDecorator.py
import time


def f_7(a, b):
    return a + b, a * b     # pack


t1, t2 = f_7(3, 5)


def f_8(*args):             # pack, 가변 매개 변수
    # print(args, type(args))
    print(*args)            # unpack (force), itertools.chain()


f_8()
f_8(1)
f_8(1, 2)
f_8(1, 2, 'hello')

print([1, 3, 5])
print(*[1, 3, 5])           # unpack (force)
print(1, 3, 5)


def f_9(**kwargs):          # keyword 가변 매개 변수
    print(kwargs, type(kwargs))


# 문제
# f_9를 호출하는 3가지 코드를 만드세요
f_9()
f_9(a=1)
f_9(a=1, b='hello')
print('-' * 30)


def exp_1(n, times):
    result = 1
    for _ in range(times):
        result *= n

    return result


def decorator(*args, **kwargs):
    print(args, kwargs)
    result = exp_1(*args, **kwargs)
    return result


print(exp_1(3, 2))
print(exp_1(n=5, times=3))

print(decorator(3, 2))
print(decorator(n=5, times=3))
print('-' * 30)


def how_long(func):
    def decorator(*args, **kwargs):
        start = time.time()
        for _ in range(100000):
            result = func(*args, **kwargs)
        print('소요 시간 :', time.time() - start)
        return result

    return decorator


what = how_long(exp_1)
print(what(3, 2))
print(what(n=5, times=3))

wow = how_long(f_7)
print(wow(3, 5))
print(wow(a=3, b=5))
print('-' * 30)


@how_long
def exp_2(n, times):
    result = 1
    for _ in range(times):
        result *= n

    return result


print(exp_2(3, 2))
print(exp_2(n=5, times=3))
