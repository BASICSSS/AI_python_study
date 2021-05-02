# Day_13_03_PythonLambda.py

# 람다: 반환값을 갖는 한 줄짜리 함수

def twice(a):
    return a * 2


def proxy(f, value):
    return f(value)


l = lambda a: a * 2
f = twice


print(twice(3))
print(twice)
print(f)
print(f(3))
print(l(3))
print((lambda a: a * 2)(3))

print(proxy(twice, 3))
print(twice(3))
print('-' * 30)

b = [39, 71, 26, 58]

print(sorted(b))
print(b)
print(sorted(b, key=lambda n: n % 10))
print('-' * 30)

# 문제
# 문자열 리스트를 길이에 맞게 역순으로 정렬하세요
c = ['YELLOW', 'RED', 'blue', 'Green']
print(sorted(c))
print(sorted(c, key=lambda s: -len(s)))
print(sorted(c, key=lambda s: len(s), reverse=True))

print(sorted(c, key=lambda s: s.lower()))
