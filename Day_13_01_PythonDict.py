# Day_13_01_PythonDict.py
import random

a = [2, 1, 3] * 5
print(a)

b = []
for i in a:
    if i not in b:
        b.append(i)
b.sort()
print(b)

c = set(a)
print(c)

c = sorted(c)
print(c)

c = {}
print(type(c))

c = set()               # <class 'set'>
print(type(c))

for _ in range(100):
    c.add(random.randrange(10))

print(c)
print('-' * 30)

# 순서: 리스트, 튜플
# 검색: 셋, 딕셔너리

# dict: key를 이용해서 value를 찾는 자료구조
# 영한 사전: 영어 단어(key)를 찾으면 한글 설명(value)이 나옴

d = {}
print(type(d))          # <class 'dict'>

d['age'] = 25
d['addr'] = '경기도'
print(d)                # {'age': 25, 'addr': '경기도'}

d = {'age': 25, 'addr': '경기도'}

d['age'] = 26
d['hobby'] = 'badminton'

print(d)
print()

# print(d.keys())     # dict_keys(['age', 'addr', 'hobby'])
print(d['age'])
print(d.values())
print(d.items())
print()

for k in d.keys():
    print(k, d[k])
print()

for k in d:
    print(k, d[k])
print()

for i in d.items():
    print(i, i[0], i[1])
print()

for k, v in d.items():
    print(k, v)
print()

# enumerate: 루프 인덱스 제공
for i, k in enumerate(d):
    print(i, k, d[k])
print()

# 문제
# 딕셔너리의 items 함수에 대해 enumerate 함수를 적용하세요
for ikv in enumerate(d.items()):
    print(ikv, ikv[0], ikv[1][0], ikv[1][1])
print()

for i, kv in enumerate(d.items()):
    print(i, kv[0], kv[1])
print()

for i, (k, v) in enumerate(d.items()):
    print(i, k, v)

# i, (k, v) = 0, ('age', 26)
