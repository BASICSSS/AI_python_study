# Day_12_03_PythonRe.py
import re               # regular expression(정규표현식)

# ... abc 123
# (A)(BC)([ABC]D)\1\3
# (A)(BC)([ABC]D)(A)([ABC]D)        # wrong
# (A)(BC)BDABD                      # right
# \ : ()의 결과값
# \d: [0-9]                         # decimal

db = '''3412    [Bob] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''

# print(db)

print(re.findall(r'a', db))

# 문제
# db에 포함된 모든 숫자를 찾으세요
# r: raw
print(re.findall(r'[0123456789]', db))
print(re.findall(r'[0-9]', db))

# 문제
# db에 포함된 모든 숫자(전화번호, 아이디)를 찾으세요
print(re.findall(r'[0-9]+', db))
# 3412: 3, 34, 341, 3412

# 문제
# db에 포함된 전화번호만 찾으세요
print(re.findall(r'[0-9][0-9][0-9][0-9]', db))
print(re.findall(r'[0-9]{4}', db))

# 문제
# db에 포함된 이름만 찾아보세요
print(re.findall(r'[A-z]+', db))        # wrong
print(re.findall(r'[A-Za-z]+', db))
print(re.findall(r'[a-zA-Z]+', db))
print(re.findall(r'[A-Z][a-z]+', db))

# 문제
# 1. db에 포함된 이름 중에서 T로 시작하는 이름만 찾아보세요
print(re.findall(r'T[a-z]+', db))
print(re.findall(r'[T][a-z]+', db))

# 2. db에 포함된 이름 중에서 T로 시작하지 않는 이름만 찾아보세요
print(re.findall(r'[^T][a-z]+', db))    # wrong
print(re.findall(r'[^Ta-z][a-z]+', db))
print(re.findall(r'[ABCDEFGHIJKLMNOPQRSUVWXYZ][a-z]+', db))
print(re.findall(r'[A-SU-Z][a-z]+', db))

