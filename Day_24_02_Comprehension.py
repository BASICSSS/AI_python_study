# Day_24_02_Comprehension.py

url = 'www.naver.com'

for c in url:
    print(c, end=' ')
print()

# 문제
# url로부터 중복되지 않는 글자로 구성된 리스트를 만드세요
print(sorted({c for c in url}))
print(''.join(sorted({c for c in url})))
print()

for i, c in enumerate(url):
    print(i, c)
print()

print({i: c for i, c in enumerate(url)})    # {0: 'w', 1: 'w', 2: 'w', 3: '.', 4: 'n', ...}
print({c: i for i, c in enumerate(url)})    # {'w': 2, '.': 9, 'n': 4, ...}

print(len({i: c for i, c in enumerate(url)}))   # 13
print(len({c: i for i, c in enumerate(url)}))   # 10
print()


# 문제
# 1 ~ 10000 사이에 들어있는 8의 갯수를 구하세요 (구글 입사문제)
# 808 -> 2
def count_8(s):
    return sum([c == '8' for c in s])


print(sum([str(i).count('8') for i in range(10000)]))
print(str(list(range(10000))).count('8'))
print(sum([count_8(str(i)) for i in range(10000)]))
print(sum([sum([c == '8' for c in str(i)]) for i in range(10000)]))
