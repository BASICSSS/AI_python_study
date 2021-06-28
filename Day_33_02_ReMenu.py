# Day_33_02_ReMenu.py
import re
import requests

# 문제
# 전북대학교에 있는 진수원의 메뉴를 파싱해서 출력하세요

url = 'http://sobi.chonbuk.ac.kr/chonbuk/m040101'
received = requests.get(url)
text = received.content.decode('utf-8')
# print(text)

# 1번
tables = re.findall(r'<table.+?</table>', text, re.DOTALL)
print(tables[0])

results = re.findall(r'<p style="line-height: 1.2;">(.+?)</p>', tables[0])
print(len(results))
print(*results, sep='\n')
print()

print('점심')
for row in results[2:7]:
    print(row.split('<br />'))

print('저녁')
for row in results[9:]:
    print(row.split('<br />'))
print('-' * 30)

# 2번
results = re.findall(r'<p style="line-height: 1.2;">(....+?)</p>', text)
print(len(results))
print(*results, sep='\n')
print()
print('점심/저녁')
for row in results:
    # for item in row.split('<br />'):
    #     print(item.replace('&amp;', '&'), end=' : ')
    # print()
    print([item.replace('&amp;', '&') for item in row.split('<br />')])
print('-' * 30)

# 3번
results = re.findall(r'<td bgcolor="#ffffff" class="">.*?<p style="line-height: 1.2;">(.+?)</p>', tables[0], re.DOTALL)
print(len(results))
print(*results, sep='\n')
print()

# 4번
results = re.findall(r'<p style="line-height: 1.2;">(.+?)</p>', text)
print(len(results))
print(*results, sep='\n')
print()

# 5번 - 1번 코드 복사
tables = re.findall(r'<table.+?</table>', text, re.DOTALL)

menu_text = tables[0].replace('&amp;', '&')
results = re.findall(r'<p style="line-height: 1.2;">(.+?)</p>', menu_text)

print('점심')
for row in results[2:7]:
    print(row.split('<br />'))

print('저녁')
for row in results[9:]:
    print(row.split('<br />'))
