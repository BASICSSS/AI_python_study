# Day_24_01_ReOpenHangul.py
import re
import requests


# 오픈한글
# http://openhangul.com/

# 문제
# 오픈한글 사이트에서 한글을 입력하면 키보드에 있는 영문을 알려주는 페이지를 파싱하세요
# 한글 -> gksrmf

# 'http://openhangul.com/nlp_ko2en?q=한글'

#            함수                 ?         매개변수
# https://www.google.com/search  ?   q=오픈한글 & oq=오픈한글

kor = '꼬깔콘'
url = 'http://openhangul.com/nlp_ko2en?q={}'.format(kor)    # get 방식
received = requests.get(url)
# print(received)
# print(received.text)

# text = received.content.decode('utf-8')
# print(text)

result = re.findall(r'<img src="images/cursor.gif"><br>(.+?)</pre>', received.text, re.DOTALL)
print(result)
print('{} : {}'.format(kor, result[0].strip()))

result = re.findall(r'<img src="images/cursor.gif"><br>(.+)', received.text)
print(result)
print('{} : {}'.format(kor, result[0].strip()))
