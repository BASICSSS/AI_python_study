# Day_18_02_ReLibrary.py
import requests
import re


# 문제
# 수지도서관 제1열람실의 빈 좌석 갯수를 알려주세요
url = "http://211.251.214.168:8080/seatmate_sj/seatmate.php?classInfo=1"

# received = requests.get(url)

# text = received.content.decode("euc-kr")

# a = re.findall(r"수리'", text)
# a = re.findall(r"수리'", text)
# b = re.findall(r"배정'", text)
# c = re.findall(r"<li class='txt2'>([0-9]+)", text)

# print(type(*c))
# print(int(*c) - len(a) - len(b))      내방식!!!!!!


received = requests.get(url)
text = received.content.decode("euc-kr")
# print(text)

# empty = re.findall(r'<li class='txt4'>14 </li>', text)        # 작은 따옴표 때문에 에러
empty = re.findall(r"<li class='txt4'>([0-9]+) </li>", text)
print("빈 좌석 갯수 :", int(empty[0]))

empty = re.findall(r'color: #003366; ">([0-9]+)</div></div>', text)
print("빈 좌석 갯수 :", len(empty))

empty = re.findall(r"좌석번호:([0-9]+)", text)
print("빈 좌석 갯수 :", len(empty))

empty = re.findall(r"alt=.좌석번호:([0-9]+).>", text)
print("빈 좌석 갯수 :", len(empty))

repair = re.findall(r"좌석번호:수리", text)
assign = re.findall(r"좌석번호:배정", text)
print("빈 좌석 갯수 :", 160 - len(repair) - len(assign))
