# Day_17_01_json.py
import json
import requests
import re


# 데이터 교환 형식: json, xml(html), yaml
# save: 메모리(단기 기억장치)에 있는 내용을 파일(영구 기억장치)에 쓰기
# load: 파일(영구 기억장치)에 있는 내용을 메모리(단기 기억장치)에 쓰기

t1 = '{"ip": "8.8.8.8"}'
print(type(t1))             # <class 'str'>

j1 = json.loads(t1)
print(j1)
print(type(j1))
print(j1['ip'])

print('-' * 30)

# 문제
# 아래 데이터를 json을 사용해서 문자열 형태로 변환하세요
t2 = {"ip": "8.8.8.8"}
print(type(t2))             # <class 'dict'>

j2 = json.dumps(t2)
print(j2)
print(type(j2))

print('-' * 30)

# 문제
# 아래 데이터로부터 key와 value를 뽑아서 출력하세요
date_and_time = '''{
   "time": "03:53:25 AM",
   "milliseconds_since_epoch": 1362196405309,
   "date": "03-02-2013"
}'''

j3 = json.loads(date_and_time)
print(j3)

for k in j3:
    print(k, j3[k])

print('-' * 30)

url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
received = requests.get(url)
print(received)
print(received.text)
print(received.content)
print(type(received.text))      # <class 'str'>
print(type(received.content))   # <class 'bytes'>

print('\xec :', 16 * 14 + 12)
# \xec

text = received.content.decode('utf-8')
print(text)
print(type(text))               # <class 'str'>

# 문제
# 기상청에서 가져온 지역 코드를 깔끔하게 출력하세요 (json)
# 11 서울특별시
# 26 부산광역시

data = json.loads(text)
print(data)
print(type(data))

for item in data:
    # print(type(item), item)
    # for k in item:
    #     print(item[k], end=' ')
    # print()

    # print(item['code'], item['value'])

    # print(item.values())        # dict_values(['11', '서울특별시'])
    print(*item.values())         # 11 서울특별시

print('-' * 30)

# 문제
# 기상청에서 가져온 지역 코드를 깔끔하게 출력하세요 (정규표현식)
print(text)
print(type(text))
# [{"code":"11","value":"서울특별시"},{"code":"26","value":"부산광역시"},{"code":"27","value":"대구광역시"},{"code":"28","value":"인천광역시"},{"code":"29","value":"광주광역시"},{"code":"30","value":"대전광역시"},{"code":"31","value":"울산광역시"},{"code":"41","value":"경기도"},{"code":"42","value":"강원도"},{"code":"43","value":"충청북도"},{"code":"44","value":"충청남도"},{"code":"45","value":"전라북도"},{"code":"46","value":"전라남도"},{"code":"47","value":"경상북도"},{"code":"48","value":"경상남도"},{"code":"50","value":"제주특별자치도"}]

# codes = re.findall(r'[0-9]+', text)
# areas = re.findall(r'[가-힣]+', text)     # 좋은 코드는 아니라고 봄
codes = re.findall(r'"code":"([0-9]+)"', text)
areas = re.findall(r'"value":"(.+?)"', text)
print(codes)
print(areas)
print(len(areas))       # 16

print('hello' > 'world')    # False
print('한글' > '대한민국')      # True

print('-' * 30)

for i in range(len(codes)):
    print(codes[i], areas[i])

print(zip(codes, areas))        # <zip object at 0x7f9625c2e9b0>
print(list(zip(codes, areas)))  # [('11', '서울특별시'), ('26', '부산광역시'), ...

for code, area in zip(codes, areas):
    print(code, area)

print('-' * 30)

codes_and_areas = re.findall(r'{"code":"(.+?)","value":"(.+?)"}', text)
print(codes_and_areas)      # [('11', '서울특별시'), ('26', '부산광역시'), ...]

for code, area in codes_and_areas:
    print(code, area)
