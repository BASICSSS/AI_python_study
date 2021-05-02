# Day_14_01_PythonWeather.py
import re
import requests


# f = open('data/weather.txt', 'w', encoding='utf-8')

url = 'https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=184'
received = requests.get(url)
# print(received)                 # <Response [200]>
# print(type(received))           # <class 'requests.models.Response'>

text = received.text
# print(text)
# print(type(text))               # <class 'str'>

# 문제
# city 태그 안쪽의 지역을 찾아주세요
# temp = re.findall(r'<city>제주</city>', text)
# temp = re.findall(r'<city>..</city>', text)
# temp = re.findall(r'<city>...</city>', text)
# temp = re.findall(r'.+', text)                    # 의미없다. 변별력 없음
# temp = re.findall(r'<city>.+</city>', text)
# print(temp)
# temp = re.findall(r'<city>(.+)</city>', text)
# print(temp)

# <city>.+</city>
# <city>제주</city>
# <city>서귀포</city>
# <city>성산</city>
# <city>성판악</city>
# <city>고산</city>
# <city>이어도</city>
# <city>추자도</city>

# 문제
# <body> 태그 안쪽에 들어있는 커다란 분량의 문자열을 검색(추출)하세요
# re.DOTALL: 개행 문자를 일반 문자로 변환. 검색 패턴이 여러 줄에 걸쳐 있을 때 사용
body = re.findall(r'<body>.+</body>', text, re.DOTALL)
# print(body)
# print(len(body))

# 문제
# body 태그 안쪽에서(검색한 내용으로부터) city 태그 안쪽의 지역을 찾아주세요
# 아래처럼 태그 단위로 검색하면 싱크(동기화)를 맞출 수 없다
# 맞출 수 있더라도 굉장히 피곤한 코딩을 수반한다. 이런 방법은 정신 건강에 해롭다
# t1 = re.findall(r'<province>(.+)</province>', body[0])
# t2 = re.findall(r'<city>(.+)</city>', body[0])
# t3 = re.findall(r'<mode>(.+)</mode>', body[0])
# t4 = re.findall(r'<tmEf>(.+)</tmEf>', body[0])
# t5 = re.findall(r'<wf>(.+)</wf>', body[0])
# print(t1)
# print(t2)
# print(t3)
# print(t4)
# print(t5)

# ['제주도', '제주도', '제주도', '제주도', '제주도', '제주도', '제주도']
# ['제주', '서귀포', '성산', '성판악', '고산', '이어도', '추자도']
# ['A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A02', 'A01', 'A01', 'A01']
# ['2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00', '2021-04-26 00:00', '2021-04-26 12:00', '2021-04-27 00:00', '2021-04-27 12:00', '2021-04-28 00:00', '2021-04-28 12:00', '2021-04-29 00:00', '2021-04-29 12:00', '2021-04-30 00:00', '2021-04-30 12:00', '2021-05-01 00:00', '2021-05-02 00:00', '2021-05-03 00:00']
# ['구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음', '구름많음', '구름많음', '구름많음', '흐림', '흐리고 비', '흐리고 비', '흐리고 비', '구름많음', '맑음', '맑음', '맑음', '맑음', '구름많음']

# 문제
# body 태그 안쪽에서(검색한 내용으로부터) location 태그를 찾아주세요
# .+ : 탐욕적(greedy)
# .+? : 비탐욕적(non-greedy)
# locations = re.findall(r'<location wl_ver="3">(.+)</location>', body[0], re.DOTALL)
# print(len(locations))
locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[0], re.DOTALL)
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[1], re.DOTALL)  # 반복문으로 변환
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[2], re.DOTALL)
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[3], re.DOTALL)
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[4], re.DOTALL)
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[5], re.DOTALL)
# locations = re.findall(r'<location wl_ver="3">(.+?)</location>', body[6], re.DOTALL)
# print(len(locations))
# print(locations[0])

# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>
# <location wl_ver="3">...</location>

# '3412^'에 '[0-9]+' 패턴 적용
# 3, 34, 341, 3412(선택), 3412^
# '3412^'에 '[0-9]+?' 패턴 적용
# 3(선택), 34, 341, 3412, 3412^

# 문제
# location 결과로부터 province 태그를 찾으세요
# 힌트: findall 함수를 몇 번 사용합니까? (반복문 사용)
# for i in range(len(locations)):
#     prov = re.findall(r'<province>(.+?)</province>', locations[i], re.DOTALL)
#     print(prov)

for loc in locations:
    # 한번밖에 나오지 않아도 '.+?' 패턴이 나을 수 있다. 찾는 즉시 멈추니까.
    # prov = re.findall(r'<province>(.+)</province>', loc)
    # city = re.findall(r'<city>(.+)</city>', loc)
    # print(prov[0], city[0])

    # 문제
    # 앞의 코드에서 province와 city를 한번에 찾을 수 있도록 수정하세요 (findall 함수 1회 사용)
    # prov_city = re.findall(r'<province>(.+)</province><city>(.+)</city>', loc, re.DOTALL)     # wrong
    prov_city = re.findall(r'<province>(.+)</province>.+<city>(.+)</city>', loc, re.DOTALL)
    # print(prov_city)        # [('제주도', '추자도')]
    # print(prov_city[0])     # ('제주도', '추자도')
    # print(prov_city[0][0], prov_city[0][1])
    # prov, city = prov_city[0]
    # print(prov, city)

    # <province>제주도</province>
    # <city>제주</city>

    # 문제
    # location 결과로부터 data 태그를 찾으세요
    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # print(len(data))
    # print(data[0])

    # 문제
    # data 결과로부터 mode, tmEf, wf, tmn, tmx, rnSt 태그를 찾으세요
    for datum in data:
        mode = re.findall(r'<mode>(.+)</mode>', datum)
        tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        wf = re.findall(r'<wf>(.+)</wf>', datum)
        tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
        tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
        rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)

        # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])
        print(prov_city[0][0], prov_city[0][1], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])

        # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0], file=f)
        # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0], file=f, sep=',')
        # base = '{},{},{},{},{},{},{},{}\n'
        # f.write(base.format(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0]))

# f.close()

# <mode>A02</mode>
# <tmEf>2021-04-26 00:00</tmEf>
# <wf>구름많음</wf>
# <tmn>13</tmn>
# <tmx>20</tmx>
# <reliability/>
# <rnSt>20</rnSt>

# 제주도 추자도
# A02 2021-04-26 00:00 구름많음 11 17 20
# A02 2021-04-26 12:00 구름많음 11 17 20
# A02 2021-04-27 00:00 구름많음 13 18 30
#     ===>
# 제주도 추자도 A02 2021-04-26 00:00 구름많음 11 17 20
# 제주도 추자도 A02 2021-04-26 12:00 구름많음 11 17 20
# 제주도 추자도 A02 2021-04-27 00:00 구름많음 13 18 30

# loc = locations[0]
# prov = re.findall(r'<province>(.+)</province>', loc)
# city = re.findall(r'<city>(.+)</city>', loc)
# # print(prov[0], city[0])
#
# # 문제
# # location 결과로부터 data 태그를 찾으세요
# data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
# # print(len(data))
# # print(data[0])
#
# # 문제
# # data 결과로부터 mode, tmEf, wf, tmn, tmx, rnSt 태그를 찾으세요
# datum = data[0]
# mode = re.findall(r'<mode>(.+)</mode>', datum)
# tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
# wf = re.findall(r'<wf>(.+)</wf>', datum)
# tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
# tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
# rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)
#
# print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])


# 문제
# 기상청에서 가져온 데이터를 파싱해서 weather.txt 파일로 저장하세요







