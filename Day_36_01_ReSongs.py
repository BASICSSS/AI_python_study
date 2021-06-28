# Day_36_01_ReSongs.py
import re
import requests

# HTML : GET, POST
# GET
# https://www.google.com/search  ?  q=한국음악저작권협회
# 데이터 노출, 길이 제한, 폼 전달 불가


# 문제
# 한국음악저작권협회로부터 지드래곤 노래의 첫 번째 페이지로부터 제목, 가수, 작사, 작곡, 편곡 데이터를 가져오세요
def get_songs(code, page):
    payload = {
        "S_PAGENUMBER": page,
        "S_MB_CD": code,  # 'W0726200'
        # 'S_HNAB_GBN': 'I',
        # 'hanmb_nm': '지드래곤',
        # 'sort_field': 'SORT_PBCTN_DAY',
    }

    url = "https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp"
    # received = requests.get(url)
    received = requests.post(url, data=payload)
    # print(received)
    # print(received.text)

    tbody = re.findall(r"<tbody>(.+?)</tbody>", received.text, re.DOTALL)
    # print(len(tbody))
    # print(tbody[1])

    # tbody_main = tbody[1].replace(' <img src="/images/common/control.gif"  alt="" />', '')
    # tbody_main = tbody_main.replace(' <img src="/images/common/control.gif" alt="" />', '')
    # tbody_main = tbody_main.replace(' <img src="/images/common/NO_control.gif"  alt="" />', '')
    # tbody_main = tbody_main.replace(' <img src="/images/common/NO_control.gif" alt="" />', '')

    tbody_main = re.sub(r" <img.+?/>", "", tbody[1])
    tbody_main = tbody_main.replace("<br/>", ",")
    # print(tbody_main)

    table_rows = re.findall(r"<tr>(.+?)</tr>", tbody_main, re.DOTALL)
    # print(len(table_rows))
    # print(*table_rows, sep='\n')

    for item in table_rows:
        # <td></td> 와 같은 비어있는 태그도 .*?로 데이터로 포함시킴
        row = re.findall(r"<td>(.*?)</td>", item)
        row = [c.strip() for c in row]
        print(row)

    # 1개라도 가져오면 True, 비어있으면 False
    return len(table_rows) > 0


# 문제
# 지드래곤의 모든 노래를 가져오세요 (크롤링 봇)
def crawler(code):
    page = 1
    while get_songs(code, page):
        print("-" * 30, page)
        page += 1

    # print(get_songs('W0726200', 1000))        # False 반환
    # print(get_songs('W0726200', 1))           # True 반환


code = "W0654100"  # 지드래곤: 'W0726200'
crawler(code)

# 윈도우 작업 스케줄러 또는 리눅스 크론탭을 사용해서 크롤링봇 구현 가능
# python3 Day_36_01_ReSongs.py
# 해당 서버에 있는 robots.txt 파일에 적힌대로 시간 간격 조절
