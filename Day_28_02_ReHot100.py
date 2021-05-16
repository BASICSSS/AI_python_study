# Day_28_02_ReHot100.py
import requests
import re


# 문제
# 빌보드 핫100 사이트로부터 순위별 노래 제목과 가수를 표시하세요

url = 'https://www.billboard.com/charts/hot-100'
received = requests.get(url)
text = received.text
# text = received.content.decode('utf-8')       # 적용하는게 맞다
# print(text)

# 1번
items = re.findall(r'<li class="chart-list__element display--flex">(.+?)</li>', text, re.DOTALL)
# print(len(items))

for item in items:
    rank = re.findall(r'<span class="chart-element__rank__number">(.+?)</span>', item)
    song = re.findall(r'truncate color--primary">(.+?)</span>', item)
    singer = re.findall(r'truncate color--secondary">(.+?)</span>', item)

    rank = rank[0]
    song = song[0]

    # 미래에는 처리하지 않는 특수 문자가 포함될 수 있기 때문에
    # 정확하게 구현하기 위해서는 모든 특수문자에 대한 처리가 필요하다
    # for sp, ch in [('&amp;', '&'), ('&#039;', "'")]:
    #     song = song.replace(sp, ch)
    #     singer = singer[0].replace(sp, ch)

    song = song.replace('&amp;', '&')
    song = song.replace('&#039;', "'")
    singer = singer[0].replace('&amp;', '&')
    print('{:3} : {} : {}'.format(rank, song, singer))

print('-------------------------------------------')

# 2번
songs = re.findall(r'truncate color--primary">(.+?)</span>', text)
singers = re.findall(r'truncate color--secondary">(.+?)</span>', text)
# print(len(songs), len(singers))

for rank, (song, singer) in enumerate(zip(songs, singers), 1):
    song = song.replace('&amp;', '&')
    song = song.replace('&#039;', "'")
    singer = singer.replace('&amp;', '&')
    print('{:>3} : {:25} : {}'.format(rank, song, singer))
