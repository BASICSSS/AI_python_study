# Day_35_01_ReStateNames.py
import re
import requests
import csv


# 문제
# 아래 사이트로부터 states.csv 파일에 들어갈 내용을 읽어서 data 폴더에 states.csv 파일을 만드세요
def get_states_1():
    # 구글에서 states.csv 검색해서 첫 번째 항목
    url = 'https://developers.google.com/public-data/docs/canonical/states_csv'
    received = requests.get(url)
    text = received.text
    # print(text)

    table = re.findall(r'<table>(.+?)</table>', text, re.DOTALL)
    # print(table[0])
    # print(len(table))         # 1

    table_rows = re.findall(r'<tr>(.+?)</tr>', table[0], re.DOTALL)

    states = []
    header = re.findall(r'<th scope="col">(.+?)</th>', table_rows[0])
    # print(header)
    states.append(header)

    for row in table_rows[1:]:
        items = re.findall(r'<td>(.+?)</td>', row)
        # print(items)
        states.append(items)

    return states


def get_states_2():
    # 구글에서 states.csv 검색해서 첫 번째 항목
    url = 'https://developers.google.com/public-data/docs/canonical/states_csv'
    received = requests.get(url)
    text = received.text

    items = re.findall(r'<td>(.+?)</td>', text)
    # return [(items[i+0], items[i+1], items[i+2], items[i+3]) for i in range(0, len(items), 4)]
    return [items[i:i+4] for i in range(0, len(items), 4)]


# 문제
# csv 모듈을 사용하지 말고 파일에 기록하세요 (states_1.csv)
def write_states_1(states):
    f = open('data/states_1.csv', 'w', encoding='utf-8')
    for row in states:
        # f.write('{},{},{},{}\n'.format(row[0], row[1], row[2], row[3]))
        f.write(','.join(row) + '\n')
    f.close()


# csv 모듈을 사용해서 파일에 기록하세요 (states_2.csv)
def write_states_2(states):
    f = open('data/states_2.csv', 'w', encoding='utf-8')
    csv.writer(f).writerows(states)
    f.close()


# states = get_states_1()
states = get_states_2()
# print(*states, sep='\n')

# write_states_1(states)
write_states_2(states)

