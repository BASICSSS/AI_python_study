# Day_16_01_sqlite.py
import sqlite3


# 문제
# weather.txt 파일을 읽어서 반환하는 함수를 만드세요
def read_weather():
    f = open('data/weather.txt', 'r', encoding='utf-8')

    rows = []
    for line in f:
        row = line.strip().split(',')
        # print(row)
        rows.append(row)

    f.close()
    return rows


def create_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    query = 'CREATE TABLE kma (prov TEXT, city TEXT, mode TEXT, tmEf TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT)'
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_row(row):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    base = 'INSERT INTO kma (prov, city, mode, tmEf, wf, tmn, tmx, rnSt) '\
           'VALUES("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    # query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
    query = base.format(*row)
    cur.execute(query)

    conn.commit()
    conn.close()


# 문제
# 전체 데이터를 한번에 추가하는 insert_all 함수를 만드세요
def insert_all(rows):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    base = 'INSERT INTO kma (prov, city, mode, tmEf, wf, tmn, tmx, rnSt) '\
           'VALUES("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'

    for row in rows:
        query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        cur.execute(query)

    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    query = 'SELECT * FROM kma'     # * : all
    # query = 'SELECT city, wf FROM kma'

    # rows = []
    # for row in cur.execute(query):
    #     # print(row)
    #     rows.append(row)
    rows = list(cur.execute(query))

    # conn.commit()             # 읽기만 하기 때문에 필요없다
    conn.close()

    return rows


def search_city(city):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # query = 'SELECT * FROM kma WHERE city="추자도"'
    # query = 'SELECT * FROM kma WHERE city="' + city + '"'
    query = 'SELECT * FROM kma WHERE city="{}"'.format(city)
    rows = list(cur.execute(query))

    # conn.commit()             # 읽기만 하기 때문에 필요없다
    conn.close()

    return rows


rows = read_weather()
# print(rows)
# for r in rows:
#     print(r[5])
# print(*rows, sep='\n')

# create_db()

# 문제
# 전체 데이터를 db에 추가하세요
# insert_row(rows[0])
# insert_row(rows[1])
# insert_row(rows[2])

# for row in rows:
#     insert_row(row)

# insert_all(rows)

# pk : primary key. 현재 테이블에서 중복되지 않는 값
# fk : foreign key. 다른 테이블에서의 pk

# CRUD : Create, Read, Update, Delete

# rows = fetch_all()
# print(*rows, sep='\n')

# 문제
# '추자도' 지역의 날씨 데이터만 반환하는 함수를 구현하세요
rows = search_city('추자도')
print(*rows, sep='\n')
