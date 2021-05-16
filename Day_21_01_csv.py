# Day_21_01_csv.py
import csv              # Comma(,) Separated Values

# 구분자: separator, delimiter


def read_weather():
    f = open('data/weather.txt', 'r', encoding='utf-8')

    # for row in csv.reader(f):
    for row in csv.reader(f, delimiter=','):
        print(row)

    f.close()


# 문제
# us-500.csv 파일을 csv 모듈을 사용해서 출력하세요
def read_us500():
    f = open('data/us-500.csv', 'r', encoding='utf-8')
    # f.readline()

    # for row in csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL):
    for row in csv.reader(f):
        assert len(row) == 12
        print(row)

    f.close()


# 문제
# weather.txt 파일을 읽어서 컬럼을 큰 따옴표로 감싸고 구분자는 세미콜론(;)인 weather.csv 파일을 만드세요
# csv.writer()
def write_weather():
    # f1 = open('data/weather.txt', 'r', encoding='utf-8')
    # f2 = open('data/weather.csv', 'w', encoding='utf-8')
    #
    # writer = csv.writer(f2, delimiter=',', quoting=csv.QUOTE_ALL)
    #
    # for row in csv.reader(f1):
    #     # csv.writer(f2, delimiter=',', quoting=csv.QUOTE_ALL).writerow(row)
    #     writer.writerow(row)
    #
    # f1.close()
    # f2.close()

    # 중간에 빈 줄 생기면 newline 옵션 추가합니다
    f1 = open('data/weather.txt', 'r', encoding='utf-8', newline='')

    # 1번
    # rows = []
    # for row in csv.reader(f1):
    #     rows.append(row)

    # 2번
    # rows = [row for row in csv.reader(f1)]

    # 3번
    rows = list(csv.reader(f1))
    # print(*rows, sep='\n')

    f1.close()

    # -------------------------- #

    f2 = open('data/weather.csv', 'w', encoding='utf-8')
    csv.writer(f2, delimiter=';', quoting=csv.QUOTE_ALL).writerows(rows)
    f2.close()


# read_weather()
# read_us500()

write_weather()
