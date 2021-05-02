# Day_13_04_PythonFile.py


def read_file_1():
    # f = open('./data/poem.txt')
    # f = open('data\\poem.txt')        # 윈도우 전용
    f = open('data/poem.txt', 'r', encoding='utf-8')     # euc-kr

    lines = f.readlines()
    print(lines)

    for line in lines:
        # print(line, end='')
        line = line.strip()
        print(line)

    f.close()

    # 공백: space, tab, enter
    # strip: 문자열 양쪽의 모든 공백 제거
    s = '\n   \n   \n\t  Good morning!!  \t  \t   \t\n\n'
    print('[{}]'.format(s))
    print('[{}]'.format(s.strip()))


def read_file_2():
    f = open('data/poem.txt', 'r', encoding='utf-8')     # euc-kr

    while True:                 # 무한 루프
        line = f.readline()

        # 참: 거짓이 아닌 모든 값
        # 거짓: 0, 0.0, False, None, '', [], {}
        # if len(line) == 0:
        if not line:
            break

        line = line.strip()
        print(line)

    f.close()


def read_file_3():
    f = open('data/poem.txt', 'r', encoding='utf-8')     # euc-kr

    for line in f:
        print(line.strip())

    f.close()


def read_file_4():
    with open('data/poem.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line.strip())

        # f.close()


def write():
    f = open('data/sample.txt', 'w', encoding='utf-8')

    f.write('hello')
    f.write('\n')
    f.write('python')

    f.close()


# 문제
# copy_file 함수를 완성하세요 (파일 복사)
# source, destination
def copy_file(src, dst):
    f1 = open(src, 'r', encoding='utf-8')
    f2 = open(dst, 'w', encoding='utf-8')

    # f2.write(f1.read())
    # f2.writelines(f1.readlines())

    for line in f1:
        f2.write(line)

    f1.close()
    f2.close()


# read_file_1()
# read_file_2()
# read_file_3()
# read_file_4()

# write()
copy_file('data/poem.txt', 'data/sample.txt')
