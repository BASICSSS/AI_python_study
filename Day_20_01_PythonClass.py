# Day_20_01_PythonClass.py


# 프로그램 구성 요소: 함수 변수
# 클래스: 함수 + 변수


class Fake:
    pass


# 생성
# 1단계: 메모리 할당
# 2단계: 변수 연결
class Info:
    age = 25                # 클래스 변수

    def __init__(self):     # 생성자
        print('Info')
        name = 'kim'        # 지역 변수(함수 안에서만 사용)
        self.name = 'kim'   # 멤버 변수

    def show_name(self):
        print(self.name)

    def how_much(self, first, second):
        return first * second


a = Info()          # (): 함수 호출 연산자
b = Info()
f = Fake()          # 생성자(constructor)
print(a)
print(f)

print(Info.age)

print(a.name)
print()

# 문제
# show_name 함수를 호출하세요 (3가지)
a.show_name()
Info.show_name(a)
# Info.show_name(1234)      # 에러. Info 객체만 가능

# 문제
# how_much 함수를 호출하세요 (3가지)
print(a.how_much(3, 5))
print(Info.how_much(a, 3, 5))
print(Info.how_much('fake', 3, 5))

a.hobby = 'climb'           # 멤버 변수
print(a.hobby)
# print(b.hobby)            # 에러

Info.age = 99
print(a.age)
print(b.age)

