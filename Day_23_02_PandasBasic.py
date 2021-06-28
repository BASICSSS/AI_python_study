# Day_23_02_PandasBasic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def dataframe_plot():
    df = pd.DataFrame(
        {
            "city": ["jeju", "jeju", "jeju", "gunsan", "gunsan", "gunsan"],
            "year": [2018, 2019, 2020, 2018, 2019, 2020],
            "population": [300, 400, 350, 500, 550, 600],
        }
    )

    # 문제
    # 제주는 왼쪽 플롯에, 군산은 오른쪽 플롯에 막대 그래프로 그려주세요
    # 막대 색상은 다르게, 플롯 위쪽에 도시 이름도 출력합니다

    plt.subplot(1, 2, 1)
    plt.title("jeju")
    plt.bar(range(3), df.population[:3], color=colors.TABLEAU_COLORS)
    plt.xticks(range(3), df.year[:3])
    plt.ylim(0, 700)

    plt.subplot(1, 2, 2)
    plt.title("gunsan")
    plt.bar(range(3), df.population[3:], color=colors.TABLEAU_COLORS)
    plt.xticks(range(3), df.year[3:])
    plt.ylim(0, 700)

    plt.show()


def dataframe_basic():
    df = pd.read_csv("data/scores.csv")
    print(df)
    print()

    # 정수 인덱스인 경우 loc와 iloc가 동일하다
    print(df.loc[2])
    print(df.iloc[2])
    print()

    print(df["kor"])
    print(df.kor)
    print("-" * 30)

    # 문제
    # 모든 학생의 점수 합계를 구하세요
    print(df.values[:, 2:].sum())
    print(sum(df.kor) + sum(df.eng) + sum(df.mat) + sum(df.bio))
    print(df.kor.values.sum() + df.eng.values.sum() + df.mat.values.sum() + df.bio.values.sum())
    print(df.kor.sum() + df.eng.sum() + df.mat.sum() + df.bio.sum())
    print(sum(df.kor + df.eng + df.mat + df.bio))
    print((df.kor + df.eng + df.mat + df.bio).sum())
    # print(df.sum('kor', 'eng', 'mat', 'bio'))             # 에러
    print()

    # print(df.columns)           # Index(['class', 'name', 'kor', 'eng', 'mat', 'bio'], dtype='object')
    # print(df.columns.values)    # ['class' 'name' 'kor' 'eng' 'mat' 'bio']

    subjects = ["kor", "eng", "mat", "bio"]
    # print(df[subjects])                   # 인덱스 배열
    # print(df[subjects].sum())
    # kor    933
    # eng    892
    # mat    936
    # bio    987
    # dtype: int64

    # print(df[subjects].sum(axis=0))     # 수직
    # print(df[subjects].sum(axis=1))     # 수평

    print(df[subjects].sum().sum())
    print(df[subjects].sum(axis=0).sum())
    print(df[subjects].sum(axis=1).sum())
    print("-" * 30)

    # 클래스와 이름을 덧셈
    # print(df.sum())

    # 문제
    # 과목별 평균과 학생별 평균을 구하세요
    df_temp = df[subjects]
    print(df_temp.sum(axis=0) / 12)
    print(df_temp.sum(axis=1) / 4)
    print()

    print(df_temp.shape)  # (12, 4)
    print(df_temp.values.shape)  # (12, 4)

    print(df_temp.sum(axis=0) / df_temp.shape[0])
    print(df_temp.sum(axis=1) / df_temp.shape[1])
    print()

    print(df[subjects].mean(axis=0))
    print(df[subjects].mean(axis=1))
    print("-" * 30)

    df["sum"] = df[subjects].sum(axis=1)
    df["avg"] = df[subjects].mean(axis=1)
    print(df)
    print("-" * 30)

    print(df.sort_values("avg"))
    print()
    print(df.sort_values("avg", ascending=False))
    print("-" * 30)

    # 문제
    # 넘파이의 argsort 함수를 사용해서 앞의 결과와 똑같이 정렬된 형태로 출력하세요
    # print(df.sort_values('avg').index.values)
    # orders = np.argsort(df.values[:, -1])
    # orders = np.argsort(df.avg).values
    orders = np.argsort(df.avg.values)
    print(orders)  # [ 1 10  6  4  8 11  2  5  0  7  3  9]
    # print(df.iloc[orders])            # 오름차순
    print(df.iloc[orders[::-1]])  # 내림차순
    print("-" * 30)

    df.index = df.name
    # del df.name               # 에러
    del df["name"]  # 성공
    print(df)

    # df.avg.plot()
    # df.avg.plot(kind='line')
    # df.avg.plot(kind='bar')
    # df.avg.plot(kind='bar', figsize=[8, 4])

    # 문제
    # matplotlib만 사용해서 df.avg.plot 함수와 똑같이 플롯을 그려주세요
    # plt.figure(figsize=[8, 4])
    # plt.bar(range(12), df.avg, color=colors.TABLEAU_COLORS)
    # plt.xticks(range(12), df.index, rotation=45)
    #
    # plt.show()
    print("-" * 30)

    # 문제
    # 1반과 2반 중에서 어느 반의 평균 점수가 높을까요?
    print("1반 :", df)
    print("1반 :", df.values[:6, -1])
    print("1반 :", df.values[:6, -1].mean())
    print("2반 :", df.values[6:, -1])
    print("2반 :", df.values[6:, -1].mean())
    exit()

    print("1반 :", df.avg[:6].mean())
    print("2반 :", df.avg[6:].mean())

    # print(df.class)           # 에러
    print(df["class"] == 1)  # broadcast

    c1 = df[df["class"] == 1]
    c2 = df[df["class"] == 2]
    print(c1, end="\n\n")
    print(c2, end="\n\n")

    print("1반 :", c1.avg.mean())
    print("2반 :", c2.avg.mean())

    print("1반 :", df.avg[df["class"] == 1].mean())
    print("2반 :", df.avg[df["class"] == 2].mean())
    print("-" * 30)

    # 문제
    # 과목별 막대 그래프를 그려주세요
    # print(df[subjects])
    # print(df[subjects].plot(kind='bar'))

    # df[subjects].boxplot()

    # plt.subplot(1, 2, 1)
    # c1[subjects].boxplot()
    # plt.subplot(1, 2, 2)
    # c2[subjects].boxplot()

    df.plot(kind="scatter", x="kor", y="mat")

    plt.show()


# dataframe_plot()
dataframe_basic()

print("\n\n\n\n\n\n\n")

