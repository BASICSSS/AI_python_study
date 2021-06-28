# Day_28_01_MatplotlibMiddle.py
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors


def plot_1():
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    plt.plot(t, s, lw=3)

    plt.annotate(
        "local max", xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor="black", shrink=0.05)
    )
    plt.ylim(-2, 2)
    plt.show()


# 문제
# 0에서 시작해서 100개의 난수(-1, 0, 1)를 발생시켜 위아래로 움직이는 꺾은 선 그래프를 그려주세요
#  3       --
#  2      -  -
#  1     -
#  0 -- -
# -1   -
def plot_2():  # random walker
    pos = 0
    y = [pos]
    for i in range(100):
        # n = random.randrange(3) - 1
        # n = np.random.random_integers(-1, 1)
        n = np.random.randint(-1, 2)
        pos += n
        y.append(pos)
        # plt.plot(i, pos, 'rx')

    plt.plot(range(len(y)), y)
    # plt.plot(range(len(y)), y, 'rx')
    plt.show()


# 문제
# 앞에서 만든 random walker 그래프를 컴프리헨션으로 바꾸세요
def plot_3():
    # randoms = np.random.randint(-1, 2, 100 + 1)
    randoms = np.random.choice([-1, 0, 1], 100 + 1)
    randoms[0] = 0
    y1 = [sum(randoms[: i + 1]) for i in range(len(randoms))]
    y2 = np.cumsum(randoms)

    # cumsum 직접 구현 1번
    pos = 0
    y3 = [0]
    for i in range(1, len(randoms)):
        pos += randoms[i]
        y3.append(pos)

    # cumsum 직접 구현 2번
    y4 = [0]
    for i in range(1, len(randoms)):
        y4.append(y4[-1] + randoms[i])

    plt.subplot(1, 2, 1)
    plt.plot(range(len(y1)), y1, "r")
    plt.plot(range(len(y2)), y2, "b")

    plt.subplot(1, 2, 2)
    plt.plot(range(len(y1)), randoms, "ro")

    plt.show()


def plot_4():
    print(plt.style.available)
    # ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background',
    # 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright',
    # 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid',
    # 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper',
    # 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
    # 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    print(len(plt.style.available))  # 26

    x = np.linspace(0, 10)
    with plt.style.context("fast"):
        plt.plot(x, np.log(x), "rx")
        plt.plot(x, np.sin(x))

    plt.show()


# 문제
# 사용가능한 모든 스타일을 한 줄에 5개씩, 피겨 1개에 표시하세요
def plot_5():
    # plt.figure(figsize=[12, 8])
    plt.figure(figsize=[30, 20])

    # x = np.linspace(1, 10)
    x = np.arange(10)
    y = x + np.random.randint(-5, 6, len(x))
    for i, style in enumerate(plt.style.available, 1):
        # plt.subplot(6, 5, i)      # 실패. 스타일 반영 안됨

        with plt.style.context(style):
            plt.subplot(6, 5, i)
            # plt.plot(x, np.log(x), 'rx')
            # plt.plot(x, np.sin(x))

            plt.bar(x, y, color=colors.TABLEAU_COLORS)

    plt.tight_layout()
    # plt.show()
    plt.savefig("data/plt_styles.png")


def plot_6():
    x = np.arange(10)
    y = x + np.random.randint(-5, 6, len(x))

    # 문제
    # 플롯 3개를 추가하세요
    ax1 = plt.subplot2grid([3, 3], [0, 0], colspan=3)
    ax2 = plt.subplot2grid([3, 3], [1, 0], colspan=2)
    ax3 = plt.subplot2grid([3, 3], [2, 0])
    ax4 = plt.subplot2grid([3, 3], [1, 2], rowspan=2)

    ax1.plot(x, y, "r")
    ax2.plot(x, y, "bx")
    ax3.bar(x, y, color=colors.TABLEAU_COLORS)
    ax4.barh(x, y, color=colors.TABLEAU_COLORS)

    plt.show()


# plot_1()
# plot_2()
plot_5()
# plot_4()
# plot_5()
# plot_6()

