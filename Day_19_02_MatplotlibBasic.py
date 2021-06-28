# Day_19_02_MatplotlibBasic.py
import matplotlib.pyplot as plt
import numpy as np


def plot_1():
    # plt.plot([10, 20, 30, 40, 50])

    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])         # 꺾은선(line)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')   # 산점도(scatter)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])  # P(plut) p(pentagon)

    plt.show()


# 문제
# x의 범위가 -10~10일 때 x^2 그래프를 그려보세요
def plot_2():
    # 1번
    # for x in range(-10, 11):
    #     plt.plot(x, x ** 2, 'ro')
    #     plt.plot(x, x ** 2)       # wrong. 좌표가 2개 이상 전달되어야 꺾은선 그래프를 그릴 수 있다

    # 2번
    # x, y = [], []
    # for i in range(-10, 11):
    #     x.append(i)
    #     y.append(i ** 2)
    #
    # plt.plot(x, y, 'ro')

    # 3번
    # x = range(-10, 11)
    # y = [i ** 2 for i in x]
    #
    # plt.plot(x, y, 'ro')

    # 4번
    # x = np.arange(-10, 11, 0.3)
    x = np.linspace(-10, 10, 51)
    plt.plot(x, x ** 2)
    plt.plot(x, x ** 2, "ro")

    plt.show()


def plot_3():
    t = np.arange(0, 5, 0.2)
    plt.plot(t, t, "r--")
    plt.plot(t, t ** 2, "g>")
    plt.plot(t, t ** 3, "bx")
    plt.show()


# 문제
# desmos.com에서 그렸던 로그 그래프 4개를 하나의 플롯에 그려보세요
def plot_4():
    x = np.arange(0.1, 2.0, 0.1)

    plt.grid(True)

    plt.plot(x, np.log(x))
    plt.plot(x, -np.log(x))

    plt.plot(-x, np.log(x))
    plt.plot(-x, -np.log(x))

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


plot_1()
# plot_2()
# plot_3()
# plot_4()

