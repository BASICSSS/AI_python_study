# Day_36_02_DeepLearningBasic.py
import matplotlib.pyplot as plt

#     x : 1 2 3
# --------------
# 가설 1 : 1 1 1 = 1 + 1 + 1 = 3 (cost, loss, error) 1 * 1 + 1 * 1 + 1 * 1 = 3
# 가설 2 : 0 0 2 = 0 + 0 + 2 = 2                     0 * 0 + 0 * 0 + 2 * 2 = 4
#                 mae(mean absolute error)          mse(mean square error)
#                 평균 절대값 오차                      평균 제곱 오차


# 문제
# cost 함수를 만드세요mae
def cost(x, y, w):
    s = 0
    for i in range(len(x)):
        hx = w * x[i]
        c = (hx - y[i]) ** 2
        s += c

    return s / len(x)


def show_cost():
    # y = ax + b
    # y = x
    # y = 1x + 0
    # hx = wx + b
    #      1    0
    x = [1, 2, 3]
    y = [1, 2, 3]

    # print(cost(x, y, -1))       # 18.666666666666668
    # print(cost(x, y, 0))        # 4.666666666666667
    # print(cost(x, y, 1))        # 0.0
    # print(cost(x, y, 2))        # 4.666666666666667
    # print(cost(x, y, 3))        # 18.666666666666668

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        print(w, c)

        plt.plot(w, c, "ro")
    plt.show()


show_cost()

