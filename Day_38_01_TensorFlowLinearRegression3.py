# Day_38_01_TensorFlowLinearRegression.py
from os import name
from numpy import dtype
import tensorflow as tf
import pandas as pd


def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b  # 1번     : broadcast, broadcast => (3,)
    loss_i = (hx - y) ** 2  # 2번, 3번 : vector, broadcast   => (3,)
    loss = tf.reduce_mean(loss_i)  # 배열에 대한 평균 => scalar

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    # clf = svm.SVM
    # clf.fit()          과 같은 과정이라고 보면됨
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss), sess.run(w), sess.run(b))
    print()

    # 문제
    # x가 3과 5일 때의 결과를 예측하세요
    # print(w * 5 + b)

    ww, bb = sess.run(w), sess.run(b)
    print("5 :", ww * 5 + bb)
    print("y :", ww * 7 + bb)
    print()

    print("5 :", sess.run(w * 5 + b))
    print("7 :", sess.run(w * 7 + b))
    print()

    print("* :", sess.run(w * [5, 7] + b))
    sess.close()


# 문제
# cars.csv 파일을 읽어서
# 속도(speed)가 10과 30일 때의 제동거리(dist)를 구하세요
def get_cars():
    # data = pd.read_csv(
    #     "./data/cars.csv", names=["index", "speed", "dist"], delimiter=",", engine="python",
    # )
    # fm = data.drop(["index"], axis=1)
    # fm = data.drop(0, axis=0)

    # speed = list(map(int, fm.speed))
    # dist = list(map(int, fm.dist))

    # return speed, dist            판다스로 처리해본거

    f = open("data/cars.csv", "r", encoding="utf-8")
    f.readline()

    x, y = [], []
    for row in f:
        _, speed, dist = row.strip().split(",")
        # print(speed, dist)

        x.append(int(speed))
        y.append(int(dist))

    f.close()

    return x, y


def linear_regression_cars():
    # x = [1, 2, 3]
    # y = [1, 2, 3]             아래는 리니어리그레이션 그대로
    x, y = get_cars()

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    print("10, 30", sess.run(w * [10, 30] + b))
    sess.close()


# linear_regression()
linear_regression_cars()
