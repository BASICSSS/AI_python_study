# Day_39_01_MultipleRegression.py
import tensorflow as tf
import pandas as pd
import numpy as np

# 문제
# 3번 출석하고 7시간 공부한 학생과
# 5번 출석하고 4시간 공부한 학생의 성적을 구하세요
def multiple_regression_1():
    # hx = w1 * x1 + w2 * x2 + b
    # y  = a1 * x1 + a2 + x2 + b
    #       1         1        0
    # y = x1 + x2
    x1 = [1, 0, 3, 0, 5]  # 공부한 시간
    x2 = [0, 2, 0, 4, 0]  # 출석한 일수
    y = [1, 2, 3, 4, 5]  # 성적

    w1 = tf.Variable(tf.random_normal([1]))
    w2 = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    hx = w1 * x1 + w2 * x2 + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))
    print()

    print(sess.run(w1 * x1 + w2 * x2 + b))  # [0.9999998 2. 3.0000002 4. 5.]
    print(sess.run(w1 * [7, 4] + w2 * [3, 5] + b))  # [9.999999 9.]

    sess.close()


def multiple_regression_2():
    x = [[1, 0, 3, 0, 5], [0, 2, 0, 4, 0]]  # 공부한 시간  # 출석한 일수
    y = [1, 2, 3, 4, 5]  # 성적

    # 문제
    # w를 2차원으로 바꾸세요
    w = tf.Variable(tf.random_normal([2]))
    b = tf.Variable(tf.random_normal([1]))

    hx = w[0] * x[0] + w[1] * x[1] + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


def multiple_regression_3():
    x = [[1.0, 0.0, 3.0, 0.0, 5.0], [0.0, 2.0, 0.0, 4.0, 0.0]]  # 공부한 시간  # 출석한 일수
    y = [1, 2, 3, 4, 5]  # 성적

    # 문제
    # w를 2차원으로 바꾸세요
    w = tf.Variable(tf.random_normal([1, 2]))
    b = tf.Variable(tf.random_normal([1]))

    # 문제
    # 에러나지 않게 수정하세요
    # (1, 5) = (1, 2) @ (2, 5)
    hx = tf.matmul(w, x) + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


def multiple_regression_4():
    # x = [[1., 0., 3., 0., 5.],          # 공부한 시간
    #      [0., 2., 0., 4., 0.],          # 출석한 일수
    #      [1., 1., 1., 1., 1.]]          # bias
    x = [
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],  # bias.  + b(이걸 추천) 하지 않고 이렇게 처리할 경우 사실 위치하는 곳은 상관없지만 기본적으로 항상 여기에 위치!!
        [1.0, 0.0, 3.0, 0.0, 5.0],  # 공부한 시간
        [0.0, 2.0, 0.0, 4.0, 0.0],
    ]  # 출석한 일수

    y = [1, 2, 3, 4, 5]  # 성적

    # 문제
    # b를 없애보세요
    w = tf.Variable(tf.random_normal([1, 3]))

    # w[0] * x[0] + w[1] * x[1] + b
    # w[0] * x[0] + w[1] * x[1] + w[2]
    # w[0] * x[0] + w[1] * x[1] + w[2] * 1
    # <---      (1, 5)    ---->   scalar            => broadcast
    # w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # <---      (1, 5)    ---->   (5,)              => vector

    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, x)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    print("-" * 30)
    print("w :", sess.run(w))
    sess.close()


# 문제
# x를 일반적인 파일의 형태로 수정하세요 (transpose) // <핵심>데이터(피쳐)는 기본적으로 횡이아닌 이와같은 열쪽으로 늘어나는 형태를 다룸
def multiple_regression_5():  # // <핵심>y값은 2차원이여야한다.!
    x = [[1.0, 1.0, 0.0], [1.0, 0.0, 2.0], [1.0, 3.0, 0.0], [1.0, 0.0, 4.0], [1.0, 5.0, 0.0]]
    y = [[1], [2], [3], [4], [5]]

    w = tf.Variable(tf.random_normal([3, 1]))

    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(x, w)
    # (5, 5) = (5, 1) - (1, 5)      y가 1차원일 경우
    # (5, 1) = (5, 1) - (5, 1)      이게 올바른 계산
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# trees.csv 파일을 공부해서
# Girth가 10이고 Height가 70일 때와
# Girth가 20이고 Height가 80일 때의 Volume을 구하세요
def multiple_regression_trees():
    # x = [[1., 1., 0.],
    #      [1., 0., 2.],
    #      [1., 3., 0.],
    #      [1., 0., 4.],
    #      [1., 5., 0.]]
    # y = [[1],
    #      [2],
    #      [3],
    #      [4],
    #      [5]]
    #
    trees = pd.read_csv("data/trees.csv", index_col=0)
    print(trees)

    values = np.float32(
        trees.values
    )  # 문자열을 Day_38_01_TensorFlowLinearRegression 에서 처리했던것처럼 리스트로 말고 넘파이로 바로 처리

    x = values[:, :-1]
    y = values[:, -1:]
    print(x.shape, y.shape)  # (31, 2) (31, 1)
    w = tf.Variable(tf.random_normal([2, 1]))
    b = tf.Variable(tf.random_normal([1]))

    # (31, 1) = (31, 2) @ (2, 1)
    hx = tf.matmul(x, w) + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    x_test = [[10.0, 70.0], [20.0, 80.0]]
    print(sess.run(tf.matmul(x_test, w) + b))
    sess.close()


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
# multiple_regression_5()

multiple_regression_trees()

# (3, 2, 1, 5, 7) + -3                   = (0 -1, -2, 2, 4)
# (3, 2, 1, 5, 7) + (-3, -3, -3, -3, -3) = (0 -1, -2, 2, 4)
