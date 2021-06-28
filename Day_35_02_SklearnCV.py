# Day_35_02_SklearnCV.py        # cross-validation
from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    LeaveOneOut,
    ShuffleSplit,
)
import matplotlib.pyplot as plt
import numpy as np
import itertools

# 머신러닝: 소규모 데이터, 학습시간 짧다
# 딥러닝: 빅데이터, 학습시간 길다


def show_blobs():
    x, y = make_blobs(random_state=10, centers=7)
    print(x.shape, y.shape)  # (100, 2) (100,)
    print(y[:10])  # [0 6 5 5 5 6 2 3 6 3]

    # 문제
    # x, y 데이터를 scatter 플롯에 출력하세요
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


# 문제
# iris 데이터셋에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요
def cv_1():
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.8)

    # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
    clf = LogisticRegression(solver="liblinear")
    clf.fit(x_train, y_train)

    print("acc :", clf.score(x_test, y_test))


# cross-validation 적용
# 1. 여러 번에 걸쳐서 모델 적용
# 2. 데이터를 중복되지 않게 분할
def cv_2():
    def extract(data, idx):
        return data[idx * 50 : idx * 50 + 50]  # idx(0) : data[0:50], idx(1) : data[50:100]

    iris = load_iris()
    clf = LogisticRegression(solver="liblinear")

    indices = np.arange(len(iris.data))
    np.random.shuffle(indices)

    # cv 3회 적용
    # 분할: 1번(0~49) 2번(50~99) 3번(100~149)
    # 학습 1회: 학습(1번, 2번) 검사(3번)
    # 학습 2회: 학습(2번, 3번) 검사(1번)
    # 학습 3회: 학습(3번, 1번) 검사(2번)

    # 문제
    # 앞의 설명을 토대로 교차 검증을 3회 진행하는 코드를 구현하세요
    x, y = iris.data[indices], iris.target[indices]

    # 아래 반복문에 들어가는 리스트 생성
    # itertools.combinations()
    # itertools.permutations()

    # for i1, i2, i3 in [(0, 50, 100), (50, 100, 0), (100, 0, 50)]:
    for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        x_train = np.vstack([extract(x, i1), extract(x, i2)])
        y_train = np.concatenate([extract(y, i1), extract(y, i2)])
        x_test = extract(x, i3)
        y_test = extract(y, i3)

        # print(x_train.shape, y_train.shape)         # (100, 4) (100,)
        # print(x_test.shape, y_test.shape)           # (50, 4) (50,)

        clf.fit(x_train, y_train)
        print("acc :", clf.score(x_test, y_test))


def cv_3():
    iris = load_iris()
    clf = LogisticRegression(solver="liblinear")

    scores = cross_val_score(clf, iris.data, iris.target)
    print("5-folds :", scores)  # 5-folds : [1. 0.96666667 0.93333333 0.9 1.]

    scores = cross_val_score(clf, iris.data, iris.target, cv=3)
    print("3-folds :", scores)  # 3-folds : [0.96 0.96 0.94]

    scores = cross_val_score(clf, iris.data, iris.target, cv=KFold(n_splits=3))
    print("KFold(3) :", scores)  # KFold(3) : [0. 0. 0.]

    scores = cross_val_score(clf, iris.data, iris.target, cv=KFold(n_splits=5))
    print("KFold(5) :", scores)  # KFold(3) : [0. 0. 0.]


def cv_4():
    iris = load_iris()
    clf = LogisticRegression(solver="liblinear")

    # for i_train, i_test in KFold().split(iris.data, iris.target):
    #     print(i_train.shape, i_test.shape)          # (120,) (30,)
    #     print(i_test)

    indices = np.arange(len(iris.data))
    np.random.shuffle(indices)

    x, y = iris.data[indices], iris.target[indices]

    for i_train, i_test in KFold(n_splits=3).split(x, y):
        x_train, x_test = x[i_train], x[i_test]
        y_train, y_test = y[i_train], y[i_test]

        clf.fit(x_train, y_train)
        # print("acc :", clf.score(x_test, y_test))


# show_blobs()

# cv_1()
# cv_2()
# cv_3()
cv_4()

