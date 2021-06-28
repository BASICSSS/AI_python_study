# Day_31_03_SklearnBasic.py
import numpy as np
from sklearn import datasets, model_selection, svm, neighbors
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


# datasets: load(로컬), fetch(원격), make(생성)


def sklearn_1():
    iris = datasets.load_iris()
    print(type(iris))  # <class 'sklearn.utils.Bunch'>
    print(
        iris.keys()
    )  # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    print(
        iris["feature_names"]
    )  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(iris["target_names"])  # ['setosa' 'versicolor' 'virginica']

    print(iris["data"])  # [[6.2 3.4 5.4 2.3] [5.9 3.  5.1 1.8] ...]
    print(iris["target"])  # [0 0 0 0 0 0 ...]


def sklearn_2():
    iris = datasets.load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df)

    # scatter_matrix(df)
    # scatter_matrix(df, c=iris.target)                             # 종류별로 색상 처리
    # scatter_matrix(df, c=iris.target, hist_kwds={'bins': 20})     # 히스토그램 막대 20개
    scatter_matrix(df, c=iris.target, hist_kwds={"bins": 20}, cmap="jet")
    plt.show()


def sklearn_3():
    digits = datasets.load_digits()  # mnist 축소판
    print(
        digits.keys()
    )  # ['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR']

    print(digits.data.shape)  # (1797, 64)
    print(digits.data[0])
    print(digits.images[0])
    print(digits.target[0])
    print("-" * 30)

    # 머신러닝 알고리즘 적용 순서
    # 1. 머신러닝/전처리 객체 생성
    # 2. 학습
    # 3. 예측/변환
    clf = svm.SVC()  # 1. 머신러닝/전처리 객체 생성
    clf.fit(digits.data, digits.target)  # 2. 학습

    # 문제
    # predict 호출의 결과를 정확도(%)로 알려주세요
    preds = clf.predict(digits.data)  # 3. 예측/변환
    print(preds)
    print(digits.target)

    print(np.mean(preds == digits.target))
    print(np.sum(preds == digits.target) / len(digits.target))
    print(clf.score(digits.data, digits.target))
    print("-" * 30)

    # 문제
    # 마지막 1개를 제외한 데이터로 학습하고, 마지막 데이터에 대해 결과를 예측하세요
    clf2 = svm.SVC()
    clf2.fit(digits.data[:-1], digits.target[:-1])  # (1796,)

    pred = clf2.predict(digits.data[-1:])  # (1, 64)
    # pred = clf2.predict([digits.data[-1]])
    print(pred)
    print(digits.target[-1:])


# 문제
# digits 데이터에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요
def sklearn_4():
    digits = datasets.load_digits()  # mnist 축소판

    x = digits.data
    y = digits.target

    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    print("acc :", clf.score(x_test, y_test))


# 문제
# iris 데이터에 대해 80%로 학습하고 20%로 정확도를 구하세요
# (90% 이상의 정확도를 달성해야 합니다)
def sklearn_5():
    iris = datasets.load_iris()

    # 문제
    # 정상적인 정확도가 나오도록 데이터를 섞어주세요
    x = iris.data  # 0 -> 16
    y = iris.target  # 0 -> 16

    # 엉망진창
    # np.random.shuffle(x)
    # np.random.shuffle(y)

    # 방법 1
    # indices = np.arange(len(x))
    # np.random.shuffle(indices)
    # print(indices)
    #
    # x = x[indices]
    # y = y[indices]
    # print(y)
    #
    # train_size = int(len(x) * 0.8)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 방법 2
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)     # 75:25
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=100)
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=100, test_size=30)
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, shuffle=False)
    print(x_train.shape, x_test.shape)

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    print("acc :", clf.score(x_test, y_test))

    # -------------------------------------- #

    clf = svm.SVC(C=0.1, gamma=0.1)  # hyper-parameter
    clf.fit(x_train, y_train)

    print("acc :", clf.score(x_test, y_test))


# 문제
# neighbors 클래스 분류기를 사용해서 digits 데이터에 대해 80%로 학습하고 20%로 정확도를 구하세요
# 분류기에 들어가는 이웃의 갯수를 가리키는 최적의 하이퍼 파라미터를 찾으세요 (1 ~ 10 중에서)
def sklearn_6():
    # wrong
    # for n in range(1, 11):
    #     digits = datasets.load_digits()
    #
    #     data = model_selection.train_test_split(digits.data, digits.target, train_size=0.8)
    #     x_train, x_test, y_train, y_test = data
    #
    #     clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    #     clf.fit(x_train, y_train)
    #
    #     print(n, ':', clf.score(x_test, y_test))

    # right
    digits = datasets.load_digits()
    print(digits.data)

    data = model_selection.train_test_split(digits.data, digits.target, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    # 문제
    # 정확도를 막대 그래프로 그려보세요
    results = []
    for n in range(1, 11):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n)
        clf.fit(x_train, y_train)

        acc = clf.score(x_test, y_test)
        print("{:2} : {}".format(n, acc))

        results.append(acc)

    plt.bar(range(1, 11), results, color=colors.TABLEAU_COLORS)
    plt.ylim(0.5, 1.0)
    plt.show()


# sklearn_1()
# sklearn_2()
# sklearn_3()
# sklearn_4()
# sklearn_5()
sklearn_6()
