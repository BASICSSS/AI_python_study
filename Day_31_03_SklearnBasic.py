# Day_31_03_SklearnBasic.py
import numpy as np
from sklearn import datasets, svm, model_selection, neighbors
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


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

    clf = svm.SVC()
    clf.fit(digits.data, digits.target)

    # 문제
    # predict 호출의 결과를 정확도(%)로 알려주세요
    preds = clf.predict(digits.data)
    print(preds)
    print(digits.target)

    print(np.mean(preds == digits.target))
    print(np.sum(preds == digits.target) / len(digits.target))
    print(accuracy_score(digits.target, preds))
    print("-" * 30)

    # 문제
    # 마지막 1개를 제외한 데이터로 학습하고, 마지막 데이터에 대해 결과를 예측하세요
    clf2 = svm.SVC()
    # clf2.fit(digits.data[:])
    clf2.fit(digits.data[:-1], digits.target[:-1])  # (1796,)

    # 문제
    # predict 호출의 결과를 정확도(%)로 알려주세요
    pred = clf2.predict(digits.data[-1:])
    print(pred)
    print(digits.target[-1:])


# 문제
# digits 데이터에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요
def sklearn_4():
    digits = datasets.load_digits()  # mnist 축소판

    x = digits.data
    y = digits.target
    print(y)  # 출력해보면 기본적으로 섞여있는것을 볼 수 있음

    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))


def sklearn_5():
    iris = datasets.load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)

    print(accuracy_score(pred, y_test))


# sklearn_1()
# sklearn_2()
# sklearn_3()
sklearn_5()
