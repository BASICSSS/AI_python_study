# Day_37_01_SklearnLeaf.py
import pandas as pd
from sklearn import model_selection, linear_model, preprocessing, svm, neighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# 문제
# leaf.csv 파일을 읽어서 80%로 학습하고 20%에 대해 정확도를 구하세요
# 목표 정확도는 최소 80% 이상.
def model_leaf_1():
    leaf = pd.read_csv("./data/leaf.csv", index_col=0)
    print(leaf, end="\n\n")
    x = leaf.values[:, 1:]  # feature
    y = leaf.values[:, 0]  # target, label
    print(x)  # (990, 192) (990,)
    print(x.shape, y.shape)  # (990, 192) (990,)
    # x = preprocessing.scale(x)
    x = preprocessing.minmax_scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    # clf = linear_model.LogisticRegression()
    # clf = linear_model.LogisticRegression(solver='liblinear')
    # clf = svm.SVC()
    clf = neighbors.KNeighborsClassifier(n_neighbors=7)

    clf.fit(x_train, y_train)
    print("acc :", clf.score(x_test, y_test))

    # LogisticRegression
    # acc : 0.1717171717171717      non-preprocessing
    # acc : 0.9949494949494949      standard scaling + solver='lbfgs'       warning
    # acc : 0.9646464646464646      standard scaling + solver='liblinear'
    # acc : 0.9797979797979798      min-max scaling + solver='lbfgs'         warning
    # acc : 0.9797979797979798      min-max scaling + solver='liblinear'

    # SVC
    # acc : 0.8585858585858586      non-preprocessing
    # acc : 0.9848484848484849      standard scaling
    # acc : 0.9696969696969697      min-max scaling

    # KNeighborsClassifier
    # acc : 0.797979797979798       non-preprocessing
    # acc : 0.9191919191919192      standard scaling
    # acc : 0.9494949494949495      min-max scaling


# 문제
# 캐글 leaf 경진대회에서 Most votes를 받은
# 10 Classifier Showdown in Scikit-Learn 코드를 정리해서 함수로 만드세요
# 최종 서브미션 파일을 만들어서 Late submission까지 진행합니다
def model_leaf_2():
    train = pd.read_csv("./leaf-classification/train.csv/train.csv")
    test = pd.read_csv("./leaf-classification/test.csv/test.csv")

    def encode(train, test):
        le = LabelEncoder().fit(train.species)
        labels = le.transform(train.species)  # encode species strings
        classes = list(le.classes_)  # save column names for submission
        test_ids = test.id  # save test ids for submission

        train = train.drop(["species", "id"], axis=1)
        test = test.drop(["id"], axis=1)

        return train, labels, test, test_ids, classes

    train, labels, test, test_ids, classes = encode(train, test)
    # print(train.head())

    # classifiers = [
    #     KNeighborsClassifier(3),
    #     SVC(kernel="rbf", C=0.025, probability=True),
    #     NuSVC(probability=True),
    #     DecisionTreeClassifier(),
    #     RandomForestClassifier(),
    #     AdaBoostClassifier(),
    #     # GradientBoostingClassifier(),
    #     GaussianNB(),
    #     LinearDiscriminantAnalysis(),
    #     QuadraticDiscriminantAnalysis()]
    #
    # # Logging for Visual Comparison
    # log_cols = ["Classifier", "Accuracy", "Log Loss"]
    # log = pd.DataFrame(columns=log_cols)
    #
    # # 10번 반복하는 데이터를 만든 것은 코드 오류.
    # # train_test_split 함수로 한 번만 만드는 것이 맞다
    # sss = StratifiedShuffleSplit(10, test_size=0.2, random_state=23)
    #
    # for train_index, test_index in sss.split(train, labels):
    #     X_train, X_test = train.values[train_index], train.values[test_index]
    #     y_train, y_test = labels[train_index], labels[test_index]
    #
    # for clf in classifiers:
    #     clf.fit(X_train, y_train)
    #     name = clf.__class__.__name__
    #
    #     print("=" * 30)
    #     print(name)
    #
    #     print('****Results****')
    #     train_predictions = clf.predict(X_test)
    #     acc = accuracy_score(y_test, train_predictions)
    #     print("Accuracy: {:.4%}".format(acc))
    #
    #     # predict_proba 함수 대신 score 함수를 사용하는 것이 쉽다
    #     train_predictions = clf.predict_proba(X_test)
    #     ll = log_loss(y_test, train_predictions)
    #     print("Log Loss: {}".format(ll))
    #
    #     log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    #     log = log.append(log_entry)
    #
    # print("=" * 30)
    #
    # sns.set_color_codes("muted")
    # sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    #
    # plt.xlabel('Accuracy %')
    # plt.title('Classifier Accuracy')
    # plt.show()
    #
    # sns.set_color_codes("muted")
    # sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
    #
    # plt.xlabel('Log Loss')
    # plt.title('Classifier Log Loss')
    # plt.show()

    # 없는 코드를 추가했음
    data = model_selection.train_test_split(train, labels, train_size=0.8)
    X_train, X_test, y_train, y_test = data

    # Predict Test Set
    favorite_clf = LinearDiscriminantAnalysis()
    favorite_clf.fit(X_train, y_train)
    test_predictions = favorite_clf.predict_proba(test)  # proba: probability
    # print(test_predictions.shape)         # (594, 99)

    # Format DataFrame
    submission = pd.DataFrame(test_predictions, columns=classes)
    submission.insert(0, "id", test_ids)
    submission.reset_index()

    # Export Submission
    # submission.to_csv('leaf-classification/submission.csv', index=False)
    print(submission.tail())
    print()

    print("id,", ",".join(classes), sep="")
    for i in range(len(submission)):
        row = submission.iloc[i].values
        print(int(row[0]), "".join([str(c) for c in row[1:]]), sep=",")


# model_leaf_1()
model_leaf_2()

