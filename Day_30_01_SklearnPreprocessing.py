# Day_30_01_SklearnPreprocessing.py
from sklearn import preprocessing, impute
import numpy as np


def add_dummy_feature():
     # y = ax + b
     # bias
     x = [[0, 1],
          [2, 3]]
     print(preprocessing.add_dummy_feature(x))
     print(preprocessing.add_dummy_feature(x, value=7))
     print()

     # 문제
     # bias를 첫 번째 행에 추가하세요
     # print(preprocessing.add_dummy_feature(x).T)   # 실패
     print(np.transpose(x))
     print(preprocessing.add_dummy_feature(np.transpose(x)))
     print(preprocessing.add_dummy_feature(np.transpose(x)).T)


def Binarizer():
    x = [[1., -1.,  2.],
         [2.,  0.,  0.],
         [0.,  1., -1.]]

    bin = preprocessing.Binarizer()
    print(bin.fit(x))
    print(bin)
    print(bin.transform(x))

    bin = preprocessing.Binarizer().fit(x)
    print(bin.transform(x))

    print(preprocessing.Binarizer(threshold=-0.5).fit_transform(x))


def Imputer():
    # 4 = (1 + 7) / 2
    # 5 = (2 + 4 + 9) / 3
    x = [[1, 2],
         [np.nan, 4],
         [7, 9]]

    # strategy: "mean", "median", "most_frequent", "constant"
    imp = impute.SimpleImputer(strategy='mean')
    imp.fit(x)

    print(imp.missing_values)
    print(imp.strategy)
    print(imp.statistics_)              # [4. 5.]

    print(imp.transform(x))

    y = [[np.nan, np.nan]]
    print(imp.transform(y))


# 클래스: 중복되지 않는 값
def LabelBinarizer():
    x = [1, 2, 6, 2, 4]

    lb = preprocessing.LabelBinarizer()
    lb.fit(x)

    # 이진번: 000 001 010 011 100 101 110 111
    # one-hot 벡터
    # 1: 1 0 0 0
    # 2: 0 1 0 0
    # 4: 0 0 1 0
    # 6: 0 0 0 1

    print(lb.classes_)        # [1 2 4 6]
    print(lb.transform(x))

    # print(np.eye(4))
    print('-' * 30)

    lb = preprocessing.LabelBinarizer()
    print(lb.fit_transform(['yes', 'no']))
    print(lb.fit_transform(['yes', 'no', 'no']))
    print(lb.fit_transform(['yes', 'no', 'no', 'cancel']))
    print('-' * 30)

    target = ['yes', 'no', 'no', 'cancel']
    lb.fit(target)

    t = lb.transform(target)
    print(t)
    print(lb.classes_)

    # 문제
    # t를 [2, 1, 1, 0]로 변환하세요
    # [[0 0 1]  : 2
    #  [0 1 0]  : 1
    #  [0 1 0]  : 1
    #  [1 0 0]] : 0
    a = np.argmax(t, axis=1)
    print(a)
    print(lb.classes_[a])

    print(lb.inverse_transform(t))
    print('-' * 30)

    lb = preprocessing.LabelBinarizer(sparse_output=True)
    lb.fit(x)

    print(lb.classes_)
    print(lb.transform(x))


def LabelEncoder():
    x = [2, 1, 2, 6]

    le = preprocessing.LabelEncoder()
    le.fit(x)

    print(le.classes_)
    print(le.transform(x))
    print('-' * 30)

    target = ['yes', 'no', 'no', 'cancel']

    le = preprocessing.LabelEncoder()
    le.fit(target)

    print(le.classes_)

    t = le.transform(target)
    print(t)
    print(le.inverse_transform(t))
    print('-' * 30)

    # 문제
    # inverse_transform 함수처럼 동작하는 코드를 구현하세요
    print(le.classes_[t])


def StandardScaler():
    x = [[1, -1, 5],
         [2, 0, -5],
         [0, 1, -10]]

    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    print(scaler.scale_)      # [0.81649658 0.81649658 6.23609564]

    t = scaler.transform(x)
    print(t)
    print(preprocessing.scale(x))


def MinMaxScaler():
    x = [[1, -1, 5],
         [2, 0, -5],
         [0, 1, -10]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)
    print(scaler.scale_)      # [0.5 0.5 0.06666667]

    t = scaler.transform(x)
    print(t)
    # print(preprocessing.scale(x))
    print(preprocessing.minmax_scale(x))
    print('-' * 30)

    # 문제
    # 넘파이를 사용해서 x에 대해 minmax 스케일링 코드를 구현하세요
    mx = np.max(x, axis=0)
    mn = np.min(x, axis=0)
    print(mx, mn)

    # (3, 3) - (1, 3) / (3,) - (3,)
    print((x - mn) / (mx - mn))


# add_dummy_feature()
# Binarizer()
# Imputer()
# LabelBinarizer()
# LabelEncoder()

# StandardScaler()
MinMaxScaler()


print('\n\n\n\n\n\n\n')