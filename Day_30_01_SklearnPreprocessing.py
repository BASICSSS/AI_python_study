# Day_30_01_SklearnPreprocessing.py
from sklearn import preprocessing
import numpy as np


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




