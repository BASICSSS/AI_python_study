# Day_34_01_PandasMovie.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm

names = pd.read_csv("data/yob1880.txt", header=None, names=["name", "gender", "births"])

# 문제
# 남자와 여자 이름에 공통으로 사용된 이름을 찾아보세요
common = names.groupby("name").size()
print(common, end="\n\n")
print(common > 1, end="\n\n")
print(common[common > 1], end="\n\n")
print(common[common > 1].index, end="\n\n")
print("-" * 30)

# 문제
# 중복된 이름에 대해 남자와 여자의 빈도를 알려주세요
#         F   M
# Addie  31 156
# Allie 250   7
by_gender = names.pivot_table(values="births", index="name", columns="gender")
print(by_gender, end="\n\n")

over_1 = common[common > 1].index.values
print(by_gender.loc[over_1], end="\n\n")
print(by_gender.loc[over_1].astype(int), end="\n\n")
print("-" * 30)

# 문제
# np.where 함수를 사용해서 중복 이름을 알려주세요
# where: 참의 위치를 알려주는 함수
# where를 사용하는 방법 1 (조건 직접 전달)
pos = np.where(common > 1)
# print(pos)

# where를 사용하는 방법 2 (불린 배열 전달)
bools = common > 1
pos = np.where(bools)
print(pos)
print(by_gender.iloc[pos], end="\n\n")
print("-" * 30)

print(by_gender)

# group_by 함수를 사용하지 않는 버전 1
f = pd.Series.notna(by_gender.F)  # notna 결측치가 아닌것을 알려줌
m = pd.Series.notna(by_gender.M)
print(pd.concat([f, m], axis=1), end="\n\n")
print(by_gender[f == m])
print("-" * 30)

# group_by 함수를 사용하지 않는 버전 2
# 문제
# isna 함수를 사용해서 중복된 이름을 찾아보세요
f = by_gender.F.isna()  # notna 와 반대로 결측치인 것을 알려줌
# f = pd.Series.isna(by_gender.F)
m = by_gender.M.isna()
print(pd.concat([f, m], axis=1), end="\n\n")
# print(by_gender[f != m], end='\n\n')        # wrong
print("============")
exit()
# print((f != m))
# print(not (f != m))                       # error
# print(~(f != m))                          # 넘파이 not 연산
print(by_gender[~(f != m)], end="\n\n")
print(by_gender[f == m], end="\n\n")  # 결측치가 아니면 False가 나오고, 양쪽 모두 False인 경우
print(by_gender[~f == ~m], end="\n\n")
print("============")
print(by_gender[np.logical_and(f == False, m == False)], end="\n\n")
print(by_gender[np.logical_and(~f, ~m)], end="\n\n")

# T T
# T F
# F T
# F F       존재하지 않는 경우

# merge 함수 버전
# m_name = names.name[names.gender == 'M']
# f_name = names.name[names.gender == 'F']
# print(f_name)
# print(m_name)
# print(pd.merge(m_name, f_name))
# print('-' * 30)


print("\n\n\n\n\n\n\n\n")

