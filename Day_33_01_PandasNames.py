# Day_33_01_PandasNames.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm

# 문제
# 구글에서 "미국 신생아 이름" 데이터를 검색한 후에 yob1880.txt 파일이나 가져와서 출력하세요
names = pd.read_csv('data/yob1880.txt',
                    header=None, names=['name', 'gender', 'births'])
print(names)
print('-' * 30)

# 문제
# 남자와 여자 아기 이름의 갯수를 알려주세요 (2가지)
# print(names.gender == 'F')

w_bools = (names.gender == 'F')
m_bools = (names.gender == 'M')
print('w: {}, m: {}'.format(sum(w_bools), sum(m_bools)))
print('w: {}, m: {}'.format(w_bools.sum(), m_bools.sum()))
print(w_bools.value_counts())
print('-' * 30)

women = names[w_bools]
men = names[m_bools]
print('w: {}, m: {}'.format(len(women), len(men)))
print('w: {}, m: {}'.format(women.shape[0], men.shape[0]))
print('-' * 30)

print(names.groupby('gender').size())
print('-' * 30)

print(names.pivot_table(values='name', index='gender', aggfunc=np.count_nonzero))
print('-' * 30)

print(names.gender.value_counts())

print(''.join(names.gender).count('F'))
print(''.join(names.gender).count('M'))
print('-' * 30)

# 문제
# names에 들어있는 여자와 남자 아이들의 이름 합계를 구하세요 (2가지)
print('w: {}, m: {}'.format(women.births.sum(), men.births.sum()))
print('w: {}, m: {}'.format(sum(w_bools * names.births), sum(m_bools * names.births)))

print(names.groupby('gender').sum())
print(names.groupby('gender').sum('births'))
print(names.groupby('gender').births.sum())
print('-' * 30)

# 문제
# 출생 인원이 높은 top5를 막대 그래프로 그려보세요 (컬러맵을 사용해서 막대 색상을 모두 다르게 표시)
# print(men)

top5 = men.sort_values('births', ascending=False)
top5 = top5[:5]
# print(top5.drop(['gender'], axis=1))

top5.index = top5.name
del top5['name']
print(top5)

# top5.plot(kind='bar')
# top5.plot(kind='bar', y='births', color=colors.TABLEAU_COLORS)

winter = cm.get_cmap('winter')
bar_colors = winter(np.linspace(0, 1, 5))
# top5.plot(kind='bar', y='births', color=bar_colors)
# top5.plot(kind='pie', y='births')
# top5.plot(kind='pie', y='births', legend=False)
# top5.plot(kind='pie', y='births', legend=False, autopct='%2.1f%%')

# plt.bar(names[names.gender == 'M'].head().name,
#         names[names.gender == 'M'].head().births,
#         color=colors.TABLEAU_COLORS)

# plt.pie(top5.births)
# plt.pie(top5.births, labels=top5.index)
# plt.pie(top5.births, labels=top5.index, autopct='%2.1f%%')
# plt.show()







