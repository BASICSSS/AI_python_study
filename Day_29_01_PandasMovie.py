# Day_29_01_PandasMovie.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


# 문제
# 사용자 파일을 팬다스로 읽어주세요
def read_movie_lens():
    users = pd.read_csv('ml-1m/users.dat', delimiter='::', engine='python', header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('ml-1m/movies.dat', delimiter='::', engine='python', header=None,
                         names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('ml-1m/ratings.dat', delimiter='::', engine='python', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    df = pd.merge(pd.merge(users, ratings), movies)
    # print(df)

    return df


df = read_movie_lens()

# 문제
# 남녀 성별에 따른 영화 평점을 알고 싶어요
by_1 = df.pivot_table(values='Rating', columns='Gender')
print(type(by_1))               # <class 'pandas.core.frame.DataFrame'>
print(by_1)
# Gender         F         M
# Rating  3.620366  3.568879

by_2 = df.pivot_table(values='Rating', index='Gender')
print(type(by_2))               # <class 'pandas.core.frame.DataFrame'>
print(by_2)
#           Rating
# Gender
# F       3.620366
# M       3.568879

# 문제
# pivot_table을 사용하지 말고 남녀 성별 평점 평균을 구하세요 (넘파이 활용)
# 1. 남자와 여자 데이터로 분리
# 2. 평점 데이터만 추출
# 3. 평균 계산
females = df[df.Gender == 'F']
males = df[df.Gender == 'M']
print('F :', np.mean(females.Rating))
print('M :', np.mean(males.Rating))
print('M :', males.Rating.mean())
print('-' * 30)

# 문제
# 남녀 성별과 연령대에 따른 영화 평점을 알고 싶어요
by_3 = pd.DataFrame.pivot_table(df, values='Rating', index='Age', columns='Gender')
print(by_3, end='\n\n')

# ages = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
# # by_3.index = ages
#
# by_3.plot(kind='bar')
# plt.xticks(range(7), ages, rotation=45)
# plt.title('Rating')
# plt.show()

by_4 = pd.DataFrame.pivot_table(df, values='Rating', index=['Age', 'Gender'])
print(by_4, end='\n\n')

# 문제
# 여자 25살 연령대의 평점을 출력하세요
print(by_4.iloc[0])
print(by_4.iloc[1])
print(by_4.iloc[2])
# Rating    3.616291
# Name: (1, F), dtype: float64
# Rating    3.517461
# Name: (1, M), dtype: float64
# Rating    3.453145
# Name: (18, F), dtype: float64

print(by_4.loc[25])
#          Rating
# Gender
# F       3.60670
# M       3.52678

print(by_4.loc[25].loc['F'])
# Rating    3.6067
# Name: F, dtype: float64

print(by_4.loc[25, 'F'])
# Rating    3.6067
# Name: (25, F), dtype: float64
print('-' * 30)

print(by_4.unstack())               # index를 columns로 이동
print(by_4.unstack().stack())       # columns를 index로 이동
print('-' * 30)

# 문제
# 평점에 대해 연령대별로 구분하고, 성별과 직업으로 구분해서 출력하세요
by_5 = pd.DataFrame.pivot_table(df, values='Rating', index='Age', columns=['Gender', 'Occupation'])
print(by_5, end='\n\n')

# 0을 넣건 중앙값을 넣건 틀렸다. 행별/열별 평균을 넣어야 하는데, 값 하나로 처리할 수 없다
by_6 = df.pivot_table(values='Rating', index='Age', columns=['Gender', 'Occupation'], fill_value=(1 + 5) // 2)
print(by_6, end='\n\n')

# by_7 = df.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=np.mean)
# by_7 = df.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc='mean')
by_7 = df.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=[np.mean, np.sum])
print(by_7, end='\n\n')
#             mean               sum
# Gender         F         M       F        M
# Age
# 1       3.616291  3.517461   31921    64665
# 18      3.453145  3.525476  156866   486900
# 25      3.606700  3.526780  329436  1072903
# 35      3.659653  3.604434  181054   538971
# 45      3.663044  3.627942   88316   215946
# 50      3.797110  3.687098   68591   200674
# 56      3.915534  3.720327   36019   110051

# print(by_7['mean'])       # mean과 sum은 컬럼 이름

# 문제
# 평균과 합계 데이터프레임을 by_7에서처럼 결합하세요
by_8_1 = df.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=np.mean)
by_8_2 = df.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=np.sum)

print(pd.concat([by_8_1, by_8_2], axis=0), end='\n\n')
print(pd.concat([by_8_1, by_8_2], axis=1), end='\n\n')
