# Day_30_02_PandasMoviepy
import pandas as pd

import Day_29_01_PandasMovie

# 특정 영화에 대한 평점 계산
# 여성들이 좋아하는 톱텐


def get_index_500(df):
    freq = df.groupby('Title').size()
    print(freq, end='\n\n')
    # print(type(freq))           # <class 'pandas.core.series.Series'>

    # 문제
    # 500번 이상의 영화들만 고르세요
    bools = (freq >= 500)
    index500 = freq[bools]
    print(index500)

    # return index500.index
    return index500.index.values


def show_favorite(rating500):
    # 문제
    # 여성들이 선호하는 영화를 알려주세요
    # top_female = pd.DataFrame.sort_values(rating500, by='F')
    top_female = rating500.sort_values(by='F', ascending=False)
    print(top_female, end='\n\n')

    # 문제
    # 남성들에 대해 여성들이 선호하는 영화를 알려주세요
    rating500['Diff'] = rating500['F'] - rating500['M']
    print(rating500, end='\n\n')
    print(rating500.sort_values(by='Diff', ascending=False), end='\n\n')

    # 문제
    # 남녀 성별에 따른 호불호가 갈리지 않는 영화를 알려주세요
    # rating500['Dist'] = abs(rating500['F'] - rating500['M'])
    # rating500['Dist'] = (rating500['F'] - rating500['M']).abs()
    rating500['Dist'] = rating500['Diff'].abs()
    print(rating500.sort_values(by='Dist'), end='\n\n')


df = Day_29_01_PandasMovie.read_movie_lens()

# 문제
# 영화에 대해 성별 평점을 구하세요
by_gender = df.pivot_table(values='Rating', index='Title', columns='Gender')
print(by_gender)

# 문제
# 영화평이 500개 이상인 영화들의 제목을 구하세요
index500 = get_index_500(df)
print(index500)
print('-' * 30)

# 문제
# by_gender와 index500을 사용해서 평점 갯수가 500개 이상인 영화들에 대한 평점만 구하세요
rating500 = by_gender.loc[index500]
print(rating500)

show_favorite(rating500)













