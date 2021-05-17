# Day_30_02_PandasMoviepy
import Day_29_01_PandasMovie

# 특정 영화에 대한 평점 계산, 여성들이 좋아하는 톱텐.


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






