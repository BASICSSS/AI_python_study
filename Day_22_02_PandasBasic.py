# Day_22_02_PandasBasic.py
import pandas as pd


# DataFrame: 테이블(표), 엑셀 시트
# Series: DataFrame 객체에서 컬럼을 담당하는 객체
def series_basic():
    a = pd.Series([5, 1, 2, 9])
    print(a)
    print()
    # 0    5
    # 1    1
    # 2    2
    # 3    9
    # dtype: int64

    print(a.index)              # RangeIndex(start=0, stop=4, step=1)
    print(a.values)             # [5 1 2 9]
    print(type(a.values))       # <class 'numpy.ndarray'>
    print('-' * 30)

    b = pd.Series([5, 1, 2, 9], index=['a', 'b', 'c', 'd'])
    print(b)
    print()
    # a    5
    # b    1
    # c    2
    # d    9
    # dtype: int64

    print(b.index)
    print(b.values)
    print()

    # 문제
    # 입력 데이터 중에서 처음에 있는 5만 가져오고 싶습니다 (3가지)
    print(b['a'])
    print(b[0])
    print(b.values[0])
    print(b[-4])
    print(b.a)
    print()

    # 문제
    # 중간에 포함된 1과 2만 출력하세요 (3가지)
    # print(type(b[1:-1]))        # <class 'pandas.core.series.Series'>

    print(b['b':'c'].values)
    print(b[1:-1].values)
    print(b[1:3].values)
    print(b.values[1:-1])


def dataframe_basic():
    df = pd.DataFrame({
        'city': ['jeju', 'jeju', 'jeju', 'gunsan', 'gunsan', 'gunsan'],
        'year': [2018, 2019, 2020, 2018, 2019, 2020],
        'population': [300, 400, 350, 500, 550, 600],
    })
    print(df)
    print()
    #      city  year  population
    # 0    jeju  2018         300
    # 1    jeju  2019         400
    # 2    jeju  2020         350
    # 3  gunsan  2018         500
    # 4  gunsan  2019         550
    # 5  gunsan  2020         600

    print(df.index)         # RangeIndex(start=0, stop=6, step=1)
    print(df.columns)       # Index(['city', 'year', 'population'], dtype='object')
    print(df.values)
    print()

    print(type(df.head()), end='\n\n')      # <class 'pandas.core.frame.DataFrame'>

    print(df.head(), end='\n\n')
    print(df.tail(), end='\n\n')

    print(df.head(2), end='\n\n')
    print(df.tail(2), end='\n\n')
    print('-' * 30)

    df.info()
    print('-' * 30)

    df.index = ['one', 'two', 'three', 'four', 'five', 'six']
    print(df)
    #          city  year  population
    # one      jeju  2018         300
    # two      jeju  2019         400
    # three    jeju  2020         350
    # four   gunsan  2018         500
    # five   gunsan  2019         550
    # six    gunsan  2020         600
    print('-' * 30)

    print(df['population'])
    print(df.population)
    print(type(df.population))      # <class 'pandas.core.series.Series'>
    print('-' * 30)

    # 문제
    # 군산의 2019년 행을 출력하세요 (3가지)
    # print(df.ix['five'])          # 에러
    # print(df.ix[0])               # 에러

    print(type(df.loc['five']))     # <class 'pandas.core.series.Series'>
    print(df.loc['five'])
    print(df.iloc[4])
    # city          gunsan
    # year            2019
    # population       550
    # Name: five, dtype: object

    print(df.values[4])             # ['gunsan' 2019 550]
    print('-' * 30)

    # 문제
    # 데이터프레임에서 군산 데이터 전체를 출력하세요 (3가지)
    print(df.loc['four':])
    print(df.loc['four':'six'])
    print(df.loc['four':'six'].values)

    print(df.iloc[3:])
    print(df.iloc[3:].values)
    print(df.values[3:])
    print('-' * 30)

    # print(df[1])              # 에러
    print(df[3:])               # 성공. iloc 단순 버전
    print()
    #         city  year  population
    # four  gunsan  2018         500
    # five  gunsan  2019         550
    # six   gunsan  2020         600

    # print(df['city':'year'])  # 에러
    print(df['four':])          # 성공


# series_basic()
dataframe_basic()
