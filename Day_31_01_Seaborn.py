# Day_31_01_Seaborn.py
import seaborn as sns
import matplotlib.pyplot as plt


def seaborn_1():
    print(sns.get_dataset_names())
    # ['anagrams', 'anscombe', 'attention', 'brain_networks', 'car_crashes',
    # 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'geyser',
    # 'iris', 'mpg', 'penguins', 'planets', 'tips', 'titanic']

    iris = sns.load_dataset('iris')
    print(iris)
    print()
    print(type(iris))           # <class 'pandas.core.frame.DataFrame'>
    print(iris.columns)
    # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')

    sns.swarmplot(x='species', y='petal_length', data=iris)
    plt.show()


def seaborn_2():
    titanic = sns.load_dataset('titanic')
    print(titanic.columns)
    # Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
    #        'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], dtype='object')
    sns.factorplot('class', 'survived', 'sex',
                   data=titanic, kind='bar', palette='muted', legend=False)
    plt.show()


# 문제
# seaborn 사이트에서 마음에 드는 것을 골라 함수로 구현해서 결과를 확인하세요
def seaborn_3():
    sns.set_theme(style="whitegrid")

    # Load the example dataset of brain network correlations
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

    # Pull out a specific subset of networks
    used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
    used_columns = (df.columns.get_level_values("network")
                    .astype(int)
                    .isin(used_networks))
    df = df.loc[:, used_columns]

    # Compute the correlation matrix and average over networks
    corr_df = df.corr().groupby(level="network").mean()
    corr_df.index = corr_df.index.astype(int)
    corr_df = corr_df.sort_index().T

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 6))

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(data=corr_df, palette="Set3", bw=.2, cut=1, linewidth=1)

    # Finalize the figure
    ax.set(ylim=(-.7, 1.05))
    sns.despine(left=True, bottom=True)
    plt.show()


# seaborn_1()
# seaborn_2()
seaborn_3()
