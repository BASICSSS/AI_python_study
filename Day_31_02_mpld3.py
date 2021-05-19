# Day_31_02_mpld3.py
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import seaborn as sns


def mpld3_1():
    def sample_plot(df, title):
        df.plot(kind='line',
                marker='p',
                color=['blue', 'red'],
                lw=3,
                ms=20,
                alpha=0.7)

        plt.title(title)
        plt.text(s='blue line', x=2.5, y=4, color='blue')
        plt.text(s='red line', x=2.5, y=3, color='red')


    c1 = [1, 2, 3, 4]
    c2 = [1, 4, 2, 3]

    df = pd.DataFrame({'c1': c1, 'c2': c2})
    # sample_plot(df, 'base')

    # plt.xkcd()
    # sample_plot(df, 'xkcd')
    # plt.show()

    sample_plot(df, 'mpld3')
    mpld3.show()


# 문제
# seaborn 사이트에서 마음에 드는 것을 골라 함수로 구현해서 결과를 확인하세요 (mpld3 사용)
def mpld3_2():
    sns.set_theme()

    # Load the brain networks example dataset
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

    # Select a subset of the networks
    used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    used_columns = (df.columns.get_level_values("network")
                    .astype(int)
                    .isin(used_networks))
    df = df.loc[:, used_columns]

    # Create a categorical palette to identify the networks
    network_pal = sns.husl_palette(8, s=.45)
    network_lut = dict(zip(map(str, used_networks), network_pal))

    # Convert the palette to vectors that will be drawn on the side of the matrix
    networks = df.columns.get_level_values("network")
    network_colors = pd.Series(networks, index=df.columns).map(network_lut)

    # Draw the full plot
    g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                       row_colors=network_colors, col_colors=network_colors,
                       dendrogram_ratio=(.1, .2),
                       cbar_pos=(.02, .32, .03, .2),
                       linewidths=.75, figsize=(12, 13))

    g.ax_row_dendrogram.remove()
    mpld3.show()


# mpld3_1()
mpld3_2()
