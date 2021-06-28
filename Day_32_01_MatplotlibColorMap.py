# Day_32_01_MatplotlibColorMap.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def color_map_1():
    x = np.random.rand(100)
    y = np.random.rand(100)
    # print(x)
    # print(np.min(x), np.max(x))     # 0.005557774706943075 0.9940215123908226

    # plt.plot(x, y, 'ro')
    # plt.scatter(x, y)

    t = np.arange(len(x))

    plt.scatter(x, y, c=t)
    plt.show()


# 문제
# 대각선(x)으로 플롯을 그려보세요
def color_map_2():
    x = np.arange(100)

    plt.figure(figsize=[10, 5])

    plt.subplot(1, 4, 1)
    plt.scatter(x, x, c=x)  # (0, 0), (1, 1), ...
    plt.scatter(x, x[::-1], c=x)  # (0, 99), (1, 98), ...

    plt.subplot(1, 4, 2)
    plt.scatter(x, x, c=x)
    plt.scatter(x, 99 - x, c=x)  # (0, 99-0), (1, 99-1), ...

    plt.subplot(1, 4, 3)
    plt.scatter(x, x, c=x)  # x: 0 ~ 99
    plt.scatter(x, np.flip(x), c=-x)  # -x: 0 ~ -99

    print(cm.viridis(0), cm.viridis(255))
    #    red       green     blue    alpha
    # (0.267004, 0.004874, 0.329415, 1.0) (0.993248, 0.906157, 0.143936, 1.0)

    plt.subplot(1, 4, 4)
    t = np.arange(0, 10000, 100)

    plt.scatter(x, x, c=x)
    # plt.scatter(x, list(reversed(x)), c=-t)
    plt.scatter(x, list(reversed(x)), c=t, cmap="viridis_r")

    plt.tight_layout()
    plt.show()


# 문제
# x 값이 난수일 때, 위와 똑같은 대각선을 그려보세요
def color_map_3():
    x = np.random.rand(100)
    # x.sort()

    plt.scatter(x, x, c=x)
    # plt.scatter(x, x[::-1], c=x)      # wrong
    # plt.scatter(x, -x, c=x)           # wrong
    plt.scatter(x, 1 - x, c=x)

    plt.show()


def color_map_4():
    print(len(plt.colormaps()))  # 166
    print(plt.colormaps())
    # ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    # 'CMRmap', 'CMRmap_r',
    # 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
    # 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
    # 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    # 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
    # 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    # 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',
    # 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r',
    # 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
    # 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
    # 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
    # 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
    # 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r',
    # 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
    # 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    # 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
    # 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r',
    # 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

    # 문제
    # 컬러맵 4가지를 골라서 어떤 색상인지 4개의 플롯에 그려보세요
    x = np.random.rand(100)

    plt.subplot(1, 4, 1)
    plt.scatter(x, x, c=x, cmap="inferno")
    plt.title("inferno")

    plt.subplot(1, 4, 2)
    plt.scatter(x, x, c=x, cmap="ocean")
    plt.title("ocean")

    plt.subplot(1, 4, 3)
    plt.scatter(x, x, c=x, cmap="summer")
    plt.title("summer")

    plt.subplot(1, 4, 4)
    plt.scatter(x, x, c=x, cmap="jet")
    plt.title("jet")

    plt.show()


def color_map_5():
    jet = cm.get_cmap("jet")

    print(jet(-1))
    print(jet(0))
    print(jet(255))
    print(jet(256))
    print("-" * 30)

    print(jet(0.0))
    print(jet(0.1))
    print(jet(0.3))
    print(jet(0.9))
    print(jet(1.0))
    print("-" * 30)

    # 문제
    # 전체 색상에 가운데 색상을 읽어오세요 (2가지)
    print(jet(128 / 255))
    print(jet(0.5))
    print(jet(128))

    print(jet(127 / 255))
    print(jet(127))
    print("-" * 30)

    # 컬러맵 전체 색상 갯수는 256개
    for c in np.arange(0.0, 0.001, 0.0001):
        print("{:.5f}".format(c), jet(c))
    print("-" * 30)

    print(jet([0, 255]))
    print(jet(range(0, 256, 32)))
    print(jet(np.linspace(0, 1.0, 10)))


# heatmap
def color_map_6():
    # 0 ~ 1 사이의 실수
    x = np.random.rand(100).reshape(10, 10)
    x = np.arange(100).reshape(10, 10)
    plt.imshow(x)
    plt.show()

    # 0 ~ 255 사이의 정수
    # x = np.arange(256).reshape(16, -1)
    # x = np.arange(400).reshape(20, -1)
    # plt.imshow(x, cmap="inferno")
    # plt.show()


# color_map_1()
# color_map_2()
# color_map_3()
# color_map_4()
# color_map_5()
color_map_6()
