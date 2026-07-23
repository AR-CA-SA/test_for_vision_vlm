import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def box_plot_maker():
    df = pd.read_csv("impactNoiseDataUseful.csv")

    frequency_distribution = []
    object_names = []
    n_obj = []

    for i in range(5):
        frequency_distribution.append(df.iloc[i, 4:7].to_list())
        n_obj.append(i+1)
        object_names.append(df.iloc[i, 0])
    print(object_names)

    print(frequency_distribution)

    fig, ax = plt.subplots()
    ax.boxplot(frequency_distribution)
    ax.set_xticks(n_obj, object_names)

    plt.ylabel("Dominant Frequency (Hz)")
    plt.tight_layout()
    plt.show()


box_plot_maker()