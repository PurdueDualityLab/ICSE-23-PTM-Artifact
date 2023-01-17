from matplotlib import pyplot as plt
from pandas import DataFrame, Series
import pandas
from pandas.core.algorithms import value_counts
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure

def main()  ->  None:
    df: DataFrame = pandas.read_csv("simplified.csv")
    data: Series = df["architecture"].value_counts(sort=True, ascending=True)

    otherCount: int = 0
    idx: str
    for idx in data.index:
        if data.get(key=idx) < 25:
            otherCount += data.get(key=idx)
            data.drop(idx, inplace=True)

    df: DataFrame = pandas.read_csv("simplified_ds.csv")
    data2: Series = df["dataset"].value_counts(sort=True, ascending=True)

    otherCount: int = 0
    idx: str
    for idx in data2.index:
        if data2.get(key=idx) < 75:
            otherCount += data2.get(key=idx)
            data2.drop(idx, inplace=True)

    print(data)
    
    fig: Figure = data.plot(kind="barh", figsize=(80, 80), fontsize=140).get_figure()
    fig2: Figure = data2.plot(kind="barh", figsize=(80, 80), fontsize=140).get_figure()
    
    ult, ((fig, fig2)) = plt.subplots(1, 2)

    # plt.figure(fig)
    # plt.ylabel("Architecture", fontsize=180, fontweight="bold")
    # plt.xlabel("Number of Models", fontsize=180, fontweight="bold")
    # plt.xticks(rotation=45)

    # plt.tight_layout()
    ult.savefig("test.pdf")

if __name__ == "__main__":
    main()
