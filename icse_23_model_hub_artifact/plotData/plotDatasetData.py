from matplotlib import pyplot as plt
from pandas import DataFrame, Series
import pandas
from pandas.core.algorithms import value_counts
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure

def main()  ->  None:
    df: DataFrame = pandas.read_csv("simplified_ds.csv")
    data: Series = df["dataset"].value_counts(sort=True, ascending=True)
    
    otherCount: int = 0
    idx: str
    for idx in data.index:
        if data.get(key=idx) < 75:
            otherCount += data.get(key=idx)
            data.drop(idx, inplace=True)
    
    # data = data.append(Series({"Other": otherCount}))

    print(data)

    fig: Figure = data.plot(kind="barh", figsize=(80, 80), fontsize=140).get_figure()
    plt.figure(fig)
    plt.ylabel("Dataset", fontsize=180, fontweight='bold')
    plt.xlabel("Number of Models", fontsize=180, fontweight='bold')
    # plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("test.pdf")

if __name__ == "__main__":
    main()
