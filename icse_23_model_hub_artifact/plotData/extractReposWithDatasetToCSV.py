from pandas import DataFrame, Series
import pandas
import numpy as np
from ast import literal_eval
from pandas.core import series

from pandas.core.algorithms import value_counts

def main()  ->  None:
    df: DataFrame = pandas.read_csv("data.csv")
    datasetSeries: Series = df["datasets"].dropna()
    
    expandedList: list = []
    for foo in datasetSeries:
        try:
            convertedList: list = literal_eval(foo)
            for bar in convertedList:
                expandedList.append(bar)
        except ValueError:
            expandedList.append(foo)
        except SyntaxError:
            expandedList.append(foo)
        
    data: Series = Series(expandedList)
    data: DataFrame = DataFrame(data, columns=["dataset"])
    data.to_csv("simplified_ds.csv")

if __name__ == "__main__":
    main()
