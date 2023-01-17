from pandas import DataFrame
import pandas
import numpy as np

def main()  ->  None:
    df: DataFrame = pandas.read_csv("data.csv")
    print(df.shape)
    df["architecture"].replace(to_replace="None", value=np.nan, inplace=True)
    df.dropna(inplace=True)
    df.to_csv("simplified.csv")
    return None

if __name__ == "__main__":
    main()
