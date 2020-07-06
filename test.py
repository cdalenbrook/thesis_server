import pandas as pd
from decisiontree import split

df = pd.read_csv('./data.csv')

X_train, X_test, y_train, y_test = split(df)
print(type(X_test))
