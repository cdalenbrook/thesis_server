import pandas as pd

df = pd.read_csv('./data.csv')
dummies = pd.get_dummies(df['main colour'])
df = df.join(dummies)
df = df.drop('main colour', axis=1)

df['size'] = df['size'].map({'small': 5, 'medium': 10, 'big': 15})

df.to_csv('./data_preprocessed.csv', index=False)
