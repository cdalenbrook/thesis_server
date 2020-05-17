import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing

df = pd.read_csv('./data_outside_vs_inside.csv')

dummies1 = pd.get_dummies(df['size'])
dummies2 = pd.get_dummies(df['main colour'])

df = df.join(dummies1)
df = df.join(dummies2)
df = df.drop('size', axis=1)
df = df.drop('main colour', axis=1)
