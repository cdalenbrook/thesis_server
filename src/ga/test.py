import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


# df.columns = ['buying price', 'maintenance cost', 'number of doors',
#               'number of persons', 'lug_boot', 'safety', 'decision']
# # map categorical values to numbers
# df['buying price'] = df['buying price'].map(
#     {'low': 5, 'medium': 10, 'high': 15, 'vhigh': 20})
# df['maintenance cost'] = df['maintenance cost'].map(
#     {'low': 5, 'medium': 10, 'high': 15, 'vhigh': 20})
# df['lug_boot'] = df['lug_boot'].map(
#     {'small': 5, 'med': 10, 'big': 15})
# df['safety'] = df['safety'].map(
#     {'low': 5, 'med': 10, 'high': 15})
# df['decision'] = df['decision'].map(
#     {'unacc': 0, 'acc': 10, 'good': 10, 'vgood': 10})

# # map non-numbers to numbers
# df['number of doors'].replace({'5more': 5}, inplace=True)
# df['number of persons'].replace({'more': 6}, inplace=True)

# # fill in NaN with average
# df['maintenance cost'].fillna(
#     df['maintenance cost'].mean(), inplace=True)
# df['buying price'].fillna(df['buying price'].mean(), inplace=True)


df = pd.read_csv('data/data_categorized_outside.csv')

df = df.drop(['id', 'toy'], axis=1)

df['main colour'] = df['main colour'].map(
    {'blue': 5, 'brown': 10, 'green': 15, 'multi': 20, 'pink': 25, 'red': 30, 'white': 35})

df['size'] = df['size'].map(
    {'small': 5, 'medium': 10, 'big': 15})
# make last column contain the class values
# df = df[[c for c in df if c not in ['outside']]
#         + [c for c in ['outside'] if c in df]]

clf = DecisionTreeClassifier(criterion="entropy", random_state=100)

X = df.iloc[:, 0:4]

print(X)


# df.iloc[:, -1] = np.random.randint(0, 2, size=len(df))
y = df.iloc[:, -1]
print(y)


clf = DecisionTreeClassifier(criterion="entropy", random_state=100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
tree_depth = clf.tree_.max_depth

print(acc)
print(tree_depth)

fn = ['wheels', 'main colour', 'size', 'fluffy']
cn = ['inside', 'outside']
# cn = ['unacc', 'acc', 'good', 'vgood']
fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(25, 10), dpi=300)
plot_tree(clf, feature_names=fn, class_names=cn,
          filled=True, rounded=True, fontsize=14)
fig.savefig('ga_best_tree.png')
