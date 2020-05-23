import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from utils import get_data_path


def importData(path):
    df = pd.read_csv(path)
    df = df.drop('toy', axis=1)
    return df


def preprocess(df, target, data):
    # make dummy variables for colours
    dummies = pd.get_dummies(df['main colour'])
    df = df.join(dummies)
    df = df.drop('main colour', axis=1)
    # change size into 5, 10, 15 (small, medium, big)
    df['size'] = df['size'].map({'small': 5, 'medium': 10, 'big': 15})
    # col name = target, data matches to col
    df[target] = df['id'].map(data)
    cols_at_end = [target]
    df = df[[c for c in df if c not in cols_at_end]
            + [c for c in cols_at_end if c in df]]
    df = df.drop('id', axis=1)
    print(df)
    return df


def split(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=100)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100)
    clf = clf.fit(X_train, y_train)
    # fn = ['wheels', 'size', 'fluffy', 'outside', 'black',
    #      'blue', 'brown', 'gray', 'green', 'multi', 'pink', 'red', 'white']
    #cn = ['inside', 'outside']
    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 10), dpi=300)
    # plot_tree(clf, feature_names=fn, class_names=cn,
    #          filled=True, rounded=True, fontsize=14)
    # fig.savefig('generated-decision-tree.png')
    return clf


def predict(tree, toy_id, true_category, target, dev: bool):
    path = get_data_path(dev)
    print(path)
    df = importData(path)

    # make dummy variables for colours
    dummies = pd.get_dummies(df['main colour'])
    df = df.join(dummies)
    df = df.drop('main colour', axis=1)
    # change size into 5, 10, 15 (small, medium, big)
    df['size'] = df['size'].map({'small': 5, 'medium': 10, 'big': 15})

    X_test = df.loc[df['id'] == toy_id]
    X_test = X_test.drop('id', axis=1)
    print(X_test)
    y_pred = tree.predict(X_test)

    return y_pred


def accuracy(y_pred, y_test):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    return "Accuracy : %d" % (accuracy_score(y_test, y_pred)*100)


def main(categories, data, dev: bool):
    target = categories[0]
    # import the data without classification
    path = get_data_path(dev)
    df = importData(path)
    if(target in df.columns):
        target = "target"
    # preprocess the data by adding the target column with the given values
    df = preprocess(df, target, data)
    X_train, X_test, y_train, y_test = split(df)
    tree = train(X_train, y_train)
    return tree


# Calling main function
if __name__ == "__main__":
    main()
