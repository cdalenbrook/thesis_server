import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
import server.utilities
import os


def importData(path):
    df = pd.read_csv(path)
    df = df.drop('toy', axis=1)
    return df


def preprocess(df, target, data):
    # col name = target, data matches to col
    df[target] = df['id'].map(data)
    cols_at_end = [target]
    df = df[[c for c in df if c not in cols_at_end]
            + [c for c in cols_at_end if c in df]]
    df = df.drop('id', axis=1)
    # print(df)
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
    return clf


def predict(tree, toy_id):
    path = server.utilities.get_data_path()
    df = importData(path)
    X_test = df.loc[df['id'] == toy_id]
    X_test = X_test.drop('id', axis=1)

    y_pred = tree.predict(X_test)
    return y_pred.tolist()


def accuracy(y_pred, y_test):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    return "Accuracy : %d" % (accuracy_score(y_test, y_pred)*100)


def create_tree(categories, data):
    target = categories[0]
    # import the data without classification
    path = server.utilities.get_data_path()
    df = importData(path)
    if(target in df.columns):
        target = "target"
    # preprocess the data by adding the target column with the given values
    df = preprocess(df, target, data)
    X_train, X_test, y_train, y_test = split(df)
    tree = train(X_train, y_train)
    return tree


def tree2image(tree, categories):
    clf = tree
    fn = ["wheels", "size", "fluffy", "blue",
          "brown", "green", "multi", "pink", "red", "white"]
    cn = categories
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(25, 10), dpi=300)
    plot_tree(clf, feature_names=fn, class_names=cn,
              filled=True, rounded=True, fontsize=14)

    # Save figure image to a bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_string = base64.b64encode(buffer.read())
    return img_string


# Calling main function
# if __name__ == "__main__":
#     from firebase_admin import credentials, storage, initialize_app
#     import cloud_storage

#     # Initialize Firestore DB
#     cred = credentials.Certificate('src/key.json')
#     firebase_app = initialize_app(cred)
#     storage_bucket = storage.bucket(cloud_storage.BUCKETNAME)
#     _id = "123456"

#     tree = create_tree(["outside", "inside"], {
#         "bike_001": 1,
#         "car_001": 0,
#         "lego_001": 0,
#         "trampoline_001": 0,
#         "doll_001": 0,
#         "teddy_001": 0,
#         "xylophone_001": 0,
#         "scooter_001": 1,
#         "slide_001": 1,
#         "swing_001": 1,
#         "piano_001": 0,
#         "train_001": 0,
#         "tractor_001": 1,
#         "play_doh_001": 0
#     })
#     prediction = predict(tree, "teddy_001", 0, 1)
#     print(prediction)
#     img = tree2image(tree, ["outside", "inside"])
#     tree_image_url = cloud_storage.upload_image(
#         storage_bucket, f'trees/{_id}.png', img)
#     print(tree_image_url)
