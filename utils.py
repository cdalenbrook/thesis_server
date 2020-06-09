from sklearn.tree import DecisionTreeClassifier
from typing import List
import sklearn_json as skljson
import json as JSON


def store_tree(tree: DecisionTreeClassifier):
    return skljson.to_dict(tree)


def get_tree(tree_json):
    return skljson.from_dict(tree_json)


def get_data_path(dev: bool) -> str:
    return './data_preprocessed.csv' if not dev else './data-test.csv'
