from sklearn.tree import DecisionTreeClassifier
from typing import List
import sklearn_json as skljson
import json as JSON
import os


def tree2json(tree: DecisionTreeClassifier):
    return JSON.dumps(skljson.to_dict(tree))


def json2tree(tree_json):
    tree = JSON.loads(tree_json)
    print(f'Found the tree: {type(tree)}')
    deserialized = skljson.from_dict(tree)
    print(f'Deserialized tree of type: {type(deserialized)}')
    return deserialized


def get_data_path() -> str:
    path = os.path.join(os.getcwd(), 'data/data_preprocessed.csv')
    return path
