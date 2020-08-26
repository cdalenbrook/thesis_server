from sklearn.tree import DecisionTreeClassifier
from typing import List
import sklearn_json as skljson
import json as JSON
import os


def tree2json(tree: DecisionTreeClassifier):
    """Transforms a DTC tree into JSON"""
    return JSON.dumps(skljson.to_dict(tree))


def json2tree(tree_json):
    """Transforms a json tree into a DTC"""
    tree = JSON.loads(tree_json)
    print(f'Found the tree: {type(tree)}')
    deserialized = skljson.from_dict(tree)
    print(f'Deserialized tree of type: {type(deserialized)}')
    return deserialized


def get_data_path() -> str:
    """Get the path of where the data for generating the decision tree should come from"""
    path = os.path.join(os.getcwd(), 'data/data_preprocessed.csv')
    return path


def get_test_data_path() -> str:
    """Get the path of where the data for new example to test the decision tree should come from"""
    path = os.path.join(os.getcwd(), 'data/test_toys.csv')
    return path
