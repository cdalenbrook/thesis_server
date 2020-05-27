from sklearn.tree import DecisionTreeClassifier
from typing import List

all_trees = {}
targets = {}
user_truths = {}


def store_tree(id: str, tree: DecisionTreeClassifier):
    all_trees[id] = tree


def store_targets(id: str, target: List[str]):
    targets[id] = target


def store_usertruth(id: str, data):
    user_truths[id] = data


def get_tree(id: str):
    if(id in all_trees):
        return all_trees[id]
    return None


def get_targets(id: str):
    if(id in targets):
        return targets[id]
    return None


def get_usertruth(id: str):
    if(id in user_truths):
        return user_truths[id]
    return None


def get_data_path(dev: bool) -> str:
    return './data_preprocessed.csv' if not dev else './data-test.csv'
