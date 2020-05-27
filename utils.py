from sklearn.tree import DecisionTreeClassifier
from typing import List

all_trees = {}
targets = {}
user_truths = {}


def store_tree(id: str, tree: DecisionTreeClassifier):
    all_trees[id] = tree
    print("Stored tree at: ", id)
    print(all_trees)


def store_targets(id: str, target: List[str]):
    targets[id] = target
    print("Stored targets at: ", id)


def store_usertruth(id: str, data):
    user_truths[id] = data
    print("Stored user_truths at: ", id)


def get_tree(id: str):
    if(id in all_trees):
        return all_trees[id]
    print("Could not find tree for: ", id)
    return None


def get_targets(id: str):
    if(id in targets):
        return targets[id]
    print("Could not find targets for: ", id)
    return None


def get_usertruth(id: str):
    if(id in user_truths):
        return user_truths[id]
    print("Could not find user_truths for: ", id)
    return None


def get_data_path(dev: bool) -> str:
    return './data_preprocessed.csv' if not dev else './data-test.csv'
