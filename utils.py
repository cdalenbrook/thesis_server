from sklearn.tree import DecisionTreeClassifier
from typing import List

all_trees = {}
targets = {}


def store_tree(tree: DecisionTreeClassifier, id: str, target: List[str]):
    all_trees[id] = tree
    targets[id] = target[0]


def get_tree(id: str):
    if(id in all_trees):
        return all_trees[id]
    return None


def get_target(id: str):
    if(id in targets):
        return targets[id]
    return None
