import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


def display_optimized_pop(pop: list, trees: list, df: pd.DataFrame, fitnesses: list, accuracies: list, depths: list, feature_importances: list):
    fittest_idx = fitnesses.index(max(fitnesses))
    print('Fittest Index: ', fittest_idx)
    fittest = pop[fittest_idx]
    print('Fittest Individual: ', fittest)

    print('Feature Importances: ', feature_importances[fittest_idx])

    features = list(df.columns)
    features.pop()
    best_features = []
    for i in range(len(fittest)):
        if(fittest[i] == 1):
            best_features.append(features[i])

    print('Most Important Features', best_features)

    print('Accuracy: ', accuracies[fittest_idx])
    print('Depth: ', depths[fittest_idx])
    print('Fitness: ', fitnesses[fittest_idx])

    clf = trees[fittest_idx]
    fn = features
    cn = ['inside', 'outside']
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(25, 10), dpi=300)
    plot_tree(clf, feature_names=fn, class_names=cn,
              filled=True, rounded=True, fontsize=14)
    fig.savefig('best-tree.png')
