import pandas as pd
from feature_selection_ga import FeatureSelectionGA
from preprocessor import ToysPreprocessor
from display import display_optimized_pop

if __name__ == "__main__":
    path = "data/data_categorized.csv"
    df = pd.read_csv(path)
    pre = ToysPreprocessor()
    ga = FeatureSelectionGA(
        df,
        target='outside',
        preprocessor=pre,
        crossover_prob=0.6,
        mutation_prob=0.2,
        tournament_size=6,
        num_gens=100
    )
    # accuracies, depths, trees, num_features, feat_importances = ga.fit_trees()
    # toys target: "outside"
    # cars target: "decision"
    # print(accuracies)
    pop, trees, df, fitnesses, accuracies, depths, feature_importances = ga.optimize()
    # assert(len(feature_importances) == len(pop))
    # assert(len(trees) == len(pop))
    # assert(len(fitnesses) == len(pop))
    # assert(len(accuracies) == len(pop))
    # assert(len(depths) == len(pop))
    # ga.display_optimized_pop(pop, trees, df, fitnesses,
    #                          accuracies, depths, feature_importances)
    display_optimized_pop(
        df=df,
        pop=pop,
        trees=trees,
        fitnesses=fitnesses,
        accuracies=accuracies,
        depths=depths,
        feature_importances=feature_importances
    )
