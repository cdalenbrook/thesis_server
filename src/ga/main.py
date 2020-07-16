import pandas as pd
from feature_selection_ga import FeatureSelectionGA
from feature_selection_5_categories import FeatureSelection5Categories
from preprocessor import ToysPreprocessor, CarsPreprocessor
from display import display_optimized_pop


if __name__ == "__main__":
    for i in range(10):
        # path of dataset to preprocess
        path = "data/car_evaluation.csv"
        df = pd.read_csv(path)
        # select the target variable
        target = 'decision'
        # select which data set to preprocess
        pre = CarsPreprocessor()

        ga = FeatureSelectionGA(
            df,
            target=target,
            pre=pre,
            crossover_prob=0.6,
            mutation_prob=0.2,
            tournament_size=8,
            num_gens=200
        )

        pop, trees, df, fitnesses, accuracies, depths, feature_importances, generation_acc, generation_fit = ga.optimize()

        accuracy, depth, fitness, best_features = display_optimized_pop(
            df=df,
            pop=pop,
            trees=trees,
            fitnesses=fitnesses,
            accuracies=accuracies,
            depths=depths,
            feature_importances=feature_importances,
        )

        # print results to text file
        print('New Run', file=open("experimental_output.txt", "a"))
        print(i, file=open("experimental_output.txt", "a"))
        print('accuracy', file=open("experimental_output.txt", "a"))
        print(accuracy, file=open("experimental_output.txt", "a"))
        print('depth', file=open("experimental_output.txt", "a"))
        print(depth, file=open("experimental_output.txt", "a"))
        print('fitness', file=open("experimental_output.txt", "a"))
        print(fitness, file=open("experimental_output.txt", "a"))
        print('best_features', file=open("experimental_output.txt", "a"))
        print(best_features, file=open("experimental_output.txt", "a"))
