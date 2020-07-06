import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import heapq
from matplotlib import pyplot as plt


class FeatureSelectionGA():
    def __init__(self, df, df_type: str, crossover_prob: float,  mutation_prob: float, tournament_size: int, num_gens: int):
        super().__init__()
        self.df = df
        self.df_type = df_type
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.num_gens = num_gens

    def generate_initial_pop(self, population_size: int):
        # generate a random population of genotypes
        initial_pop = [[random.randint(0, 1) for i in range(
            population_size)] for j in range(population_size)]
        return initial_pop

    def preprocess_toys(self, df: pd.DataFrame):
        # remove ID and name columns
        df = df.drop(['id', 'toy'], axis=1)
        for column in df:
            # one hot encode string values and remove their corresponding columns
            if df[column].dtype == object:
                one_hot_enc = pd.get_dummies(df[column])
                df = pd.concat([df, one_hot_enc], axis=1)
                df = df.drop([column], axis=1)
        # make last column contain the class values
        df = df[[c for c in df if c not in ['outside']]
                + [c for c in ['outside'] if c in df]]
        print('preprocessed DF', df)
        return df

    def preprocess_cars(self, df: pd.DataFrame):
        df.columns = ['buying price', 'maintenance cost', 'number of doors',
                      'number of persons', 'lug_boot', 'safety', 'decision']
        # map categorical values to numbers
        df['buying price'] = df['buying price'].map(
            {'low': 5, 'medium': 10, 'high': 15, 'vhigh': 20})
        df['maintenance cost'] = df['maintenance cost'].map(
            {'low': 5, 'medium': 10, 'high': 15, 'vhigh': 20})
        df['lug_boot'] = df['lug_boot'].map(
            {'small': 5, 'med': 10, 'big': 15})
        df['safety'] = df['safety'].map(
            {'low': 5, 'med': 10, 'high': 15})
        df['decision'] = df['decision'].map(
            {'unacc': 0, 'acc': 10, 'good': 15, 'vgood': 20})

        # map non-numbers to numbers
        df['number of doors'].replace({'5more': 5}, inplace=True)
        df['number of persons'].replace({'more': 6}, inplace=True)

        # fill in NaN with average
        df['maintenance cost'].fillna(
            df['maintenance cost'].mean(), inplace=True)
        df['buying price'].fillna(df['buying price'].mean(), inplace=True)

        return df

    def fit_trees(self, population: list, df: pd.DataFrame, target: str):
        # run the DT algorithm with the subset of features in the population
        # X contains columns in the population
        accuracies = []
        depths = []
        trees = []
        num_features = []
        feat_importances = []

        # loop through genotypes in population and fit a tree to the given features and then find accuracy & depth
        for pop in population:
            clf = DecisionTreeClassifier(criterion="entropy", random_state=100)
            X = df[target].to_frame()
            count = 0
            # if the feature is in the population (i.e. 1) then add it to the features in X
            for feature in pop:
                if feature == 1:
                    X = pd.concat([X, df.iloc[:, count]], axis=1)
                count += 1
            # remove the target column from the X values
            X = X.drop(target, 1)
            # if there are no features in the final column, skip this population
            if X.empty:
                accuracies.append(0)
                depths.append(0)
                trees.append(None)
                num_features.append(0)
                feat_importances.append(0)
                print('No Tree Made')
                continue
            # make the target column the y values
            y = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=100)
            # fit the tree to the given training values
            clf = clf.fit(X_train, y_train)

            # take the features out of the population that are not used in the tree
            feature_importances = clf.feature_importances_
            change = []
            for j in range(len(feature_importances)):
                if feature_importances[j] == 0:
                    change.append(j)
            count = 0
            for k in range(len(pop)):
                if pop[k] == 1:
                    if count in change:
                        pop[k] = 0
                    count += 1

            # make predictions according to the fitted tree
            y_pred = clf.predict(X_test)
            # get the accuracy of the DT for current population
            pop_accuracy = accuracy_score(y_test, y_pred)
            # get the depth of the DT for current pop
            tree_depth = clf.tree_.max_depth
            # add both values to their respective arrays
            feat_importances.append(feature_importances)
            accuracies.append(pop_accuracy)
            depths.append(tree_depth)
            trees.append(clf)

            # add the number of features used in training to the list of num features
            num_features.append(sum(pop))
        print('trees fitted')
        return accuracies, depths, trees, num_features, feat_importances

    # get the fitnesses of the genomes in the current population
    def get_fitness(self, accuracies: list, depths: list, num_features: list):
        # fitness aims to have high depth, high accuracy and low number of features
        alpha = 0.7
        fitnesses = [0] * len(accuracies)
        for i in range(len(accuracies)):
            fitnesses[i] = depths[i] + \
                2*accuracies[i] - (0.5*num_features[i])
        print('fitnesses calculated')
        return fitnesses

    def tournament_selection(self, population: list, fitnesses: list):
        tournament_members = []
        # chose individuals randomly from the population
        for i in range(self.tournament_size):
            random_int = random.randint(0, len(population)-1)
            tournament_members.append(population[random_int])
        parents = []
        fitness_parents = []
        best = tournament_members[0]
        best_fitness = 0
        # pick the 2 best individuals from the tournament
        while(len(parents) < 4):
            for i in range(len(tournament_members)):
                if fitnesses[i] > best_fitness:
                    best = tournament_members[i]
                    best_fitness = fitnesses[i]
            parents.append(best)
            fitness_parents.append(best_fitness)
        print('tournament completed')
        return parents, fitness_parents

    # get the crossover of the fittest in the population
    def crossover(self, parents: list):
        # get 4 fittest members of the population from tournament selection
        children = parents
        # crossover their attributes
        for i in range(len(parents[0])):
            if random.random() < self.crossover_prob:
                children[0][i], children[1][i] = parents[1][i], parents[0][i]
            if random.random() < self.crossover_prob:
                children[2][i], children[3][i] = parents[3][i], parents[2][i]
        print('crossover performed')
        return children

    # mutate the children with probability pm
    def mutate(self, children: list):
        mutants = children
        for genotype in children:
            for i in range(len(children[0])):
                random_val = random.random()
                if random_val < self.mutation_prob:
                    genotype[i] = 1 - genotype[i]
        print('mutation performed')
        return mutants

    def optimize(self):
        if(self.df_type == 'toys'):
            df = self.preprocess_toys(self.df)
            target = 'outside'
        elif(self.df_type == 'cars'):
            df = self.preprocess_cars(self.df)
            target = 'decision'

        population_size = len(df.columns)-1
        initial_pop = self.generate_initial_pop(population_size)
        pop = initial_pop

        for i in range(self.num_gens):
            accuracies, depths, trees, num_features, feature_importances = self.fit_trees(
                pop, df, target)
            fitnesses = self.get_fitness(accuracies, depths, num_features)
            print(fitnesses)
            parents, fitness_parents = self.tournament_selection(
                pop, fitnesses)
            children = self.crossover(parents)
            mutants = self.mutate(children)
            new_population = []
            # add the parents and the mutants to the next population
            for i in range(4):
                new_population.append(parents[i])
                new_population.append(mutants[i])
            # make the rest of the population random
            while(len(new_population) < population_size):
                new_population.append([random.randint(0, 1)
                                       for i in range(population_size)])
            pop = new_population

        print('Final Population: \n', pop)
        return pop, trees, df, fitnesses, accuracies, depths, feature_importances

    def display_optimized_pop(self, pop: list, trees: list, df: pd.DataFrame, fitnesses: list, accuracies: list, depths: list, feature_importances: list):
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
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 10), dpi=300)
        plot_tree(clf, feature_names=fn, class_names=cn,
                  filled=True, rounded=True, fontsize=14)
        fig.savefig('best-tree.png')


path = "./data_categorized.csv"
df = pd.read_csv(path)
ga = FeatureSelectionGA(df, df_type='toys', crossover_prob=0.6,
                        mutation_prob=0.2, tournament_size=6, num_gens=10)
pop, trees, df, fitnesses, accuracies, depths, feature_importances = ga.optimize()
ga.display_optimized_pop(pop, trees, df, fitnesses,
                         accuracies, depths, feature_importances)
