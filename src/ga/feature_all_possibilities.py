import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import heapq
from matplotlib import pyplot as plt
from preprocessor import Preprocessor
from preprocessor import ToysPreprocessor
from tqdm import tqdm
import functools


class FeatureSelectionAllPosibilities():
    def __init__(self, preprocessor: Preprocessor, dataframe: pd.DataFrame.T, crossover_prob: float,  mutation_prob: float, tournament_size: int, num_gens: int):
        super().__init__()
        self.df = preprocessor.preprocess(dataframe, 'outside')
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.num_gens = num_gens
        self.population_size = 12
        self.initial_pop = self.generate_initial_pop()
        self.pop = self.initial_pop.copy()
        self.values = ['outside', 'girls', 'green', 'big', 'has_wheels']

    def generate_initial_pop(self):
        # generate a random population of genotypes
        initial_pop = [[random.randint(0, 1) for i in range(
            self.population_size)] for j in range(self.population_size)]
        return initial_pop

    def fit_trees(self):
        # run the DT algorithm with the subset of features in the population
        # X contains columns in the population
        new_pop = []
        accuracies = []
        depths = []
        trees = []
        num_features = []
        feat_importances = []

        # loop through genotypes in population and fit a tree to the given features and then find accuracy & depth
        for genotype in self.pop:
            depth_sum = 0
            accuracy_sum = 0
            for value in self.values:
                clf = DecisionTreeClassifier(
                    criterion="entropy", random_state=100)
                path = 'data/data_categorized_' + value + '.csv'
                target_df = pd.read_csv(path)

                individual = genotype.copy()

                X = target_df[value].to_frame()

                count = 0
                # if the feature is in the population (i.e. 1) then add it to the features in X
                for feature in individual:
                    if feature == 1:
                        X = pd.concat([X, self.df.iloc[:, count]], axis=1)
                    count += 1
                # remove the target column from the X values
                X = X.drop(value, 1)

                # if there are no features in the final column, skip this population
                if X.empty:
                    accuracy_sum += 0
                    depth_sum += 0
                    continue

                # make the target column the y values
                y = target_df.iloc[:, -1]

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
                for k in range(len(individual)):
                    if individual[k] == 1:
                        if count in change:
                            individual[k] = 0
                        count += 1

                # make predictions according to the fitted tree
                y_pred = clf.predict(X_test)
                # get the accuracy of the DT for current population
                pop_accuracy = accuracy_score(y_test, y_pred)
                # get the depth of the DT for current pop
                tree_depth = clf.tree_.max_depth

                accuracy_sum += (pop_accuracy)
                depth_sum += tree_depth

            accuracy = (accuracy_sum/len(self.values))
            depth = (depth_sum/len(self.values))

            new_pop.append(genotype)
            accuracies.append(accuracy)
            depths.append(depth)

        self.pop = new_pop.copy()
        return accuracies, depths

    # get the fitnesses of the genomes in the current population
    def get_fitness(self, accuracies: list, depths: list):
        # fitness aims to have high depth, high accuracy and low number of features
        fitnesses = [0] * len(accuracies)
        for i in range(len(accuracies)):
            if(depths[i] > 4):
                fitnesses[i] = -1 + 4*accuracies[i]
            else:
                fitnesses[i] = depths[i] + \
                    4*accuracies[i]
        return fitnesses

    def tournament_selection(self, fitnesses: list):
        parents = []

        while(len(parents) < 4):
            tournament_members = []
            fitnesses_tmembers = []
            # choose individuals randomly from the population
            for i in range(self.tournament_size):
                random_int = random.randint(0, (len(self.pop)-1))
                tournament_members.append(self.pop[random_int])
                fitnesses_tmembers.append(fitnesses[random_int])

            best = tournament_members[0]
            best_fitness = 0
            # pick the best individuals from the tournament
            for i in range(len(tournament_members)):
                if fitnesses_tmembers[i] > best_fitness:
                    best = tournament_members[i]
            parents.append(best)

        return parents

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
        return children

    # mutate the children with probability pm
    def mutate(self, children: list):
        mutants = children
        for genotype in children:
            for i in range(len(children[0])):
                random_val = random.random()
                if random_val < self.mutation_prob:
                    genotype[i] = 1 - genotype[i]
        return mutants

    def optimize(self):
        for i in tqdm(range(self.num_gens)):
            accuracies, depths = self.fit_trees()
            fitnesses = self.get_fitness(accuracies, depths)
            parents = self.tournament_selection(fitnesses)
            children = self.crossover(parents)
            mutants = self.mutate(children)
            new_population = []
            # add the parents and the mutants to the next population
            for i in range(len(parents)):
                new_population.append(parents[i])
                new_population.append(mutants[i])
            # make the rest of the population random
            while(len(new_population) < self.population_size):
                new_population.append([random.randint(0, 1)
                                       for i in range(self.population_size)])
            self.pop = new_population.copy()

        # calculate info for final population
        accuracies, depths = self.fit_trees()
        fitnesses = self.get_fitness(accuracies, depths)

        return self.pop, self.df, fitnesses, accuracies, depths


df = pd.read_csv('data/data_categorized_outside.csv')
ga = FeatureSelectionAllPosibilities(ToysPreprocessor(
), dataframe=df, crossover_prob=0.6, mutation_prob=0.2, tournament_size=8, num_gens=10)

pop, df, fitnesses, accuracies, depths = ga.optimize()

print(accuracies)
