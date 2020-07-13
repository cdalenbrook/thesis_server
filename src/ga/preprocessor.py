import pandas as pd
from abc import ABC, abstractmethod
import numpy as np


class Preprocessor(ABC):

    @abstractmethod
    def preprocess(self, df: pd.DataFrame.T, target):
        print("Preprocessing started!")


class CarsPreprocessor(Preprocessor):

    def preprocess(self, df: pd.DataFrame.T, target) -> pd.DataFrame.T:
        super().preprocess(df, target)
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
            {'unacc': 0, 'acc': 10, 'good': 10, 'vgood': 10})

        # map non-numbers to numbers
        df['number of doors'].replace({'5more': 5}, inplace=True)
        df['number of persons'].replace({'more': 6}, inplace=True)

        # fill in NaN with average
        df['maintenance cost'].fillna(
            df['maintenance cost'].mean(), inplace=True)
        df['buying price'].fillna(df['buying price'].mean(), inplace=True)

        return df


class ToysPreprocessor(Preprocessor):

    def preprocess(self, df: pd.DataFrame.T, target) -> pd.DataFrame.T:
        super().preprocess(df, target)
        # remove ID and name columns
        df = df.drop(['id', 'toy'], axis=1)
        for column in df:
            # one hot encode string values and remove their corresponding columns
            if df[column].dtype == object:
                one_hot_enc = pd.get_dummies(df[column])
                df = pd.concat([df, one_hot_enc], axis=1)
                df = df.drop([column], axis=1)
        # make last column contain the class values
        df = df[[c for c in df if c not in [target]]
                + [c for c in [target] if c in df]]
        return df


class ToysPreprocessorRandom(Preprocessor):

    def preprocess(self, df: pd.DataFrame.T, target) -> pd.DataFrame.T:
        super().preprocess(df, target)
        # remove ID and name columns
        df = df.drop(['id', 'toy'], axis=1)
        for column in df:
            # one hot encode string values and remove their corresponding columns
            if df[column].dtype == object:
                one_hot_enc = pd.get_dummies(df[column])
                df = pd.concat([df, one_hot_enc], axis=1)
                df = df.drop([column], axis=1)
        # make last column contain the class values
        df = df[[c for c in df if c not in [target]]
                + [c for c in [target] if c in df]]

        # randomize class values
        df[target] = np.random.randint(0, 2, df.shape[0])

        return df
