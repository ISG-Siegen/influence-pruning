import random

import pandas as pd
import numpy as np


# Function to create cross-validation folds
def cross_validation(dataset: pd.DataFrame, n_splits: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create cross-validation folds for the given dataset
    :param dataset: Data to perform cross-validation on
    :param n_splits: Number of folds
    :return fold_results: List of tuples containing train and test sets as pd.DataFrame for each fold
    """
    users = dataset.groupby('user')
    folds = [[] for _ in range(n_splits)]

    for user, items in users:
        shuffled_items = items.sample(frac=1, random_state=42).reset_index(drop=True)
        user_folds = np.array_split(shuffled_items, n_splits)
        for i in range(n_splits):
            folds[i].append(user_folds[i])
    folds = [pd.concat(fold).reset_index(drop=True) for fold in folds]
    fold_results = []
    for i in range(n_splits):
        test_set = folds[i]
        train_set = pd.concat([folds[j] for j in range(n_splits) if j != i], ignore_index=True)
        fold_results.append((train_set, test_set))

    return fold_results


def n_core_pruning(dataset: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """
    Prune the dataset to only include users and items with at least min_interactions interactions
    :param dataset: Data to be pruned
    :param min_interactions: Minimum number of interactions a user or item should have
    :return pruned_dataset: Pruned dataset
    """
    pruned = True
    pruned_dataset = dataset
    while pruned:
        user_counts = dataset['user'].value_counts()
        item_counts = dataset['item'].value_counts()
        pruned_dataset = dataset[
            dataset['user'].isin(user_counts[user_counts >= min_interactions].index) &
            dataset['item'].isin(item_counts[item_counts >= min_interactions].index)
        ]
        if len(pruned_dataset) == len(dataset):
            pruned = False
        dataset = pruned_dataset
    return pruned_dataset
