# Conducts the targeted data pruning for the Pruning_and_Influence_Analysis datasets
import pandas as pd
import numpy as np


# Pruning Strategies
def rating_based(dataset: pd.DataFrame, lw_bound: float, up_bound: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the number of ratings given by the users. Users > lw_bound to <= up_bound
    percentiles of the ratings number are removed from the dataset.
    :param dataset: A DataFrame containing the original dataset with user ratings
    :param lw_bound: Lower bound of the user group which is pruned
    :param up_bound: Upper bound of the user group which is pruned
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    # Group the ratings DataFrame by 'user' and count the number of ratings per user
    ratings_per_user = dataset.groupby('user')['rating'].count()

    # Sort users by the number of ratings each user has
    ratings_per_user_srt = ratings_per_user.sort_values(ascending=True).reset_index()

    # Calculate the nth percentiles of the 'Number of Ratings'
    upper_threshold = ratings_per_user_srt['rating'].quantile(up_bound)
    lower_threshold = ratings_per_user_srt['rating'].quantile(lw_bound)
    remaining_unique_users = ratings_per_user_srt[(ratings_per_user_srt['rating'] <= lower_threshold) or
                                                  (ratings_per_user_srt['rating'] > upper_threshold)]['user'].unique()
    removed_unique_users = ratings_per_user_srt[(ratings_per_user_srt['rating'] > lower_threshold) and
                                                (ratings_per_user_srt['rating'] <= upper_threshold)]['user'].unique()

    return remaining_unique_users, removed_unique_users


def performance_based(results: pd.DataFrame, lw_bound: float, up_bound: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the performance data from the baseline recommendations. Users > lw_bound to <= up_bound
    percentiles of the performance are removed from the dataset.
    :param results: DataFrame containing the performance data (NDCG Values) of the baseline recommendations
    :param lw_bound: Lower bound of the user group which is pruned
    :param up_bound: Upper bound of the user group which is pruned
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    # Sort users based on the NDCG values
    results_srt = results.sort_values(by=['ndcg'], ascending=[True]).reset_index()

    # Calculate the nth percentiles of the 'NDCG' depending on the step size
    upper_threshold = results_srt['ndcg'].quantile(up_bound)
    lower_threshold = results_srt['ndcg'].quantile(lw_bound)

    remaining_unique_users = results_srt[(results_srt['ndcg'] <= lower_threshold) or (results_srt['ndcg'] >
                                                                                    upper_threshold)]['user'].unique()
    removed_unique_users = results_srt[(results_srt['ndcg'] > lower_threshold) and (results_srt['ndcg'] <=
                                                                                    upper_threshold)]['user'].unique()
    return remaining_unique_users, removed_unique_users


def influence_mean_based(influence_data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the mean user influence. Users above the threshold are removed from the dataset.
    :param influence_data: A DataFrame containing the user influence data like mean influence or number of
                        influenced users
    :param threshold: Gives the threshold for the influence score, above which users are pruned from the dataset
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    remaining_unique_users = influence_data[(influence_data['Mean'] < threshold)]['user'].unique()
    removed_unique_users = influence_data[(influence_data['Mean'] >= threshold)]['user'].unique()
    return remaining_unique_users, removed_unique_users


def influence_cl_mean_based(influence_data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the cleaned mean user influence. Users above the threshold are removed from the dataset.
    :param influence_data: A DataFrame containing the user influence data like mean influence or number of
                        influenced users
    :param threshold: Gives the threshold for the influence score, above which users are pruned from the dataset
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    remaining_unique_users = influence_data[(influence_data['Clean Mean'] < threshold)]['user'].unique()
    removed_unique_users = influence_data[(influence_data['Clean Mean'] >= threshold)]['user'].unique()
    return remaining_unique_users, removed_unique_users


def influence_diff_based(influence_data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the difference between positively and negatively influenced users.
    Users below the threshold are removed from the dataset.
    :param influence_data: A DataFrame containing the user influence data like mean influence or number of
                        influenced users
    :param threshold: Gives the threshold for the influence score, below which users are pruned from the dataset
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    remaining_unique_users = influence_data[(influence_data['Influence Diff'] > threshold)]['user'].unique()
    removed_unique_users = influence_data[(influence_data['Influence Diff'] <= threshold)]['user'].unique()
    return remaining_unique_users, removed_unique_users


def influence_score_based(influence_data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the user influence score. Users below the threshold are removed from the dataset.
    :param influence_data: A DataFrame containing the user influence data like mean influence or number of
                        influenced users
    :param threshold: Gives the threshold for the influence score, below which users are pruned from the dataset
    :return remaining_unique_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_unique_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    remaining_unique_users = influence_data[(influence_data['Influence Score'] > threshold)]['user'].unique()
    removed_unique_users = influence_data[(influence_data['Influence Score'] <= threshold)]['user'].unique()
    return remaining_unique_users, removed_unique_users


def random_based(dataset: pd.DataFrame, pr_perc: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset randomly based on the given percentage of users to be pruned.
    :param dataset: Dataset to be pruned
    :param pr_perc: Percentage of users to be pruned, corresponds to lw_bound
    :return remaining_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    # Get unique users
    unique_users = dataset['user'].unique()
    # Calculate the number of users to be removed
    num_removed = int(len(unique_users) * pr_perc)
    # Randomly select users to be removed
    removed_users = np.random.choice(unique_users, num_removed, replace=False)
    # Get remaining users
    remaining_users = unique_users[~np.isin(unique_users, removed_users)]
    return remaining_users, removed_users


# Pruning Function
def prune(eval_data: pd.DataFrame, strategy: str, lw_bound: float = 1.0, up_bound: float = 1.0) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prunes the dataset based on the given strategy
    :param eval_data: The data to be pruned. Depending on the strategy, this can be ratings data, performance data
                    or user influence data
    :param strategy: The pruning strategy to be used. Can be 'rating_based', 'performance_based',
                    'influence_mean_based', 'influence_score_based', 'influence_diff_based' or 'none'
    :param lw_bound: The lower bound of the user group which is pruned, corresponds to threshold for
                    influence based pruning and pr_perc for random pruning
    :param up_bound: The upper bound of the user group which is pruned
                    Example: lw_bound = 0.95 & up_bound = 1.0 prunes the users in the 95th percentile of a given metric
    :return remaining_users: A DataFrame containing the user IDs of the remaining users in the pruned dataset
    :return removed_users: A DataFrame containing the user IDs which are removed from the dataset
    """
    if strategy == 'rating_based':
        remaining_users, removed_users = rating_based(eval_data, lw_bound, up_bound)
    elif strategy == 'performance_based':
        remaining_users, removed_users = performance_based(eval_data, lw_bound, up_bound)
    elif strategy == 'influence_mean_based':
        remaining_users, removed_users = influence_mean_based(eval_data, lw_bound)
    elif strategy == 'influence_cl_mean_based':
        remaining_users, removed_users = influence_cl_mean_based(eval_data, lw_bound)
    elif strategy == 'influence_score_based':
        remaining_users, removed_users = influence_score_based(eval_data, lw_bound)
    elif strategy == 'influence_diff_based':
        remaining_users, removed_users = influence_diff_based(eval_data, lw_bound)
    elif strategy == 'random_based':
        remaining_users, removed_users = random_based(eval_data, lw_bound)
    elif strategy == 'none':
        remaining_users = eval_data['user'].unique()
        removed_users = []
    else:
        raise ValueError('Invalid pruning strategy')
    return remaining_users, removed_users
