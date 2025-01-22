# Functions to analyze recommendation data and identify and separate user groups
# Splitting users based on a separation strategy and analyzing the recommendation results for each user group
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import pdist, squareform


def rec_analysis(dataset: pd.DataFrame, results: pd.DataFrame, separation_strategy: str, step_size: float = 0.1) -> \
        list[pd.DataFrame]:
    """
    Analyze the recommendation results for different user groups by separating user groups based on a separation strategy
    :param dataset: A DataFrame containing the dataset
    :param results: A DataFrame containing the recommendation results
    :param separation_strategy: A string representing the separation strategy, either 'rating_based',
                                    'performance_based', influence_based or 'none'
    :param step_size: A float value between 0 and 1, representing the relative size of the user groups
    :return analysis_results: A list of DataFrames, each representing one user group, containing the NDCG values
                                for each user
    """
    if separation_strategy == 'rating_based':
        ratings_per_user = dataset.groupby('user')['rating'].count()
        user_groups = sep_rating_based(dataset, step_size)
        group_count = 1
        all_group_results = []
        for group in user_groups:
            user_ids = group['user'].unique()
            valid_user_ids = [user_id for user_id in user_ids if user_id in results.index.get_level_values(1)]
            group_results = results.loc[(slice(None), valid_user_ids), :].copy()
            group_results.loc[:, 'number of ratings'] = group_results.index.get_level_values(1).map(
                ratings_per_user.to_dict())
            all_group_results.append(group_results)
            group_count += 1
        analysis_results = all_group_results
    elif separation_strategy == 'performance_based':
        user_groups = sep_performance_based(dataset, results, step_size)
        group_count = 1
        all_group_results = []
        for group in user_groups:
            user_ids = group['user'].unique()
            valid_user_ids = [user_id for user_id in user_ids if user_id in results.index.get_level_values(1)]
            group_results = results.loc[(slice(None), valid_user_ids), :].copy()
            all_group_results.append(group_results)
            group_count += 1
        analysis_results = all_group_results
    elif separation_strategy == 'none':
        analysis_results = [results]
    else:
        raise ValueError('Invalid separation strategy')
    return analysis_results


# Separating users based on the number of ratings they have given
# Separation starts with the users with the least number of ratings
def sep_rating_based(dataset: pd.DataFrame, step_size: float) -> list[pd.DataFrame]:
    """
    Separate the users based on the number of ratings they have given
    :param dataset: DataFrame containing the dataset
    :param step_size: A float value between 0 and 1, representing the step size for the separation. Separation is done
                    along the number of ratings percentiles
    :return group_list: A list of DataFrames, each representing one user group, separated based on the number of ratings
    """
    # Group the ratings DataFrame by 'user' and count the number of ratings per user
    ratings_pu = dataset.groupby('user')['rating'].count()
    # Sort users by the number of ratings each user has
    ratings_pu_srt = ratings_pu.sort_values(ascending=True).reset_index()
    # Calculate the nth percentiles of the 'Number of Ratings' depending on the step size
    # to separate the users into groups
    quant_up = 0
    group_list = []
    for i in range(1, int(1 / step_size) + 1):
        # Calculate the nth percentiles of the 'Number of Ratings'
        quant_lw = quant_up
        quant_up = round(quant_up + step_size, 5)
        perc_group_lw = ratings_pu_srt['rating'].quantile(quant_lw)
        perc_group_up = ratings_pu_srt['rating'].quantile(quant_up)
        if i == 1:
            group = ratings_pu_srt[ratings_pu_srt['rating'] <= perc_group_up]
        else:
            group = ratings_pu_srt[(ratings_pu_srt['rating'] <= perc_group_up) and
                                   (ratings_pu_srt['rating'] > perc_group_lw)]
        group = dataset[dataset['user'].isin(group['user'])].reset_index(drop=True)
        group_list.append(group)

    return group_list


def sep_performance_based(dataset: pd.DataFrame, results: pd.DataFrame, step_size: float = 0.1) -> list[pd.DataFrame]:
    """
    Separate the users based on the performance of the baseline and separate them into groups
    :param dataset: A DataFrame containing the dataset
    :param results: A DataFrame containing the recommendation results
    :param step_size: A float value between 0 and 1, representing the step size for the separation. Separation is done
                    along performance percentiles
    :return group_list: A list of DataFrames, each representing one user group, separated based on the ndcg values
    """
    # Sort users based on the NDCG values
    results_srt = results.sort_values(by=['ndcg'], ascending=[True]).reset_index()
    # Calculate the nth percentiles of the 'NDCG' depending on the step size
    quant_up = 0
    group_list = []
    for i in range(1, int(1 / step_size) + 1):
        quant_lw = quant_up
        quant_up = round(quant_up + step_size, 5)
        perc_group_lw = results_srt['ndcg'].quantile(quant_lw)
        perc_group_up = results_srt['ndcg'].quantile(quant_up)
        if i == 1:
            group = results_srt[results_srt['ndcg'] <= perc_group_up]
        else:
            group = results_srt[(results_srt['ndcg'] <= perc_group_up) and
                                (results_srt['ndcg'] > perc_group_lw)]
        group = dataset[dataset['user'].isin(group['user'])].reset_index(drop=True)
        group_list.append(group)
    return group_list


def similarity(ndcg_delta: np.array, method: str) -> np.array:
    """
    Calculate the similarity between users based on the NDCG delta values
    :param ndcg_delta: A matrix with the NDCG delta vectors for each user
    :param method: The desired similarity method, either 'cosine' or 'euclidean'
    :return sim_matrix: A matrix containing the similarity values between users
    """
    if method == 'cosine':
        # Calculate cosine similarity
        sim_matrix = 1 - squareform(pdist(ndcg_delta, 'cosine'))
    elif method == 'euclidean':
        # Calculate euclidean distance
        sim_matrix = squareform(pdist(ndcg_delta, 'euclidean'))
    else:
        raise ValueError('Invalid similarity method')
    return sim_matrix


def mean_influence(ndcg_delta: np.array) -> pd.DataFrame:
    # TODO: Use the index to user mapping to get the user IDs
    """
    Calculate the mean influence for each user
    :param ndcg_delta: A matrix with the NDCG delta vectors for each user
    :return influence_mean: A DataFrame containing the mean influence data for each user. This includes the columns:
                            For details see the description of function influence_score.
    """
    influence_mean = []
    ndcg_delta_cl = np.copy(ndcg_delta)
    for i in range(len(ndcg_delta)):
        ndcg_delta_cl[i][i] = 0
        influence_mean.append([i, np.mean(ndcg_delta[i]), np.mean(ndcg_delta_cl[i]), np.std(ndcg_delta[i])])
    influence_mean_df = pd.DataFrame(influence_mean, columns=['user', 'Mean', 'Clean Mean', 'Std'])
    return influence_mean_df


def influence_capture(ndcg_delta: np.array) -> pd.DataFrame:
    # TODO: Use the index to user mapping to get the user IDs
    """
    Count the number of users that each user influences
    :param ndcg_delta: A matrix with the NDCG delta vectors for each user
    :return influence_capture_df: A DataFrame containing the influence characteristics for each user.
                                    For details see the description of function influence_analysis.
    """
    influence_capture_df = []
    for i in range(len(ndcg_delta)):
        pos_count = 0
        neg_count = 0
        inf_sum = np.sum(ndcg_delta[i])
        for j in range(len(ndcg_delta)):
            if ndcg_delta[i][j] < 0:
                pos_count += 1
            elif ndcg_delta[i][j] > 0:
                neg_count += 1
        if neg_count or pos_count > 0:
            influence_capture_df.append(
                [i, neg_count, pos_count, neg_count + pos_count, (neg_count + 1) / (pos_count + 1),
                 pos_count - neg_count, inf_sum])
        else:
            influence_capture_df.append([i, 0, 0, 0, 0, 0, 0])
    influence_capture_df = pd.DataFrame(influence_capture_df, columns=['user', 'Negative Influence Count',
                                                                       'Positive Influence Count', 'Total Count',
                                                                       'Influence Ratio', 'Influence Diff',
                                                                       'Influence Sum'])
    return influence_capture_df


# User Features
def item_count(dataset: pd.DataFrame, relative: bool = True) -> pd.DataFrame:
    """
    Calculates the average number of interactions the items a user has rated have.
    :param dataset: The dataset containing the user-item interactions
    :param relative: Boolean value to determine if the average item count should be relative to the mean item count
    :return:
    """
    ratings_data = dataset[['user', 'item']]
    items_count = ratings_data.groupby('item').size().reset_index(name='Item Count')
    ratings_data_item_count = ratings_data.merge(items_count, on='item', how='left')
    user_avg_counts = ratings_data_item_count.groupby('user')['Item Count'].mean().reset_index(name='Average Item Count')

    if relative:
        mean_item_count = items_count['Item Count'].mean()
        user_avg_counts['Average Item Count'] = user_avg_counts['Average Item Count'] / mean_item_count

    return user_avg_counts


def calculate_average_distances(dataset: pd.DataFrame, metric: str = 'cosine', k: int = 50) -> pd.DataFrame:
    """
    Calculate the average distance of each user to its k-nearest neighbours  in the dataset.
    This is a measure of how central the user is in the user-item matrix.
    :param dataset: The dataset containing the user-item interactions
    :param metric: The metric to use for the distance calculation. Choose between 'cosine' and 'euclidean'.
    :param k: The number of nearest neighbors to consider for the average distance calculation
    :return user_distance_df: A DataFrame containing the average distance of each user to all other users in the
                                dataset.
    """
    user_item_matrix = dataset.pivot_table(index='user', columns='item', aggfunc='size', fill_value=0)

    if metric == 'cosine':
        distances = cosine_distances(user_item_matrix)
    elif metric == 'euclidean':
        distances = euclidean_distances(user_item_matrix)
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    average_distances = []
    for i in range(distances.shape[0]):
        sorted_distances = np.sort(distances[i])[1:k + 1]
        avg_distance = np.mean(sorted_distances)
        average_distances.append(avg_distance)

    user_distance_df = pd.DataFrame({
        'user': user_item_matrix.index,
        'Centrality': average_distances
    })

    return user_distance_df


# TODO: Add item_count and calculate_average_distances to the user features
def influence_analysis(ndcg_delta: np.array, dataset: pd.DataFrame, results: pd.DataFrame, user_to_index_map: list = None,
                       values_relative: bool = True) -> pd.DataFrame:
    """
    Function to calculate the influence score and add different user characteristics to the influence_score DataFrame.
    :param ndcg_delta: A matrix with the NDCG delta vectors for each user describing the influence of each user
    :param dataset: The orginal dataset with all users
    :param results: A DataFrame with the mean NDCG values for each user (baseline recommendations)
    :param user_to_index_map: A list containing the mapping of the user IDs to the index in the ndcg_delta matrix
    :param values_relative: Boolean value to determine if the absolute values should be converted to relative values.
                            Converts ['Mean', 'Clean Mean', 'Std', 'Negative Influence Count',
                            'Positive Influence Count', 'Total Count', 'Influence Diff', 'Rating Age Diff',
                            'Rating Age Std Diff'] to relative values.
    :return influence_analysis: DataFrame containing the influence metrics and additional user characteristics.
                                The DataFrame contains the following columns:
                                - Mean: The mean influence of each user
                                    (sum of divergence from the baseline / number of users)
                                - Clean Mean: The mean influence of each user without the influence of the user on
                                    himself
                                - Std: The standard deviation of the influence of each user
                                - Negative Influence Count: The number of users that each user negatively influences
                                - Positive Influence Count: The number of users that each user positively influences
                                - Total Count: The total number of users that each user influences
                                - Influence Ratio: The Influence Ratio of each user
                                    (Negative Influence Count / Positive Influence Count)
                                - Influence Diff: The difference between the Negative Influence Count and the
                                    Positive Influence Count (Positive Influence Count - Negative Influence Count)
                                - Influence Sum: The sum of the influence of each user
                                - Influence Score: Influence Diff - Clean Mean, depending on the values_relativ
                                    parameter Influence Score can be calculated with relative or absolute value.
                                    Lower values indicates a worse influence.
                                - Number of Ratings: The total number of ratings each user has given for the given
                                    dataset
                                - User NDCG: The mean NDCG value of each user
                                - Rating Age: The average timestamp of the users ratings
                                - Rating Age Diff: The difference between the average timestamp of the ratings and the
                                    timestamp of the users ratings (Rating Age - avg_timestamp)
                                    Diff > 0: The user has rated more recent than the average user
                                    Diff < 0: The user has rated older than the average user
                                - Rating Age Std: The standard deviation of the timestamp of the users ratings
                                - Rating Age Std Diff: The Difference in the standard deviation of the user ratings
                                    timestamp and the dataset timestamp std (Rating Age Std - std_timestamp)
                                - Average Item Count: The average number of total interactions the items a user has
                                    rated have.
                                - Centrality: The average distance of each user to its 50-nearest neighbours in the
                                    dataset.
    """
    # Calculate the influence metrics
    mean = mean_influence(ndcg_delta)
    count = influence_capture(ndcg_delta)
    influence_analysis_df = pd.merge(mean, count, on='user')

    if user_to_index_map is None:
        influence_analysis_df['user'] = influence_analysis_df['user'] + 1
    else:
        influence_analysis_df['user'] = influence_analysis_df['user'].map(lambda x: user_to_index_map[x])

    # Calculate average values for baseline
    bl_mean = results['ndcg'].mean()
    bl_std = results['ndcg'].std()
    user_count = len(dataset['user'].unique())

    # Convert absolute values to relative values
    if values_relative:
        influence_analysis_df['Mean'] = influence_analysis_df['Mean'] / bl_mean
        influence_analysis_df['Clean Mean'] = influence_analysis_df['Clean Mean'] / bl_mean
        influence_analysis_df['Std'] = influence_analysis_df['Std'] / bl_std
        influence_analysis_df['Negative Influence Count'] = (influence_analysis_df['Negative Influence Count'] /
                                                             user_count)
        influence_analysis_df['Positive Influence Count'] = (influence_analysis_df['Positive Influence Count'] /
                                                             user_count)
        influence_analysis_df['Total Count'] = influence_analysis_df['Total Count'] / user_count
        influence_analysis_df['Influence Diff'] = influence_analysis_df['Influence Diff'] / user_count
        influence_analysis_df['Influence Score'] = (influence_analysis_df['Influence Diff'] -
                                                    influence_analysis_df['Clean Mean'])
    else:
        influence_analysis_df['Influence Score'] = (influence_analysis_df['Influence Diff'] -
                                                    influence_analysis_df['Clean Mean'])

    # Add additional user characteristics
    influence_analysis_df['Number of Ratings'] = influence_analysis_df['user'].map(dataset.groupby('user').size())
    influence_analysis_df['User NDCG'] = influence_analysis_df['user'].map(results.groupby('user')['ndcg'].mean())
    avg_distances = calculate_average_distances(dataset)
    influence_analysis_df['Centrality'] = influence_analysis_df['user'].map(avg_distances.set_index('user')['Centrality'])
    avg_item_count = item_count(dataset, relative=values_relative)
    influence_analysis_df['Average Item Count'] = influence_analysis_df['user'].map(
        avg_item_count.set_index('user')['Average Item Count'])
    if 'timestamp' in dataset.columns:
        avg_timestamp = dataset['timestamp'].mean()
        std_timestamp = dataset['timestamp'].std()
        influence_analysis_df['Rating Age'] = influence_analysis_df['user'].map(
            dataset.groupby('user')['timestamp'].mean())
        influence_analysis_df['Rating Age Std'] = influence_analysis_df['user'].map(
            dataset.groupby('user')['timestamp'].std())
        influence_analysis_df['Rating Age Diff'] = influence_analysis_df['Rating Age'] - avg_timestamp
        influence_analysis_df['Rating Age Std Diff'] = influence_analysis_df['Rating Age Std'] - std_timestamp
        if values_relative:
            influence_analysis_df['Rating Age Diff'] = influence_analysis_df['Rating Age Diff'] / avg_timestamp
            influence_analysis_df['Rating Age Std Diff'] = influence_analysis_df['Rating Age Std Diff'] / std_timestamp

    return influence_analysis_df


def group_analysis(influence_df: pd.DataFrame, user_group: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    # TODO: Add a function to analyze user groups based on their characteristics
    pass


def pruning_effect_analysis(base_results: pd.DataFrame, pruned_results: pd.DataFrame, inf_analysis_df: pd.DataFrame,
                            removed_users: pd.DataFrame) -> tuple[float, float]:
    """
    Function to analyze if pruning had the expected effect on performance based on the influence data of the
    removed users.
    :param base_results: The NDCG values of the baseline recommendations
    :param pruned_results: The NDCG values of the pruned recommendations
    :param inf_analysis_df: DataFrame containing the influence data for each user
    :param removed_users: DataFrame containing the user IDs which are removed from the dataset
    :return actual_mean_diff: The actual mean difference in NDCG between the pruned and the baseline recommendations
        (relative value)
    :return expected_mean_diff: The expected mean difference in NDCG based on the influence data of the removed users
        (relative value)
    """
    # Actual performance difference
    base_mean_ndcg = base_results['ndcg'].mean()
    pruned_mean_ndcg = pruned_results['ndcg'].mean()
    actual_mean_diff = pruned_mean_ndcg - base_mean_ndcg
    removed_users = inf_analysis_df[inf_analysis_df['user'].isin(removed_users)]
    expected_mean_diff = np.sum(removed_users['Mean'])
    print("_______________________________________________________")
    print("Actual Mean Difference: ", actual_mean_diff / base_mean_ndcg)
    print("Expected Mean Difference: ", expected_mean_diff / base_mean_ndcg)
    print("_______________________________________________________")
    return actual_mean_diff, expected_mean_diff
