import numpy as np
import pandas as pd
import json
import lenskit
from lenskit.datasets import ML100K, ML1M
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn, user_knn
from lenskit.algorithms.basic import Popular
from pandas import DataFrame
from data_pruning import prune
from data_analysis import rec_analysis, influence_analysis
from rec_utility import cross_validation, n_core_pruning
from load_datasets import load_data_lastfm, load_data_amazon_luxury_beauty, load_data_amazon_digital_music

# Load the ML100K dataset
ml100k = ML100K(r'Datasets\ml-100k')
ml100k_data = ml100k.ratings

# Load the ML1M dataset
ml1m = ML1M(r'Datasets\ml-1m')
ml1m_data = ml1m.ratings

# Load the Last.FM dataset
lastfm_data = n_core_pruning(load_data_lastfm())

# Load the Amazon Luxury and Beauty dataset
amazon_luxury_data = n_core_pruning(load_data_amazon_luxury_beauty())

# Load the Amazon Digital Music dataset
amazon_digital_music_data = n_core_pruning(load_data_amazon_digital_music())

# Set up algorithms
# ALS has to be set up in the pipeline to ensure reproducibility
algo_KNNuu = user_knn.UserUser(57, feedback='implicit')  # Hyperparameter <70 optimized
algo_KNNii = item_knn.ItemItem(62, feedback='implicit')  # Hyperparameter <70 optimized


def rec_execute(data: pd.DataFrame, partitions: int, algorithm: str, pruning_strategy: str, lw_bound: float,
                up_bound: float, eval_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function receives a dataset and partitions it into training and test data, to perform recommendations
    :param data: The dataset to be partitioned
    :param partitions: The number of partitions
    :param algorithm: The algorithm to be used for the recommendations as a string,
                    either 'UserUser', 'ItemItem' or 'ALS'
    :param pruning_strategy: The desired pruning strategy as a string, either 'rating_based',
                    'performance_based' or 'none'
    :param lw_bound: Lower bound for the user group to be pruned
    :param up_bound: Upper bound for the user group to be pruned
    :param eval_data: Evaluation Data necessary for certain pruning strategies, like the results of the baseline
        recommendations or the user influence data
    :return all_recs: A DataFrame or list of DataFrames containing the recommendations for each user
    :return test_data: A DataFrame or list of DataFrames containing the test data
    """
    all_recs = []
    test_data = []

    # Prune the dataset and get the user IDs
    remaining_user_ids, removed_user_ids = prune(eval_data, pruning_strategy, lw_bound, up_bound)
    # print("Remaining Users: " + str(len(remaining_user_ids)))
    # print("Removed Users: " + str(len(removed_user_ids)))

    if partitions == 1:
        # Hold out
        for train, test in xf.partition_users(data[['user', 'item']], partitions, xf.SampleFrac(0.2, rng_spec=42),
                                              rng_spec=42):
            # Filter the train and test sets to only include the user IDs from the pruned dataset
            train_remain = train[train['user'].isin(remaining_user_ids)].reset_index(drop=True)
            test_remain = test[test['user'].isin(remaining_user_ids)].reset_index(drop=True)
            train_pruned = train[train['user'].isin(removed_user_ids)]
            test_pruned = test[test['user'].isin(removed_user_ids)]
            test_data.append(test)
            if algorithm == 'UserUser':
                all_recs.append(rec(algorithm, algo_KNNuu, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    all_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
            elif algorithm == 'ItemItem':
                all_recs.append(rec(algorithm, algo_KNNii, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    all_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
            elif algorithm == 'ALS':
                all_recs.append(rec(algorithm, None, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    all_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
            else:
                ValueError('Invalid algorithm')
        all_recs = pd.concat(all_recs, ignore_index=True)
        test_data = pd.concat(test_data, ignore_index=True)
    else:
        for train, test in cross_validation(data[['user', 'item']], n_splits=partitions):
            # Filter the train and test sets to only include the user IDs from the pruned dataset
            train_remain = train[train['user'].isin(remaining_user_ids)]
            test_remain = test[test['user'].isin(remaining_user_ids)]
            train_pruned = train[train['user'].isin(removed_user_ids)]
            test_pruned = test[test['user'].isin(removed_user_ids)]
            test_data.append(test)
            temp_recs = []  # Necessary to combine the recommendations of most popular and selected algorithm
            if algorithm == 'UserUser':
                temp_recs.append(rec(algorithm, algo_KNNuu, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    temp_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
                recs = pd.concat(temp_recs, ignore_index=True)
                all_recs.append(recs)
            elif algorithm == 'ItemItem':
                temp_recs.append(rec(algorithm, algo_KNNii, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    temp_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
                recs = pd.concat(temp_recs, ignore_index=True)
                all_recs.append(recs)
            elif algorithm == 'ALS':
                temp_recs.append(rec(algorithm, None, train_remain, test_remain))
                if len(removed_user_ids) > 0:
                    temp_recs.append(rec_most_popular(algorithm, train_pruned, test_pruned))
                recs = pd.concat(temp_recs, ignore_index=True)
                all_recs.append(recs)
            else:
                ValueError('Invalid algorithm')
    return all_recs, test_data


# Compute recommendations
def rec(name: str, algo: lenskit.algorithms, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fit the algorithm on the training data and compute recommendations
    :param name: Name of the used algorithm
    :param algo: The algorithm object itself that is used for the recommendations
    :param train: Training data
    :param test: Test data
    :return recs: A DataFrame containing the recommendations for each user
    """
    if name == 'ALS':
        # Cloning ALS algorithm causes it to become not reproducible
        # ALS misbehaves when fitted multiple times, so it must be initialized each time
        algo = als.ImplicitMF(105, rng_spec=42)  # Hyperparameter features >10 and <200 optimized
        algo = Recommender.adapt(algo)
        algo.fit(train, implicit=True)
        users = test.user.unique()
        recs = batch.recommend(algo, users, 10)
    else:
        algo_cl = util.clone(algo)
        algo_cl = Recommender.adapt(algo_cl)
        algo_cl.fit(train, implicit=True)
        users = test.user.unique()
        recs = batch.recommend(algo_cl, users, 10)
    recs['Algorithm'] = name
    return recs


def rec_most_popular(name: str, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute recommendations for the pruned users using the method 'most popular'.
    :param name: Name of the algorithm used for the recommendations of the remaining users
    :param train: Training data
    :param test: Test data
    :return recs: A DataFrame containing the recommendations for each user
    """
    pop = Popular()
    pop.fit(train)
    users = test.user.unique()
    recs = batch.recommend(pop, users, 10)
    recs['Algorithm'] = name
    return recs


def eval_recs(recommendations: pd.DataFrame, test_data: pd.DataFrame, folds: int = 1) -> pd.DataFrame:
    """
    Function to evaluate the recommendations on the test data. The metric used for the evaluation is NDCG@10.
    :param recommendations: A DataFrame or a list of DataFrames containing the recommendations for each user in the
                            test data
    :param test_data: The test data to evaluate the recommendations on
    :param folds: Number of folds for cross-validation
    :return results: A DataFrame containing the results of the evaluation
    """
    rec_la = topn.RecListAnalysis()
    rec_la.add_metric(topn.ndcg, k=10)
    if folds > 1:
        results_list = []
        for i in range(folds):
            temp_results = rec_la.compute(recommendations[i], test_data[i])
            temp_results = temp_results.reset_index()
            results_list.append(temp_results)
        all_results = pd.concat(results_list)
        # Group by user and compute the average NDCG
        avg_ndcg = all_results.groupby('user')['ndcg'].mean().reset_index()
        results = results_list[0].copy()
        results['ndcg'] = results['user'].map(avg_ndcg.set_index('user')['ndcg'])
    else:
        results = rec_la.compute(recommendations, test_data)
        results = results.reset_index()
    return results


def rec_analysis_pipeline(data: pd.DataFrame, partitions: int, algorithm: str, sep_strategy: str = 'none',
                          pruning_strategy: str = 'none', lw_bound: float = 1.0, up_bound: float = 1.0,
                          step_size: float = 0.1,
                          eval_data: pd.DataFrame = None, analysis: bool = False) -> list[DataFrame] | DataFrame:
    """
    Pipeline to prune the dataset, train the algorithm, compute the recommendations and return the results
    :param data: The dataset to train the recommendations on
    :param partitions: Number of partitions for training, 1 is hold out, >1 is cross-validation
    :param algorithm: Algorithm to be used for the recommendations, either 'UserUser', 'ItemItem' or 'ALS'
    :param sep_strategy: The desired separation strategy, either 'rating_based', 'performance_based' or 'none'
    :param pruning_strategy: The desired pruning strategy, either 'rating_based', 'performance_based' or 'none'
    :param lw_bound: Lower bound for the user group to be pruned
    :param up_bound: Upper bound for the user group to be pruned
    :param step_size: Determines the size of the user groups
    :param eval_data: Evaluation Data necessary for certain pruning strategies, like the results of the baseline
        recommendations or the user influence data
    :param analysis: Boolean value to determine if a user group analysis should be performed
    :return results: A DataFrame containing the results
    :return analysis_results: A list of DataFrames, each containing the results of one user group
    """
    if pruning_strategy == 'rating_based' or pruning_strategy == 'random_based' or pruning_strategy == 'none':
        eval_data = data
    all_recs, test_data = rec_execute(data, partitions, algorithm, pruning_strategy, lw_bound, up_bound, eval_data)
    results = eval_recs(all_recs, test_data, partitions)
    if analysis:
        analysis_results = rec_analysis(data, results, sep_strategy, step_size)
        return analysis_results
    return results


def main():
    ###########################
    # INFLUENCE BASED PRUNING #
    ###########################

    # For instructions how to use the influence based pruning, please refer to the README.md file

    #########################
    # Load influence matrix #
    #########################

    # ML-100K #
    # User-User KNN
    ml100k_knn_uu_user_influence_matrix = np.load(r'influence_matrices\ml100k_knn_uu_ndcg_delta.npy')
    # Item-Item KNN
    ml100k_knn_ii_user_influence_matrix = np.load(r'influence_matrices\ml100k_knn_ii_ndcg_delta.npy')
    # ALS
    ml100k_als_user_influence_matrix = np.load(r'influence_matrices\ml100k_als_ndcg_delta.npy')

    # ML-1M #
    # User-User KNN
    ml1m_knn_uu_user_influence_matrix = np.load(r'influence_matrices\ml1m_knn_uu_ndcg_delta.npy')
    # Item-Item KNN
    ml1m_knn_ii_user_influence_matrix = np.load(r'influence_matrices\ml1m_knn_ii_ndcg_delta.npy')
    # ALS
    ml1m_als_user_influence_matrix = np.load(r'influence_matrices\ml1m_als_ndcg_delta.npy')

    # Last.FM #
    # User-User KNN
    lastfm_knn_uu_user_influence_matrix = np.load(r'influence_matrices\lastfm_knn_uu_ndcg_delta.npy')
    # Item-Item KNN
    lastfm_knn_ii_user_influence_matrix = np.load(r'influence_matrices\lastfm_knn_ii_ndcg_delta.npy')
    # ALS
    lastfm_als_user_influence_matrix = np.load(r'influence_matrices\lastfm_als_ndcg_delta.npy')

    # Amazon Luxury and Beauty #
    # User-User KNN
    amazon_luxury_knn_uu_user_influence_matrix = np.load(r'influence_matrices\amazon_luxury_knn_uu_ndcg_delta.npy')
    # Item-Item KNN
    amazon_luxury_knn_ii_user_influence_matrix = np.load(r'influence_matrices\amazon_luxury_knn_ii_ndcg_delta.npy')
    # ALS
    amazon_luxury_als_user_influence_matrix = np.load(r'influence_matrices\amazon_luxury_als_ndcg_delta.npy')

    # Amazon Digital Music #
    # User-User KNN
    amazon_digital_music_knn_uu_user_influence_matrix = (
        np.load(r'influence_matrices\amazon_digital_music_knn_uu_ndcg_delta.npy'))
    # Item-Item KNN
    amazon_digital_music_knn_ii_user_influence_matrix = (
        np.load(r'influence_matrices\amazon_digital_music_knn_ii_ndcg_delta.npy'))
    # ALS
    amazon_digital_music_als_user_influence_matrix = (
        np.load(r'influence_matrices\amazon_digital_music_als_ndcg_delta.npy'))

    ##############################
    # Load user to index mapping #
    ##############################
    # Last.FM #
    with open(r'user_to_index_mapping\lastfm_knn_uu_user_to_index_mapping.json', 'r') as json_file:
        lastfm_user_to_index_map = json.load(json_file)
    # ML-1M #
    with open(r'user_to_index_mapping\ml1m_knn_uu_user_to_index_mapping.json', 'r') as json_file:
        ml1m_user_to_index_map = json.load(json_file)

    # Amazon Luxury and Beauty #
    # User-User KNN
    with open(r'user_to_index_mapping\amazon_luxury_knn_uu_user_to_index_mapping.json', 'r') as json_file:
        amazon_luxury_knn_uu_user_to_index_map = json.load(json_file)
    # Item-Item KNN
    with open(r'user_to_index_mapping\amazon_luxury_knn_ii_user_to_index_mapping.json', 'r') as json_file:
        amazon_luxury_knn_ii_user_to_index_map = json.load(json_file)
    # ALS
    with open(r'user_to_index_mapping\amazon_luxury_als_user_to_index_mapping.json', 'r') as json_file:
        amazon_luxury_als_user_to_index_map = json.load(json_file)

    # Amazon Digital Music #
    # User-User KNN
    with open(r'user_to_index_mapping\amazon_digital_music_knn_uu_user_to_index_mapping.json', 'r') as json_file:
        amazon_digital_music_knn_uu_user_to_index_map = json.load(json_file)
    # Item-Item KNN
    with open(r'user_to_index_mapping\amazon_digital_music_knn_ii_user_to_index_mapping.json', 'r') as json_file:
        amazon_digital_music_knn_ii_user_to_index_map = json.load(json_file)
    # ALS
    with open(r'user_to_index_mapping\amazon_digital_music_als_user_to_index_mapping.json', 'r') as json_file:
        amazon_digital_music_als_user_to_index_map = json.load(json_file)

    ###########################################
    # Change depending on the test parameters #
    ###########################################
    # Define dataset
    data = ml1m_data
    # Number of data partitions for cross-validation
    folds = 5
    # Define the algorithm
    algorithm = 'ItemItem'  # 'UserUser', 'ItemItem' or 'ALS'
    # Select data for influence pruning
    # Choose which user influence matrices to use, depends on the algorithm and dataset
    user_influence_matrix = ml1m_knn_ii_user_influence_matrix
    # Choose the corresponding user to index mapping (none for ML-100K and ML-1M)
    user_to_index_map = None
    # Define the pruning strategy
    pruning_strategy = 'influence_score_based'  # 'influence_mean_based', 'influence_cl_mean_based', 'influence_diff_based', 'influence_score_based'
    # Define the threshold for the pruning strategy
    threshold = -0.00266
    ###########################################
    ###########################################

    # Baseline #
    bl_result = rec_analysis_pipeline(data, 5, algorithm)

    num_users = len(data['user'].unique())  # Number of users in the dataset

    # Calculate user influence data #
    influence_analysis_df = influence_analysis(user_influence_matrix, data,
                                               bl_result, user_to_index_map)

    influence_pr_result = rec_analysis_pipeline(data, folds, algorithm, pruning_strategy=pruning_strategy,
                                                lw_bound=threshold, eval_data=influence_analysis_df)

    remaining_user_ids, removed_user_ids = prune(influence_analysis_df, pruning_strategy,
                                                 threshold, 1)
    print('Remaining Users: ' + str((len(remaining_user_ids) / num_users) * 100) + ' %')
    print('Removed Users: ' + str((len(removed_user_ids) / num_users) * 100) + ' %')
    print('Baseline Results:')
    print('NDCG@10: ' + str(bl_result['ndcg'].mean()))
    print('Std: ' + str(bl_result['ndcg'].std()))
    print('Influence Pruning Results:')
    print('NDCG@10: ' + str(influence_pr_result['ndcg'].mean()))
    print('Std: ' + str(influence_pr_result['ndcg'].std()))


if __name__ == '__main__':
    main()
