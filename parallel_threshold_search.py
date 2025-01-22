# Implements parallel threshold search for influence threshold

import numpy as np
import pandas as pd
from pandas import DataFrame
import argparse
import json
import lenskit
from lenskit.datasets import ML100K, ML1M
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn, user_knn
from lenskit.algorithms.basic import Popular
from load_datasets import load_data_lastfm, load_data_amazon_digital_music, load_data_amazon_luxury_beauty
from data_analysis import influence_analysis
from data_pruning import prune
from rec_utility import cross_validation, n_core_pruning

# Load ML100K dataset
ml100k = ML100K(r'Datasets\ml-100k')
ml_100k_data = ml100k.ratings

# Load ML1M dataset
ml1m = ML1M(r'Datasets\ml-1m')
ml_1m_data = ml1m.ratings

# Set up algorithms
# ALS has to be set up in the pipeline to ensure reproducibility
algo_KNNuu = user_knn.UserUser(57, feedback='implicit')
algo_KNNii = item_knn.ItemItem(62, feedback='implicit')


def rec_analysis_pipeline(data: pd.DataFrame, partitions: int, algorithm: str, pruning_strategy: str = 'none',
                          lw_bound: float = 1.0, up_bound: float = 1.0, eval_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Pipeline to prune the dataset, train the algorithm, compute the recommendations and return the results
    :param data: The dataset to train the recommendations on
    :param partitions: Number of partitions for training, 1 is hold out, >1 is cross-validation
    :param algorithm: Algorithm to be used for the recommendations, either 'UserUser', 'ItemItem' or 'ALS'
    :param pruning_strategy: The desired pruning strategy, either 'rating_based', 'performance_based' or 'none'
    :param lw_bound: Lower bound for the user group to be pruned
    :param up_bound: Upper bound for the user group to be pruned
    :param eval_data: Evaluation Data necessary for certain pruning strategies, like the results of the baseline
        recommendations or the user influence data
    :return results: A DataFrame containing the results
    :return analysis_results: A list of DataFrames, each containing the results of one user group
    """
    if pruning_strategy == 'rating_based' or pruning_strategy == 'random_based' or pruning_strategy == 'none':
        eval_data = data
    all_recs, test_data = rec_execute(data, partitions, algorithm, pruning_strategy, lw_bound, up_bound, eval_data)
    results = eval_recs(all_recs, test_data, partitions)
    return results


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

    if partitions == 1:
        # Hold out
        for train, test in xf.partition_users(data[['user', 'item']], partitions, xf.SampleFrac(0.2, rng_spec=42),
                                              rng_spec=42):
            # Filter the train and test sets to only include the user IDs from the pruned dataset
            train_remain = train[train['user'].isin(remaining_user_ids)]
            test_remain = test[test['user'].isin(remaining_user_ids)]
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
        algo = als.ImplicitMF(105, rng_spec=42)
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


def influence_threshold_search(dataset: pd.DataFrame, algo: str, folds: int, strategy: str, name: str, num_thr: int,
                               file_id: int, thr_lw_bound: float = 0, thr_up_bound: float = 0):
    """
    Function to perform a random search for the best threshold for a given influence based pruning strategy.
    :param dataset: The dataset to be used
    :param algo: The algorithm to be used
    :param folds: The number of folds for cross-validation
    :param strategy: The employed pruning strategy
    :param name: The name of the dataset/algorithm combination (e.g. 'lastfm_hl_knn_uu')
    :param num_thr: The number of thresholds to be tested
    :param file_id: The ID of the threshold search, necessary for future reference. Must start at 0 and
            continuously ascend.
    :param thr_up_bound: The upper bound for the threshold search
    :param thr_lw_bound: The lower bound for the threshold search
    :return: None
    """
    bl_result = rec_analysis_pipeline(dataset, folds, algo)
    print("Algorithm: " + algo)
    print("Strategy: " + strategy)
    print("_______________________________________________________")
    print("Baseline NDCG Mean:")
    print(bl_result['ndcg'].mean())
    print("_______________________________________________________")
    influence_matrix = np.load(f'{name}_ndcg_delta.npy')
    if dataset is ml_100k_data or dataset is ml_1m_data:
        influence_analysis_df = influence_analysis(influence_matrix, dataset, bl_result, None)
    else:
        with open(f'{name}_user_to_index_mapping.json', 'r') as json_file:
            user_to_index_map = json.load(json_file)
        influence_analysis_df = influence_analysis(influence_matrix, dataset, bl_result, user_to_index_map)
    all_results = []
    tested_thresholds = []
    for i in range(num_thr):
        threshold = np.round(np.random.uniform(thr_lw_bound, thr_up_bound), 7)
        safeguard = 0
        while threshold in tested_thresholds:
            threshold = np.round(np.random.uniform(thr_lw_bound, thr_up_bound), 7)
            safeguard += 1
            if safeguard > 100:
                print("Did not find a threshold.")
                break
        tested_thresholds.append(threshold)
        influence_pr_result = rec_analysis_pipeline(dataset, folds, algo, pruning_strategy=strategy,
                                                    lw_bound=threshold, eval_data=influence_analysis_df)
        all_results.append([threshold, influence_pr_result['ndcg'].mean(), influence_pr_result['ndcg'].std()])
        print("NDCG Mean:")
        print(influence_pr_result['ndcg'].mean())
        print("Threshold:")
        print(threshold)
        print("_______________________________________________________")
    all_results = pd.DataFrame(all_results, columns=['Threshold', 'NDCG Mean', 'NDCG Standard Deviation'])
    all_results.to_csv(f'{name}_{file_id}_{strategy}_threshold_search.csv', index=False)


def search_data_fusion(name: list[str], strategy: list[str], search_num: int):
    """
    Function to combine a list of .csv files containing the results of threshold searches for different methods.
    :param name: Name of the dataset/algorithm combinations
    :param strategy: Name of the pruning strategy
    :param search_num: Number of search files to be combined
    :return: None
    """
    all_results = []
    for i in range(search_num):
        try:
            results = pd.read_csv(f'{name}_{i}_{strategy}_threshold_search.csv')
            all_results.append(results)
        except FileNotFoundError:
            print(f'File {name}_{i}_{strategy}_threshold_search.csv not found. Skipping.')
            continue
    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(f'{name}_{strategy}_threshold_search.csv', index=False)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enter Parameters to randomly search possible influence threshold.')
    parser.add_argument('--dataset', type=str, required=False, help='The dataset to be used, either '
                                                                    'lastfm, ml100k, ml1m, amazon_luxury or '
                                                                    'amazon_digital_music')
    parser.add_argument('--algo', type=str, required=False, default='UserUser', help='The algorithm to be used')
    parser.add_argument('--lw', type=float, required=False, default=0, help='The lower bound for the threshold search')
    parser.add_argument('--up', type=float, required=False, default=0, help='The upper bound for the threshold search')
    parser.add_argument('--file_num', type=int, required=True, help='The ID of the resulting file for later '
                                                                    'identification. If fusion=True then the number '
                                                                    'of files')
    parser.add_argument('--folds', type=int, required=False, default=5, help='The number of folds for cross-validation')
    parser.add_argument('--strategy', type=str, required=True, help='The employed pruning strategy')
    parser.add_argument('--name', type=str, required=True, help='The name of the dataset/algorithm combination')
    parser.add_argument('--n_thr', type=int, required=False, default=10, help='The number of thresholds to be tested')
    parser.add_argument('--fusion', type=bool, required=False, default=False, help='Execute the datafusion function')
    args = parser.parse_args()

    if args.fusion:
        # Fuse results of threshold search
        search_data_fusion(args.name, args.strategy, args.file_num)
    else:
        # Execute threshold search
        if args.dataset == 'lastfm':
            # Load LastFM dataset
            dataset = n_core_pruning(n_core_pruning(load_data_lastfm()))
        elif args.dataset == 'ml100k':
            # Load ML100K dataset
            dataset = ml_100k_data
        elif args.dataset == 'ml1m':
            # Load ML1M dataset
            dataset = ml_1m_data
        elif args.dataset == 'amazon_luxury':
            # Load Amazon Luxury dataset
            dataset = n_core_pruning(n_core_pruning(load_data_amazon_luxury_beauty()))
        elif args.dataset == 'amazon_digital_music':
            # Load Amazon Digital Music dataset
            dataset = n_core_pruning(n_core_pruning(load_data_amazon_digital_music()))
        else:
            ValueError('Invalid dataset')
        influence_threshold_search(dataset, args.algo, args.folds, args.strategy, args.name, args.n_thr, args.file_num,
                                   args.lw, args.up)


if __name__ == '__main__':
    main()
