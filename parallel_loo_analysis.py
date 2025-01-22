# This file contains the same logic as LOO the recommender pipeline, but with an implementation of the
# leave-one-out analysis that allows for parallel execution.
import numpy as np
import pandas as pd
import json
import argparse
import lenskit
from lenskit.datasets import ML1M
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn, user_knn
from load_datasets import load_data_lastfm, load_data_amazon_digital_music, load_data_amazon_luxury_beauty
from rec_utility import n_core_pruning, cross_validation

# Set up algorithms
# ALS has to be set up in the pipeline to ensure reproducibility
algo_KNNuu = user_knn.UserUser(57, feedback='implicit')  # Hyperparameter <70 optimized
algo_KNNii = item_knn.ItemItem(62, feedback='implicit')  # Hyperparameter <70 optimized

# Load the Last.FM dataset
last_fm_data = load_data_lastfm()

# Load the MovieLens 1M dataset
ml1m = ML1M(r'Datasets\ml-1m')
ml_1m_data = ml1m.ratings


def leave_one_out(data: pd.DataFrame, partitions: int, algorithm: str, pruned_user: int | str) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs a reproducible data split and removes a single user from the dataset. The algorithm is then
    trained on the pruned dataset and the recommendations are returned with the test data.
    :param data: Dataset to be used for recommendation
    :param partitions: Number of dataset partitions, =1 is hold out, >1 is cross-validation
    :param algorithm: Algorithm to be used for recommendation
    :param pruned_user: User ID of the user that is removed from the dataset
    :return all_recs: All recommendations for the pruned dataset
    :return test_data: Pruned test data
    """
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(data[['user', 'item']], partitions, xf.SampleFrac(0.2, rng_spec=42),
                                          rng_spec=42):
        test = test[test['user'] != pruned_user]
        train = train[train['user'] != pruned_user]
        test_data.append(test)
        if algorithm == 'UserUser':
            all_recs.append(rec(algorithm, algo_KNNuu, train, test))
        elif algorithm == 'ItemItem':
            all_recs.append(rec(algorithm, algo_KNNii, train, test))
        elif algorithm == 'ALS':
            all_recs.append(rec(algorithm, None, train, test))
        else:
            ValueError('Invalid algorithm')
    all_recs = pd.concat(all_recs, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)
    return all_recs, test_data


def loo_pipeline(data: pd.DataFrame, algo: str, prts: int, lw_index: int, up_index: int, iteration: int,
                 analysis_name: str, save_memory: bool = True) -> None:
    """
    Function executes the leave-one-out pipeline for the given dataset, algorithm and number of partitions.
    First a baseline is calculated, then every user is pruned one by one and the algorithm is trained on each pruned
    dataset. The NDCG values are compared to the baseline to determine the impact of each user on every other user.
    The result is a vector for each individual user, containing the NDCG delta from baseline to the
    pruned dataset (pruned - baseline).
    Important:
    For delta value d, d>0  indicates a negative impact and d<0 indicates a positive impact of the user.
    The lw_index and up_index parameters can be used to specify a range of users to be analyzed.
    :param data: Dataset to be analyzed
    :param algo: Algorithm to be used
    :param prts: Number of partitions, 1: hold out, >1: cross-validation
    :param lw_index: Lower bound for the users, if lw_index=10 the analysis starts with the user at index 10
    :param up_index: Upper bound for the users, for up_index=20 the analysis ends with the user at index 19
    :param iteration: Iteration number for the analysis, for later identification
    :param analysis_name: Name of the file to save the results
    :param save_memory: Boolean value to determine if a more memory efficient implementation should be used. The
        disadvantage is that the iteration names MUST continuously ascend from 0 to n.
    :return: None
    """
    baseline_results = rec_analysis_pipeline(data, prts, algo)
    bl_user_ndcg = baseline_results[['user', 'ndcg']]
    bl_user_ndcg_srt = bl_user_ndcg.sort_values(by=['user'], ascending=True).reset_index().drop(columns=['index'])
    # All unique users in the dataset that received recommendations
    unique_users = sorted(baseline_results['user'].unique())
    user_number = len(unique_users)
    if save_memory:
        index_diff = up_index - lw_index
        ndcg_delta_array = np.zeros((index_diff, user_number))
    else:
        ndcg_delta_array = np.zeros((user_number, user_number))
    index = 0
    i_mem = 0
    user_to_index_mapping = [0 for _ in range(user_number)]
    for user in unique_users:
        if lw_index <= index < up_index:
            # Fit algorithm, compute the recommendations and evaluate the results on the pruned dataset
            all_recs, test_data = leave_one_out(data, prts, algo, user)
            loo_pruned_results = eval_recs(all_recs, test_data)
            loo_pruned_results_reset = loo_pruned_results.reset_index()
            loo_pruned_user_ndcg = loo_pruned_results_reset[['user', 'ndcg']]
            # Add pruned user back to the results to calculate the delta
            pruned_user = pd.DataFrame({'user': [user], 'ndcg': [0]})
            loo_pruned_user_ndcg = pd.concat([loo_pruned_user_ndcg, pruned_user], ignore_index=True)
            loo_pruned_user_ndcg_srt = loo_pruned_user_ndcg.sort_values(by=['user'], ascending=True).reset_index().drop(
                columns=['index'])
            user_ndcg_delta = loo_pruned_user_ndcg_srt['ndcg'] - bl_user_ndcg_srt['ndcg']
            if save_memory:
                ndcg_delta_array[i_mem] = user_ndcg_delta.values
                user_influence_mean = ndcg_delta_array[i_mem].mean()
                i_mem += 1
            else:
                ndcg_delta_array[index] = user_ndcg_delta.values
                user_influence_mean = ndcg_delta_array[index].mean()
            if isinstance(user, str):
                user_to_index_mapping[index] = user
            else:
                user_to_index_mapping[index] = int(user)
            print("User " + str(user) + ": " + str(user_influence_mean))
        index += 1
    np.save(f'iter_{iteration}_{analysis_name}_ndcg_delta.npy', ndcg_delta_array)
    with open(f'list_{iteration}_{analysis_name}_user_to_index_mapping.json', 'w') as json_file:
        json.dump(user_to_index_mapping, json_file)


def data_fusion(iterations: int, analysis_name: str, save_memory: bool = True) -> None:
    """
    Function to fuse the results of parallel leave-one-out analyses. Saves the result in one file.
    :param iterations: Number of iterations to be fused
    :param analysis_name: Name of the analysis
    :param save_memory: Boolean value to determine if a more memory efficient implementation should be used. The
        disadvantage is that the iteration names must be in the right order.
    :return: None
    """
    if save_memory:
        with open(f'list_0_{analysis_name}_user_to_index_mapping.json', 'r') as json_file:
            fused_user_mapping = json.load(json_file)
            fused_data = np.zeros((len(fused_user_mapping), len(fused_user_mapping)))
        index_fused_data = 0
        for i in range(iterations):
            file_name = f'iter_{i}_{analysis_name}_ndcg_delta.npy'
            iter_data = np.load(file_name)
            file_name_list = f'list_{i}_{analysis_name}_user_to_index_mapping.json'
            with open(file_name_list, 'r') as json_file:
                user_mapping = json.load(json_file)
            for j in range(len(iter_data)):
                if index_fused_data + j >= len(fused_data):
                    continue
                fused_data[index_fused_data + j] = iter_data[j]
            index_fused_data += len(iter_data)
            for n in range(len(user_mapping)):
                if fused_user_mapping[n] == 0 and user_mapping[n] != 0:
                    fused_user_mapping[n] = user_mapping[n]
    else:
        fused_data = np.load(f'iter_0_{analysis_name}_ndcg_delta.npy')
        with open(f'list_0_{analysis_name}_user_to_index_mapping.json', 'r') as json_file:
            fused_user_mapping = json.load(json_file)
        for i in range(iterations):
            file_name = f'iter_{i}_{analysis_name}_ndcg_delta.npy'
            iter_data = np.load(file_name)
            file_name_list = f'list_{i}_{analysis_name}_user_to_index_mapping.json'
            with open(file_name_list, 'r') as json_file:
                user_mapping = json.load(json_file)
            for n in range(len(fused_data)):
                if fused_data[n].sum() == 0 and iter_data[n].sum() != 0:
                    fused_data[n] = iter_data[n]
                if fused_user_mapping[n] == 0 and user_mapping[n] != 0:
                    fused_user_mapping[n] = user_mapping[n]
    np.save(f'{analysis_name}_ndcg_delta.npy', fused_data)
    with open(f'{analysis_name}_user_to_index_mapping.json', 'w') as json_file:
        json.dump(fused_user_mapping, json_file)


def rec_execute(data: pd.DataFrame, partitions: int, algorithm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function receives a dataset and partitions it into training and test data, to perform recommendations
    :param data: The dataset to be partitioned
    :param partitions: The number of partitions
    :param algorithm: The algorithm to be used for the recommendations as a string,
                    either 'UserUser', 'ItemItem' or 'ALS'
        recommendations or the user influence data
    :return all_recs: A DataFrame or list of DataFrames containing the recommendations for each user
    :return test_data: A DataFrame or list of DataFrames containing the test data
    """
    all_recs = []
    test_data = []

    if partitions == 1:
        # Hold out
        for train, test in xf.partition_users(data[['user', 'item']], partitions, xf.SampleFrac(0.2, rng_spec=42),
                                              rng_spec=42):
            test_data.append(test)
            if algorithm == 'UserUser':
                all_recs.append(rec(algorithm, algo_KNNuu, train, test))
            elif algorithm == 'ItemItem':
                all_recs.append(rec(algorithm, algo_KNNii, train, test))
            elif algorithm == 'ALS':
                all_recs.append(rec(algorithm, None, train, test))
            else:
                ValueError('Invalid algorithm')
        all_recs = pd.concat(all_recs, ignore_index=True)
        test_data = pd.concat(test_data, ignore_index=True)
    else:
        for train, test in cross_validation(data[['user', 'item']], n_splits=partitions):
            # Filter the train and test sets to only include the user IDs from the pruned dataset
            test_data.append(test)
            temp_recs = []  # Necessary to combine the recommendations of most popular and selected algorithm
            if algorithm == 'UserUser':
                temp_recs.append(rec(algorithm, algo_KNNuu, train, test))
                recs = pd.concat(temp_recs, ignore_index=True)
                all_recs.append(recs)
            elif algorithm == 'ItemItem':
                temp_recs.append(rec(algorithm, algo_KNNii, train, test))
                recs = pd.concat(temp_recs, ignore_index=True)
                all_recs.append(recs)
            elif algorithm == 'ALS':
                temp_recs.append(rec(algorithm, None, train, test))
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
            results_list.append(rec_la.compute(recommendations[i], test_data[i]))
        all_results = pd.concat(results_list)
        # The results DataFrame must retain its index structure for future use in the pipeline
        results = results_list[0].copy()
        results['ndcg'] = all_results.groupby('user')['ndcg'].mean().values
    else:
        results = rec_la.compute(recommendations, test_data)
    return results


def rec_analysis_pipeline(data: pd.DataFrame, partitions: int, algorithm: str) -> pd.DataFrame:
    """
    Pipeline to train the algorithm, compute the recommendations and return the results
    :param data: The dataset to train the recommendations on
    :param partitions: Number of partitions for training, 1 is hold out, >1 is cross-validation
    :param algorithm: Algorithm to be used for the recommendations, either 'UserUser', 'ItemItem' or 'ALS'
    :return results: A DataFrame containing the results
    """
    all_recs, test_data = rec_execute(data, partitions, algorithm)
    results = eval_recs(all_recs, test_data, partitions)
    results = results.reset_index()
    return results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the leave-one-out pipeline with specified parameters.')
    parser.add_argument('--algo', type=str, required=False, default='UserUser', help='Algorithm to be used '
                                                                                     'for the analysis')
    parser.add_argument('--lw', type=int, required=False, default=0, help='Lower bound for the user number')
    parser.add_argument('--up', type=int, required=False, default=1, help='Upper bound for the user number')
    parser.add_argument('--iter', type=int, required=True, help='Iteration of the analysis or the '
                                                                'number of iterations if --fusion is set')
    parser.add_argument('--name', type=str, required=True, help='Name of the analysis')
    parser.add_argument('--dataset', type=str, required=False, help='Dataset to perform the LOO analysis '
                                                                    'on, either "lastfm" (default) or "ml1m"')
    parser.add_argument('--fusion', type=bool, required=False, help='Data fusion after analysis')
    parser.add_argument('--save_mem', type=bool, required=False, default=True, help='Save memory during '
                                                                                    'the analysis, but requires the '
                                                                                    'iteration numbers to be in '
                                                                                    'strictly ascending order')
    # Parse arguments
    args = parser.parse_args()

    if args.fusion:
        # Data fusion
        data_fusion(args.iter, args.name)
    else:
        # Execute the pipeline
        if args.dataset == 'ml1m':
            # MovieLens 1M
            loo_pipeline(ml_1m_data, args.algo, 1, args.lw, args.up, args.iter, args.name)
        elif args.dataset == 'amazon_luxury':
            # Amazon Luxury Beauty #
            amazon_luxury_beauty_data = n_core_pruning(load_data_amazon_luxury_beauty())
            loo_pipeline(amazon_luxury_beauty_data, args.algo, 1, args.lw, args.up, args.iter, args.name)
        elif args.dataset == 'amazon_digital_music':
            # Amazon Digital Music #
            amazon_digital_music_data = n_core_pruning(load_data_amazon_digital_music())
            loo_pipeline(amazon_digital_music_data, args.algo, 1, args.lw, args.up, args.iter, args.name)
        else:
            # Last.FM #
            core_pruned_last_fm_data = n_core_pruning(last_fm_data)
            loo_pipeline(core_pruned_last_fm_data, args.algo, 1, args.lw, args.up, args.iter, args.name)


if __name__ == '__main__':
    main()
