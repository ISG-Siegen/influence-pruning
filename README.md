This repository contains the code for the paper "Removing Bad Influence: Identifying and Pruning Detrimental Users in
Collaborative Filtering Recommender Systems".

We use Python 3.11.9. The required packages can be installed with pip from requirements.txt.
  
The following sections give a quick overview how the analysis can be performed.

The paths for the datasets have to be changed in the code itself. The datasets can be downloaded from the following links:
- LastFM: https://grouplens.org/datasets/hetrec-2011/
- MovieLens 100k: https://grouplens.org/datasets/movielens/100k/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- Amazon: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html

We perform 5-core-pruning on each dataset.

# Influence-based Pruning for Collaborative Filtering
The pruning pipeline has to be set up by changing the code according to the desired dataset and algorithm.
Put all datasets into a Folder called "Datasets". Alternatively, change the corresponding file paths in the pruning_pipeline.py and load_datasets.py files.
All other changes in the code have to be made between 

    ###########################################
    # Change depending on the test parameters #
    ###########################################
    
and

    ###########################################
    ###########################################

The comments in our code further specify how to select a dataset and algorithm.
The thresholds we used for the pruning tests are in the Folder "influence_thresholds".
They are specific for each dataset and algorithm.

The appropriate user_influence_matrix has to be selected from the section "Load influence matrix" and the corresponding  
user_to_index_mapping from "Load user to index mapping". 
!Important!
The user_influence_matrix has to be calculated beforehand using
the Leave-one-out analysis. The user_to_index_mapping is the mapping from the user id to the index in the user_influence_matrix.

Put the influence matrices in a folder called "influence_matrices" and the user_to_index_mapping in a folder called "user_to_index_mapping"
or change the path in the code.


# Leave-one-out analysis
Execution of parallel_loo_analysis.py:

    python parallel_loo_analysis.py --algo ALGO --lw LW --up UP --iter ITER --name NAME --dataset DATASET --fusion FUSION --save_mem SAVE_MEM

- Parameters:

    --algo: The algorithm to be used for the analysis. Possible values are UserUser, ItemItem, or ALS. This parameter is optional, default is UserUser.
  
    --lw: The lower bound for the user number. This parameter is optional, default is 0.
  
    --up: The upper bound for the user number. This parameter is optional, default is 1.
        !Important! 
        Set lw (next) = up (previous) for the next iteration to not miss any users.
        
    --iter: The iteration of the analysis or the number of iterations if --fusion is set True. This parameter is required.
        !Important!
        The naming for n iterations must be continuous from 0 to n.
        
    --name: The name of the analysis. This parameter is required.
    
    --dataset: The dataset to perform the LOO analysis on. Possible values are lastfm (internal default) or ml1m. This parameter is optional.
    
    --fusion: A boolean flag indicating whether to perform data fusion after analysis. This parameter is optional. Default is False.
    
    --save_mem: A boolean flag determining wheter LOO should be done in a memory efficient way. This additionally requires the iterations to be in order. Default is True.

Example:

    python parallel_loo_analysis.py --algo UserUser --lw 5 --up 10 --iter 1 --name ml1m_hl_knn_uu --dataset ml1m

- In this example prunes the 5th to 9th user of the ml1m dataset using User-User kNN. For a complete analysis without 
  parallelization select lw = 0 and up = number of users in the dataset.

If the analysis is split in multiple jobs (for efficiency), each job creates its own npy file containing the results for the analyzed users.
The results can be merged by setting --fusion=True, --iter equal to the number of files to be merged and --name equal to the name of the analysis.

# Threshold search
Execution of parallel_threshold_search.py:

    python parallel_threshold_search.py --dataset DATASET --algo ALGO --lw LW --up UP --file_num FILE_NUM --folds FOLDS --strategy STRATEGY --name NAME --n_thr N_THR --fusion FUSION
    
- Parameters
- 
    --dataset: The dataset to be used for the analysis. Possible values are lastfm, ml100k, ml1m, amazon_luxury or amazon_digital_music. This parameter is optional.
  
    --algo: The algorithm to be used for the recommendations. Possible values are UserUser, ItemItem, or ALS. This parameter is optional, with a default value of UserUser.
  
    --lw: The lower bound for the threshold search. This parameter is optional, with a default value of 0.
  
    --up: The upper bound for the threshold search. This parameter is optional, with a default value of 0.
  
    --file_num: The ID of the resulting file for later identification. If --fusion=True, this indicates the number of files. This parameter is required.
  
    --folds: The number of folds for cross-validation. This parameter is optional, with a default value of 5.
  
    --strategy: The employed pruning strategy. This parameter is required and must be specified. Possible pruning strategies are 'influence_mean_based', influence_cl_mean_based',
                'influence_diff_based' or 'influence_score_based'
  
    --name: The name of the dataset/algorithm combination. This parameter is required.
  
    --n_thr: The number of thresholds to be tested during the search. This parameter is optional, with a default value of 10.
  
    --fusion: A boolean flag indicating whether to execute the data fusion function after the threshold search. This parameter is optional, with a default value of False.

Example:

    python parallel_threshold_search.py --dataset ml100k --algo ItemItem --lw -0.05 --up 0 --file_num 1 --folds 5 --strategy influence_score_based --name ml100k_knn_ii --n_thr 20

- In this example, the influence score based threshold search is performed on the ml100k dataset using Item-Item kNN and cross-validation with 5 folds.
  20 random thresholds between -0.05 and 0 are tested and saved in a .csv file. The ranges we use are provided in the "influence_threshold" folder.

For efficiency, the threshold search can be split into multiple jobs. Each job creates its own .csv file containing the results for the analyzed thresholds.
The results can be merged for a specific dataset, algorithm and strategy by setting --fusion=True, --file_num equal to the number of files to be merged,
strategy equal to the employed pruning strategy and --name equal to the name of the dataset/algorithm combination.
