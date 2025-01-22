# Loads datasets for the analysis

import pandas as pd
import gzip
import json


def parse(path: str):
    """
    Parse the Gzipped JSON file
    :param path:
    :return:
    """
    g = gzip.open(path, 'rb')
    for line in g:
        yield json.loads(line)


def get_df(path: str) -> pd.DataFrame:
    """
    Get the DataFrame from the Gzipped JSON file
    :param path:
    :return: A DataFrame containing the user-item interactions
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def load_data_lastfm() -> pd.DataFrame:
    """
    Load the Last.FM dataset containing user-artist interactions
    :return interactions: A DataFrame containing the user-artist interactions
    """
    # Load the user_artists.dat file
    file_path = r'Datasets/user_artists.dat'
    column_names = ['userID', 'artistID', 'weight']
    interactions = pd.read_csv(file_path, sep='\t', names=column_names, header=0)

    # Rename columns to match the dataset format
    interactions.rename(columns={'userID': 'user', 'artistID': 'item', 'weight': 'rating'}, inplace=True)

    return interactions


def load_data_amazon_pantry() -> pd.DataFrame:
    """
    Load the Amazon Pantry dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Prime_Pantry_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


def load_data_amazon_musical_instruments() -> pd.DataFrame:
    """
    Load the Amazon Musical Instruments dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Musical_Instruments_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


def load_data_amazon_digital_music() -> pd.DataFrame:
    """
    Load the Amazon Digital Music dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Digital_Music_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


def load_data_amazon_software() -> pd.DataFrame:
    """
    Load the Amazon Software dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Software_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


def load_data_amazon_luxury_beauty() -> pd.DataFrame:
    """
    Load the Amazon Luxury Beauty dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Luxury_Beauty_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


def load_data_amazon_industrial_scientific() -> pd.DataFrame:
    """
    Load the Amazon Industrial & Scientific dataset containing user-item interactions
    :return interactions: A DataFrame containing the user-item interactions
    """
    # Load the reviews_Video_Games.json.gz file
    file_path = r'Datasets/Industrial_and_Scientific_5.json.gz'
    interactions = get_df(file_path)

    # Rename columns to match the dataset format
    interactions.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating',
                                 'unixReviewTime': 'timestamp'}, inplace=True)
    return interactions


