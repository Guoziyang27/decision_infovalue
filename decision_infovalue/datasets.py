"""
Dataset loading and processing utilities for the Info Value Toolkit.
"""
from typing import Tuple, Dict, Any, Final
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
from urllib.request import urlretrieve
import numpy as np

github_data_url: Final[str] = "https://github.com/Guoziyang27/decision_infovalue/raw/main/data/"

def load_housing_data(with_human_data: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the housing price dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    df = pd.read_csv(cache(github_data_url + "AmesHousing.csv"))

    if with_human_data:
        human_df = pd.read_csv(cache(github_data_url + "house_price_human.csv"))
        df = pd.merge(df, human_df, left_on="Order", right_on="test_order", how="left")

    metadata = {
        "name": "Ames Iowa Housing Prices",
        "source": "Kaggle",
        "n_samples": len(df),
        "n_features": df.shape[1],
        "feature_names": list(df.columns),
        "target_name": "SalePrice",
        "description": "Ames Iowa Housing Prices dataset with various features"
    }
    
    return df, metadata


def load_recidivism_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the recidivism dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    
    # Load the data
    human_df = pd.read_csv(cache(github_data_url+ "compas-scores-two-years_human_response.csv"))
    human_df.drop(human_df[human_df['individual_id'].apply(lambda x: not (str(x).isdigit()))].index, inplace=True)
    human_df["individual_id"] = human_df["individual_id"].astype("int64")
    
    data_df = pd.read_csv(cache(github_data_url + "compas-scores-two-years.csv"))
    
    data_df = pd.merge(data_df, human_df[["individual_id", "predicted_decision"]], left_on='id', right_on='individual_id', how='inner')
    
    # Create metadata
    metadata = {
        "name": "Recidivism Risk Assessment",
        "source": "Stanford Policy Lab",
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "feature_names": list(data_df.columns),
        "target_name": "two_year_recid",
        "description": "Recidivism risk assessment dataset with user responses"
    }
    
    return data_df, metadata


def load_cxr_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the CXR dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    
    # Load the data
    data_df = pd.read_csv(cache(github_data_url+ "CXRForecasting.csv"))
    
    # Create metadata
    metadata = {
        "name": "CXR Vision Model and Human Diagnosis",
        "source": "MIMIC-CXR",
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "feature_names": list(data_df.columns),
        "target_name": "abnormal_stratified",
        "description": "CXR Vision Model and Human Diagnosis dataset"
    }
    
    return data_df, metadata


def load_deepfake_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the deepfake dataset.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    
    # Load the data
    data_df = pd.read_csv(cache(github_data_url+ "deepfake_data.csv"))
    video_feature_df = pd.read_csv(cache(github_data_url+ "deepfake_video_features.csv"))

    data_df['guess_int'] = data_df['guess_int'] / 100
    data_df['human_guess'] = data_df['guess_int']
    
    # Shift human_ai_guess by 1 row (equivalent to tail(-1) in R)
    data_df['human_ai_guess'] = data_df['guess_int'].shift(-1)
    
    # Filter for first round guesses
    data_df = data_df[data_df['guess_round'] == 1]
    
    # Calculate ai_guess based on fake and c_score
    data_df['ai_guess'] = np.where(~data_df['fake'], 1 - data_df['c_score'], data_df['c_score'])
    
    # Select relevant columns
    data_df = data_df[['video', 'fake', 'human_guess', 'human_ai_guess', 'ai_guess']]
    # Join with video features, replacing NaN with 0
    data_df = data_df.merge(video_feature_df.fillna(0), how='left')
    data_df = data_df.drop(columns=['connection', 'clear example of deepfake that people have trouble with'])
    
    # Create metadata
    metadata = {
        "name": "Deepfake Detection",
        "source": "Deepfake Detection Dataset",
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "feature_names": list(data_df.columns),
        "target_name": "fake",
        "description": "Deepfake Detection dataset"
    }
    
    return data_df, metadata


def load_haiid_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and process the Human-AI Interactions Dataset by Vodrahalli et al. 2022.
    
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
    """
    
    # Load the data
    data_df = pd.read_csv(cache(github_data_url+ "haiid_dataset.csv"))

    data_df = data_df.rename(columns={'advice': 'ai_pred', 'response_1': 'h_pred', 'response_2': 'h_ai_pred'})
    data_df = data_df[['task_name', 'ai_pred', 'h_pred', 'h_ai_pred']]
    data_df[['ai_pred', 'h_pred', 'h_ai_pred']] = (data_df[['ai_pred', 'h_pred', 'h_ai_pred']] + 1) / 2
    data_df['gt'] = np.random.randint(0, 2, len(data_df))
    data_df = data_df.assign(ai_pred = lambda x: np.where(x['gt'] == 0, 1 - x['ai_pred'], x['ai_pred']),
                             h_pred = lambda x: np.where(x['gt'] == 0, 1 - x['h_pred'], x['h_pred']),
                             h_ai_pred = lambda x: np.where(x['gt'] == 0, 1 - x['h_ai_pred'], x['h_ai_pred']))
    
    # Create metadata
    metadata = {
        "name": "Human-AI Interactions Dataset",
        "source": "Vodrahalli et al. 2022",
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "feature_names": list(data_df.columns),
        "target_name": "gt",
        "description": "Human-AI Interactions dataset"
    }
    
    return data_df, metadata

def get_dataset(name: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset ('housing' or 'recidivism' or 'cxr' or 'deepfake' or 'haiid')
        
    Returns:
        Tuple containing:
        - DataFrame with features and target
        - Dictionary with dataset metadata
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if name.lower() == 'housing':
        return load_housing_data(**kwargs)
    elif name.lower() == 'recidivism':
        return load_recidivism_data(**kwargs)
    elif name.lower() == 'cxr':
        return load_cxr_data(**kwargs)
    elif name.lower() == 'deepfake':
        return load_deepfake_data(**kwargs)
    elif name.lower() == 'haiid':
        return load_haiid_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: 'housing', 'recidivism', 'cxr', 'deepfake', 'haiid'") 
    
def cache(url: str, file_name: str | None = None) -> str:
    """Loads a file from the URL and caches it locally."""
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(__file__), "cached_data")
    os.makedirs(data_dir, exist_ok=True)

    file_path: str = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path