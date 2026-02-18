import numpy as np
import scipy.stats as st
import pdb
from scipy import stats
from decision_infovalue.scoring_rules._scoring_rule import _mse_score, _brier_score
import pandas as pd

def _rational_decision(signals, signal_values, use_data, eval_data, gt, scoring_rule, ret_std: bool = False, agg_func: str = "mean", ret_orig_value: bool = False):
    if agg_func == "mean":
        # prior_action = np.mean(use_data[gt])
        agg_func = np.mean
    elif agg_func == "median":
        # prior_action = np.median(use_data[gt])
        agg_func = np.median
    elif agg_func == "mode":
        # prior_action = st.mode(use_data[gt])[0]
        agg_func = lambda x: pd.Series.mode(x)[0]
    elif callable(agg_func):
        # prior_action = agg_func(use_data[gt])
        agg_func = agg_func
    else:
        raise ValueError(f"Invalid aggregation function: {agg_func}, must be 'mean' or 'median' or 'mode' or a callable function")
    prior_action = agg_func(use_data[gt])

    grouped = use_data[signals + [gt]].groupby(signals, dropna=False).agg(calibrated_action=(gt, agg_func)).reset_index()
    if isinstance(signal_values, list) or (isinstance(signal_values, np.ndarray) and signal_values.ndim == 1):
        if isinstance(signal_values, np.ndarray):
            signal_values = signal_values.tolist()
        decision = grouped.loc[grouped[signals].apply(lambda x: tuple(x) == tuple(signal_values), axis=1)]['calibrated_action'].values[0]
    elif isinstance(signal_values, np.ndarray) and signal_values.ndim == 2:
        signal_value_df = pd.DataFrame(signal_values, columns=signals)
        decision = signal_value_df.merge(grouped, on=signals, how="left")['calibrated_action'].values
    else:
        raise ValueError(f"Invalid signal values: {signal_values}, must be a list or a numpy array")
    return decision

def _calculate_rational_payoff(signals, use_data, eval_data, gt, scoring_rule, ret_std: bool = False, agg_func: str = "mean", ret_orig_value: bool = False):
    if agg_func == "mean":
        # prior_action = np.mean(use_data[gt])
        agg_func = lambda x: np.mean(x)
    elif agg_func == "median":
        # prior_action = np.median(use_data[gt])
        agg_func = lambda x: np.median(x)
    elif agg_func == "mode":
        # prior_action = st.mode(use_data[gt])[0]
        agg_func = lambda x: pd.Series.mode(x)[0]
    elif callable(agg_func):
        # prior_action = agg_func(use_data[gt])
        agg_func = agg_func
    else:
        raise ValueError(f"Invalid aggregation function: {agg_func}, must be 'mean' or 'median' or 'mode' or a callable function")
    prior_action = agg_func(use_data[gt])

    if signals == []:
        score = eval_data[[gt]].copy()
        score['calibrated_action'] = prior_action
    else:
        grouped = use_data[signals + [gt]].groupby(signals, dropna=False).agg(calibrated_action=(gt, agg_func)).reset_index()
        score = eval_data[signals + [gt]].merge(grouped, on=signals, how="left").copy()
        score['calibrated_action'] = score['calibrated_action'].fillna(prior_action)
    score['payoff'] = scoring_rule(score['calibrated_action'].to_numpy(), score[gt].to_numpy())
    if ret_orig_value:
        return score['payoff'].to_numpy()
    if ret_std:
        return np.mean(score['payoff']), stats.sem(score['payoff'])
    return np.mean(score['payoff'])

def _linear_constraint_rational_decision(signals, signal_values, use_data, eval_data, gt, scoring_rule, ret_std: bool = False, agg_func: str = "mean", ret_orig_value: bool = False):
    pass

def _linear_constraint_rational_payoff(signals, use_data, eval_data, gt, scoring_rule, ret_std: bool = False, agg_func: str = "mean", ret_orig_value: bool = False):
    # If no signals, return prior action like in _calculate_rational_payoff
    if agg_func == "mean":
        prior_action = np.mean(use_data[gt])
        # agg_func = np.mean
    elif agg_func == "median":
        prior_action = np.median(use_data[gt])
        # agg_func = np.median
    elif agg_func == "mode":
        prior_action = st.mode(use_data[gt])[0]
        # agg_func = lambda x: st.mode(x)[0]
    elif callable(agg_func):
        prior_action = agg_func(use_data[gt])
    else:
        raise ValueError(f"Invalid aggregation function: {agg_func}, must be 'mean' or 'median' or 'mode' or a callable function")
    if signals == []:
        score = eval_data[[gt]].copy()
        score['calibrated_action'] = prior_action
        score['payoff'] = scoring_rule(score['calibrated_action'].to_numpy(), score[gt].to_numpy())
        if ret_orig_value:
            return score['payoff']
        if ret_std:
            return np.mean(score['payoff']), stats.sem(score['payoff'])
        return np.mean(score['payoff'])

    # Fit linear model using signals
    X = use_data[signals].copy()
    y = use_data[gt]
    
    # Split signals into numeric and categorical
    numeric_signals = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_signals = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    # Create dummy variables for categorical signals
    if categorical_signals:
        X = pd.get_dummies(X, columns=categorical_signals, drop_first=True).astype(float)
        
    # Add constant term for intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # Fit linear regression using normal equation
    beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
    
    # Make predictions on eval data
    X_eval = eval_data[signals]
    if categorical_signals:
        X_eval = pd.get_dummies(X_eval, columns=categorical_signals, drop_first=True).astype(float)
    X_eval_with_const = np.column_stack([np.ones(len(X_eval)), X_eval])
    predictions = X_eval_with_const @ beta
    
    # Clip predictions to [0,1] since we're predicting probabilities
    predictions = np.clip(predictions, 0, 1) if scoring_rule != _mse_score else predictions
    
    # Calculate payoff
    score = eval_data[[gt]].copy()
    score['calibrated_action'] = predictions
    score['payoff'] = scoring_rule(score['calibrated_action'].to_numpy(), score[gt].to_numpy())
    
    if ret_orig_value:
        return score['payoff']
    if ret_std:
        return np.mean(score['payoff']), stats.sem(score['payoff'])
    return np.mean(score['payoff'])