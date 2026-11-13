#!/usr/bin/env python3
"""
Evaluation metrics.
"""

import numpy as np
from sklearn.metrics import r2_score


def calculate_pcc(y_true, y_pred):
    """
    Pearson correlation coefficient.
    
    Parameters
    ----------
    y_true : np.ndarray
        Observed values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    pcc : float
        Pearson correlation coefficient.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    
    if len(y_true) < 2:
        return np.nan
    
    # constant arrays
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    
    return np.corrcoef(y_true, y_pred)[0, 1]


def calculate_r2(y_true, y_pred):
    """
    Coefficient of determination R².
    
    Parameters
    ----------
    y_true : np.ndarray
        Observed values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    r2 : float
        Coefficient of determination.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    
    if len(y_true) < 2:
        return np.nan
    
    return r2_score(y_true, y_pred)


if __name__ == "__main__":
    # self-test
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.5
    
    print(f"PCC: {calculate_pcc(y_true, y_pred):.4f}")
    print(f"R2:  {calculate_r2(y_true, y_pred):.4f}")
