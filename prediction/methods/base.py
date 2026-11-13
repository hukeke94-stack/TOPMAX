#!/usr/bin/env python3
"""
Base class for prediction methods.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseMethod(ABC):
    """
    Base class of all prediction methods.
    """
    
    name = "BaseMethod"
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.fit_time = 0.0
        self.best_alpha = None
        self.runtime_metrics = {}
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Fit the model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training genotypes (n_samples, n_features).
        y_train : np.ndarray
            Training phenotypes (n_samples,).
        """
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """
        Predict.
        
        Parameters
        ----------
        X_test : np.ndarray
            Genotypes to predict (n_samples, n_features).
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values (n_samples,).
        """
        pass
    
    def fit_predict(self, X_train, y_train, X_test):
        """
        Fit, then predict.
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted values.
        total_time : float
            Total time in seconds (fit plus predict).
        """
        start_time = time.time()
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        total_time = time.time() - start_time
        
        self.fit_time = total_time
        return y_pred, total_time
    
    def preprocess(self, X_train, y_train, X_test):
        """
        Standardize the data.
        
        Returns
        -------
        X_train_scaled, y_train_scaled, X_test_scaled
        """
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        return X_train_scaled, y_train_scaled, X_test_scaled
    
    def inverse_transform_y(self, y_pred_scaled):
        """
        Back-transform standardized predictions.
        """
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred_scaled
    
    def get_info(self):
        """
        Return method information.
        """
        return {
            "name": self.name,
            "total_time": self.fit_time,  # fit plus predict time
            "best_alpha": self.best_alpha,
        }


# Method registry
AVAILABLE_METHODS = {}


def register_method(cls):
    """Decorator that registers a method."""
    AVAILABLE_METHODS[cls.name] = cls
    return cls


def get_method(method_name, **kwargs):
    """
    Create a method instance.
    
    Parameters
    ----------
    method_name : str
        Method name.
    **kwargs
        Method parameters.
        
    Returns
    -------
    method : BaseMethod
        Method instance.
    """
    if method_name not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(AVAILABLE_METHODS.keys())}")
    
    return AVAILABLE_METHODS[method_name](**kwargs)


def list_methods():
    """List the available methods."""
    print("Available methods:")
    for name in AVAILABLE_METHODS:
        print(f"  - {name}")
