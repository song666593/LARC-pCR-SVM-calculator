# -*- coding: utf-8 -*-
"""与建模脚本一致的 Winsorizer，独立模块以便 joblib 反序列化。"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lb_ = np.nanquantile(X, self.lower, axis=0)
        self.ub_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.clip(X, self.lb_, self.ub_)
