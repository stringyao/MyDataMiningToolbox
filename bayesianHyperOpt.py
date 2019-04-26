import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
import catboost

import gc
import sys
import warnings

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import *

from bayes_opt import BayesianOptimization

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def LGB_bayesian(num_leaves,
                 max_depth,
                 max_bin,
                 bagging_fraction,
                 bagging_freq,
                 feature_fraction,
                 min_split_gain,
                 min_child_samples,
                 min_child_weight,
                 subsample,                
                 reg_alpha,
                 reg_lambda):
    
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    max_bin = int(max_bin)
    bagging_freq = int(bagging_freq)
    min_child_samples = int(min_child_samples)
    
    oof_preds = np.zeros(train_df.shape[0])
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    feats = [f for f in train_df.columns if f not in (['index', 'fraud'])]
    
    score = 0.
    
    for train_idx, valid_idx in kf.split(train_df[feats], target):

        train_x, train_y = train_df[feats].iloc[train_idx].append(aug_df[feats]), target.iloc[train_idx].append(aug_df.fraud)
        valid_x, valid_y = train_df[feats].iloc[valid_idx], target.iloc[valid_idx]

        xgtrain = lgb.Dataset(train_x, label=train_y,
                          feature_name=feats#, categorical_feature=cat_fts
                          )
        xgvalid = lgb.Dataset(valid_x, label=valid_y,
                          feature_name=feats#, categorical_feature=cat_fts
                          )

        evals_results = {}
        
        watchlist = [xgvalid]
        
        params = {
            'application': 'binary', 
            'boosting': 'gbdt', 
            'metric': 'auc', 
            'num_leaves': num_leaves, 
            'max_depth': max_depth, 
            'max_bin': max_bin, 
            'bagging_fraction': bagging_fraction, 
            'bagging_freq': bagging_freq, 
            'feature_fraction': feature_fraction, 
            'min_split_gain': min_split_gain, 
            'min_child_samples': min_child_samples, 
            'min_child_weight': min_child_weight, 
            'subsample': subsample,            
            'reg_alpha': reg_alpha, 
            'reg_lambda': reg_lambda, 
            'learning_rate': 0.01,
            'verbosity': -1, 
            'seed': SEED
        }
        
        model = lgb.train(params,
                      train_set=xgtrain,
                      num_boost_round=50000,
                      valid_sets=watchlist,
                      verbose_eval=0,
                      early_stopping_rounds=100)
        
        oof_preds[valid_idx] = model.predict(valid_x, num_iteration=model.best_iteration)
           
    score += roc_auc_score(target, oof_preds) 
    
    return score
    
# Initializing
bounds_LGB = {
    'num_leaves': (5, 30), 
    'max_depth': (-1, 12), 
    'max_bin': (30, 80), 
    'bagging_fraction': (0.2, 0.8), 
    'bagging_freq': (1, 10), 
    'feature_fraction': (0.2, 0.6), 
    'min_split_gain': (0.0, 1.0), 
    'min_child_samples': (25, 125), 
    'min_child_weight': (0.0, 1.0), 
    'reg_alpha': (0.0, 3.0),
    'reg_lambda': (0.0, 3.0),    
    'subsample': (0.0, 1.0),     
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=SEED)    
    
# Optimizing starts
init_points = 5
n_iter = 100

print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
    
# Get the best result
best_score = LGB_BO.max['target']
best_param = {
    'application': 'binary', 
    'boosting': 'gbdt', 
    'metric': 'auc', 
    'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
    'max_depth': int(LGB_BO.max['params']['max_depth']), 
    'max_bin': int(LGB_BO.max['params']['max_bin']), 
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']), 
    'feature_fraction': LGB_BO.max['params']['feature_fraction'], 
    'min_split_gain': LGB_BO.max['params']['min_split_gain'], 
    'min_child_samples': int(LGB_BO.max['params']['min_child_samples']), 
    'min_child_weight': LGB_BO.max['params']['min_child_weight'], 
    'subsample': LGB_BO.max['params']['subsample'],            
    'reg_alpha': LGB_BO.max['params']['reg_alpha'], 
    'reg_lambda': LGB_BO.max['params']['reg_lambda'], 
    'learning_rate': 0.01,
    'verbosity': -1, 
    'seed': SEED
}
    
    
