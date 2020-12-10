#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import os
import random

from tqdm import tqdm
from nltk.tokenize.casual import TweetTokenizer
from typing import List, Union, Optional
from scipy.sparse import issparse, isspmatrix, spmatrix
from time import time
import pickle

from bpemb import BPEmb
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from vowpalwabbit.sklearn_vw import VWClassifier
from catboost import CatBoostClassifier
from argparse import ArgumentParser

from wrappers import Embedder, BPETokenizer, Lemmatizer, Stemmer, TokenizerWrapper, FeaturePipeline, OnlineSVM


df = pd.read_csv('train-balanced-sarcasm.csv', usecols=['comment', 'parent_comment', 'label'])
df.dropna(axis='index', how='any', inplace=True)
test_txt = df.comment.iloc[2]
y = df.label.values


preprocessing_configurations = [
    # tokenizer, [postprocessor, ]vectorizer
    ('bpe', 'bow'),
    ('bpe', 'tfidf'),
    ('bpe', 'hashing'),
    ('base', 'stem', 'bow'),
    ('base', 'lemma', 'bow'),
    ('base', 'stem', 'tfidf'),
    ('base', 'lemma', 'tfidf'),
    ('base', 'stem', 'hashing'),
    ('base', 'lemma', 'hashing'),
    ('base', 'lemma', 'glove_emb'),
]


def optimize_grid(model, params, n_jobs=10, verbose=0):
    return GridSearchCV(
        model, param_grid=params,
        n_jobs=n_jobs, scoring='f1',
        cv=5, refit=True, verbose=verbose
    )


def calc_metrics(model, X_test, y_test, X_train, y_train):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    metrics = [f1_score, accuracy_score, precision_score, recall_score, roc_auc_score]
    names = ['f1', 'acc', 'precision', 'recall', 'roc_auc']
    res = {}
    for m, n in zip(metrics, names):
        res[f'train_{n}'] = m(y_train, y_pred_train)
        res[f'test_{n}'] = m(y_test, y_pred_test)
    return res


result = {}
agg = {}

parser = ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

frac = 0.001 if args.debug else 0.2
X_train, X_test, y_train, y_test = train_test_split(df.comment.values, df.label.values, test_size=frac)

if args.debug:
    X_train = X_test
    y_train = y_test


for preproc_conf in tqdm(preprocessing_configurations, desc='optimization...'):
    print(f'Examination of configuration: {preproc_conf}')
    if len(preproc_conf) == 3:
        tokenizer, postprocessor, vectorizer = preproc_conf
    else:
        tokenizer, vectorizer = preproc_conf
        postprocessor = 'none'
    t = time()
    preprocessor = FeaturePipeline(tokenizer=tokenizer,
                                   postprocessor=postprocessor,
                                   vectorizer=vectorizer)
    X_train_features = preprocessor.fit_transform(X_train)
    X_test_features = preprocessor.transform(X_test)
    del preprocessor
    print(f'preprocessing took {time() - t:.4f} seconds')
    
    t = time()
    gs = optimize_grid(
        LinearSVC(dual=False),
        {'C': [.1, .5,  1., 3, 5,], 'penalty': ['l1', 'l2']},
        n_jobs=16, verbose=10
    ).fit(X_train_features, y_train.astype(np.bool))
    result[(preproc_conf, 'svm')] = gs.cv_results_
    agg[(preproc_conf, 'svm')] = calc_metrics(
        gs.best_estimator_, X_test_features, y_test.astype(np.bool),
        X_train_features, y_train.astype(np.bool))
    agg[(preproc_conf, 'svm')]['best_params'] = gs.best_params_
    print(agg[(preproc_conf, 'svm')])
    print(f'svm training took {time() - t:.4f} seconds')
    
    t = time()
    gs = optimize_grid(
        LogisticRegression(solver='liblinear', penalty='l1'),
        {'C': [.01, .1, .2, .5, .8, 1., 3, 5, 10]},
        n_jobs=16, verbose=10
    ).fit(X_train_features, y_train.astype(np.bool))
    result[(preproc_conf, 'sparse_logreg')] = gs.cv_results_
    agg[(preproc_conf, 'sparse_logreg')] = calc_metrics(
        gs.best_estimator_, X_test_features, y_test.astype(np.bool),
        X_train_features, y_train.astype(np.bool))
    agg[(preproc_conf, 'sparse_logreg')]['best_params'] = gs.best_params_
    print(agg[(preproc_conf, 'sparse_logreg')])
    print(f'sparse_logreg training took {time() - t:.4f} seconds')
    
    
    t = time()
    gs = optimize_grid(
        LogisticRegression(solver='liblinear'),
        {'C': [.01, .1, .2, .5, .8, 1., 3, 5, 10]},
        n_jobs=16, verbose=10
    ).fit(X_train_features, y_train.astype(np.bool))
    result[(preproc_conf, 'logreg')] = gs.cv_results_
    agg[(preproc_conf, 'logreg')] = calc_metrics(
        gs.best_estimator_, X_test_features, y_test.astype(np.bool),
        X_train_features, y_train.astype(np.bool))
    agg[(preproc_conf, 'logreg')]['best_params'] = gs.best_params_
    print(agg[(preproc_conf, 'logreg')])
    print(f'logreg training took {time() - t:.4f} seconds')

    t = time()
    gs = optimize_grid(
        XGBClassifier(n_estimators=1000, n_jobs=16),
        {'max_depth': [3, 5, 10], 'n_estimators': [500, 1000, 2000]}, n_jobs=1,
        verbose=10
    ).fit(X_train_features, y_train)
    result[(preproc_conf, 'xgb')] = gs.cv_results_
    agg[(preproc_conf, 'xgb')] = calc_metrics(
        gs.best_estimator_, X_test_features, y_test,
        X_train_features, y_train)
    agg[(preproc_conf, 'xgb')]['best_params'] = gs.best_params_
    print(f'xgb training took {time() - t:.4f} seconds')

    #t = time()
    #if isinstance(X_train_features, spmatrix):
    #    X_train_cat = X_train_features.tocsc()
    #    X_test_cat = X_test_features.tocsc()
    #else:
    #    X_train_cat = X_train_features
    #    X_test_cat = X_test_features
    #gs = optimize_grid(
    #    CatBoostClassifier(),
    #    {'max_depth': [3, 5, 10], 'n_estimators': [500, 1000, 2000]}, n_jobs=1,
    #    verbose=10
    #).fit(X_train_cat, y_train)
    #result[(preproc_conf, 'cat')] = gs.cv_results_
    #agg[(preproc_conf, 'cat')] = calc_metrics(
    #    gs.best_estimator_, X_test_cat, y_test,
    #    X_train_cat, y_train)
    #agg[(preproc_conf, 'cat')]['best_params'] = gs.best_params_
    #print(f'cat training took {time() - t:.4f} seconds')

    t = time()
    gs = optimize_grid(
        RandomForestClassifier(n_estimators=700, n_jobs=16, verbose=0),
        {'max_depth': [3, 5, 10], 'n_estimators': [700, 1000, 2000]}, n_jobs=1,
        verbose=10
    ).fit(X_train_features, y_train.astype(np.bool))
    result[(preproc_conf, 'rf')] = gs.cv_results_
    agg[(preproc_conf, 'rf')] = calc_metrics(
        gs.best_estimator_, X_test_features, y_test.astype(np.bool),
        X_train_features, y_train.astype(np.bool))
    agg[(preproc_conf, 'rf')]['best_params'] = gs.best_params_
    print(f'rf training took {time() - t:.4f} seconds')


results_df = pd.DataFrame.from_records(list(agg.values()))
results_df['model'] = [k[1] for k in agg.keys()]
results_df['tokenizer'] = [k[0][0] for k in agg.keys()]
results_df['postprocessor'] = [(k[0][1] if len(k[0]) == 3 else 'none') for k in agg.keys()]
results_df['vectorizer'] = [k[0][-1] for k in agg.keys()]

results_df.to_csv('training_results.csv', index=False)

with open('full_results.pkl', 'wb') as f:
    pickle.dump(result, f)
