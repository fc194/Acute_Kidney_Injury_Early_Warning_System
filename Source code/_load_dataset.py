#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, copy
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import sys, os, platform
import pickle
import itertools
import argparse
import urllib
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score
from tqdm import tqdm
import gc, logging, copy, pickle, math, random, argparse, time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler
np.warnings.filterwarnings('ignore')
tqdm.pandas()

workdir = ''# change to your working directory
sys.path.append(workdir)

def processing(data, series = 6, gap = 6):
    Y = data[['AKI', 'HOURS']]
    X = data.loc[:, [c for c in data.columns if c not in \
                     ['INPUT_MINUS_OUTPUT_6HR','INPUT_MINUS_OUTPUT_12HR','INPUT_MINUS_OUTPUT_24HR', 'AKI']]
                ]
        
    if not all(np.isreal(item) == True for item in X['GENDER'].values):
        X['GENDER'][X['GENDER'] == 'F'] = 0
        X['GENDER'][X['GENDER'] == 'M'] = 1
    
        
    diff = [y-x for x, y in zip(Y.loc[:,'HOURS'], Y.loc[1:,'HOURS'])]
    diff.append(-1)
    diff = np.asarray(diff)
    
    y_is_vent = Y.loc[[a and b for a, b in zip((Y['AKI'] == 1).values, diff < 0)],:]
    y_non_vent = Y.loc[Y['AKI'] == 0, ]

    y_is_vent_index = y_is_vent.index.values
    y_non_vent_index = y_non_vent.index.values
    
    
    X_is_vent_reshaped, columns, ICUSTAY_ID_is_vent = crop_reshape(data, X, y_is_vent_index, series, gap)
    y_is_vent = [1]*X_is_vent_reshaped.shape[0]
    
    X_non_vent_reshaped, columns, ICUSTAY_ID_non_vent = crop_reshape(data, X, y_non_vent_index, series, gap)
    y_non_vent = [0]*X_non_vent_reshaped.shape[0]
    
    X = np.concatenate((X_is_vent_reshaped, X_non_vent_reshaped), axis=0)
    y = np.concatenate((y_is_vent, y_non_vent))
    ICUSTAY_ID = np.concatenate((ICUSTAY_ID_is_vent,ICUSTAY_ID_non_vent)).astype(int)
    
    columns = columns[[c not in ['AKI'] for c in columns]]
    return X, y, columns, ICUSTAY_ID

def crop_reshape(data, X, y, series, gap):
    data_index = y - gap - 1
    data_index = data_index.reshape(1, len(data_index))
    data_index_concat = y.reshape(1, len(y)) # also add the label row
    for i in range(series):
        data_index_concat = np.concatenate((data_index-i, data_index_concat), axis = 0)
    X_ICU = X['ICUSTAY_ID'].values
    X_ICU_series = X_ICU[data_index_concat]
    index2keep1 = data_index_concat[0,:] >= 0 # remove columns that have negative index
    index2keep2 = np.all(X_ICU_series == X_ICU_series[0,:], axis = 0) # is all the column with the same ICUSTAY_ID?
    index2keep = np.logical_and(index2keep1, index2keep2)
    data_index_concat = data_index_concat[:, index2keep]
    data_index_concat = data_index_concat[0:data_index_concat.shape[0]-1,:] #remove the label row
    y = y[index2keep]
    
    # before reshape, let's remove some irrelevant columns
    cols2remove = ['ICUSTAY_ID', 'HOS_DEATH', 'HOS_LOS', 'ICU_DEATH', 'ICU_LOS']
    cols2keep = [X.columns.get_loc(c) for c in X.columns if c not in cols2remove]
    cols2keep_names = X.columns[cols2keep].values
    X_np = X.values.astype(float)
    
#     Reshape
    X_reshaped = X_np[data_index_concat, :]
    ICUSTAY_ID = X_reshaped[0,:,0]
    X_reshaped = X_reshaped[:,:, cols2keep]
    X_reshaped = np.swapaxes(X_reshaped,0,1)
    return X_reshaped, cols2keep_names, ICUSTAY_ID


def imputation_on_the_fly(data_expand, series = 6, gap = 6, imputation_limit = 0):
    # =============================================================================
    #       Data interpolation
    # =============================================================================
    print('Now perform data interpolation...')
    def fillNAinterpolation(x):
        if imputation_limit == 0:
            x2 = x.fillna(method='ffill', axis = 0)
            x2 = x2.fillna(method='bfill', axis = 0)
        if imputation_limit > 0:
            x2 = x.fillna(method='ffill', axis = 0, limit = imputation_limit)
            x2 = x2.fillna(method='bfill', axis = 0, limit = imputation_limit)
        return x2
    data_expand_fillNA = data_expand.groupby(by = 'ICUSTAY_ID').progress_apply(lambda x: fillNAinterpolation(x))
    
    print('Before drop NA - data shape:', data_expand_fillNA.shape)
    data_expand_fillNA.dropna(inplace = True)
    data_expand_fillNA.reset_index(inplace = True, drop = True)
    print('After drop NA - data shape:', data_expand_fillNA.shape)
    # =============================================================================
    #       Data reshape
    # =============================================================================
    X, y, columns, ICUSTAY_ID = processing(data = copy.deepcopy(data_expand_fillNA),\
                                              series = series, gap = gap)
    print('After data reshape, AKI:', sum(y == 1), '( unique:',len(np.unique(ICUSTAY_ID[y == 1])),')', \
          ' |  None AKI:', sum(y == 0), '( unique:',len(np.unique(ICUSTAY_ID[y == 0])),')')
        
    return X, y, columns, ICUSTAY_ID
        


def normalization(df, scaler=None):
    if not scaler:
        scaler = StandardScaler().fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df))
    df_scaled.columns, df_scaled.index = df.columns, df.index
    df_scaled['ICUSTAY_ID'], df_scaled['AKI'] = df['ICUSTAY_ID'], df['AKI']
    return df, scaler
        
        
def averageImputation(df, by='ICUSTAY_ID', columns = ['INPUT_12HR', 'OUTPUT_12HR']):
    def aveimp(x, averages):
        for c in averages.index:
            if x.loc[:, c].isnull().all():
                x.loc[:, c] = averages.loc[c]
        return x
    averages = df.loc[:, columns].mean(axis=0, skipna=True)
    df = df.groupby(by=by).progress_apply(lambda x: aveimp(x, averages))
    return df

def randomOverSampling(X, y, ICUSTAY_ID, random_state=0):
    print('Handling the imbalanced dataset by RandomOverSampler ...')
    ros = RandomOverSampler(random_state=random_state)
    X_index, y_resampled = ros.fit_resample(np.array(range(len(X))).reshape(len(X), 1), y)
    X_index = X_index.reshape(-1)
    X_resampled = X[X_index,:,:]
    ICUSTAY_ID = ICUSTAY_ID[X_index]
    return X_resampled, y_resampled, ICUSTAY_ID

def load_data(args, save=False):
    series = args.series
    gap = args.gap
    if args.fit == 'EICU':
        fit_dataset, external_dataset = 'EICU', 'MIMIC'
    elif args.fit == 'MIMIC':
        fit_dataset, external_dataset = 'MIMIC', 'EICU'
        
    if os.path.exists(args.result_dir + 'dataset.pickle'):
        with open(args.result_dir + 'dataset.pickle', 'rb') as f: dataset = pickle.load(f)
    else:
        data_expand_MIMIC = pd.read_csv(workdir + 'Processed_Data/feature_drop_threshold=%.2f/data_expand_filtered_AKI%s_MIMIC.csv' % (args.feature_drop_threshold, args.stage), index_col = 0)
        data_expand_EICU = pd.read_csv(workdir + 'Processed_Data/feature_drop_threshold=%.2f/data_expand_filtered_AKI%s_EICU.csv' % (args.feature_drop_threshold, args.stage), index_col = 0)
        
    #    print('Current column names: \n', data_expand_EICU.columns.values)
        data_expand_EICU = data_expand_EICU.loc[:,[c for c in data_expand_EICU.columns.values if 'TIME' not in c]]
        data_expand_MIMIC = data_expand_MIMIC.loc[:,[c for c in data_expand_MIMIC.columns.values if 'TIME' not in c]]
        
        
        if args.no_input_output:
            print('Remove input output feature ...')
            data_expand_MIMIC = data_expand_MIMIC[[col for col in data_expand_MIMIC.columns if 'INPUT' not in col and 'OUTPUT' not in col]]
            data_expand_EICU = data_expand_EICU[[col for col in data_expand_MIMIC.columns if 'INPUT' not in col and 'OUTPUT' not in col]]
        if args.no_SCr:
            data_expand_MIMIC = data_expand_MIMIC[[col for col in data_expand_MIMIC.columns if 'CR' not in col]]
            data_expand_EICU = data_expand_EICU[[col for col in data_expand_MIMIC.columns if 'CR' not in col]]
            print('Remove SCr feature ...')
        
        col_2b_removed = ['ETHNICITY', 'AKI_RRT', 'ICU_CLASS'] #['ETHNICITY','PF','ALT','AST','TBB']
        print('Remove features that has so many missing data:', col_2b_removed)
        data_expand_MIMIC.drop(col_2b_removed, axis = 1, inplace = True)
        data_expand_EICU.drop(col_2b_removed, axis = 1, inplace = True)
        # move ICUSTAY_ID front
        data_expand_MIMIC = data_expand_MIMIC.loc[:, ['ICUSTAY_ID'] + [x for x in data_expand_MIMIC.columns if x != "ICUSTAY_ID"]]
        data_expand_EICU = data_expand_EICU.loc[:, ['ICUSTAY_ID'] + [x for x in data_expand_EICU.columns if x != "ICUSTAY_ID"]]
    
    
        if args.fit == 'EICU':
            df_fit = data_expand_EICU
            df_ext = data_expand_MIMIC
        elif args.fit == 'MIMIC':
            df_fit = data_expand_MIMIC
            df_ext = data_expand_EICU
            
        print('forming dataset ... [Training]: %s; [External validation]: %s.' % (fit_dataset, external_dataset))
                
        dataset = {}
        
        dataset['fit'] = {}
        # split trainval and test
        testing_uniqueID = random.sample(set(df_fit['ICUSTAY_ID']), round(len(np.unique(df_fit['ICUSTAY_ID']))*args.testing_ratio))
        trainval_uniqueID = np.array([i for i in np.unique(df_fit['ICUSTAY_ID']) if i not in testing_uniqueID])
        dataset['fit']['ICUSTAY_ID_test'] = testing_uniqueID
        dataset['fit']['ICUSTAY_ID_trainval'] = trainval_uniqueID
        
        # trainval
        dataset['fit']['trainval'] = {}
        df_fit_trainval = df_fit.loc[np.isin(df_fit['ICUSTAY_ID'], trainval_uniqueID), :]
        # normalization on trainval
        df_fit_trainval_scaled, scaler = normalization(df_fit_trainval)
        # average imputation
        df_fit_trainval_scaled = averageImputation(df_fit_trainval_scaled, by='ICUSTAY_ID', columns = ['INPUT_12HR', 'OUTPUT_12HR'])
        # imputation_on_the_fly
        X_fit_trainval, y_fit_trainval, colnames_trainval, ICUSTAY_ID_trainval = imputation_on_the_fly(df_fit_trainval_scaled, series = series, gap = gap, imputation_limit = args.imputation_limit)
        # random oversampling
        dataset['fit']['trainval']['X'], dataset['fit']['trainval']['y'], dataset['fit']['trainval']['ICUSTAY_ID'] = \
            randomOverSampling(X_fit_trainval, y_fit_trainval, ICUSTAY_ID_trainval, random_state=args.seed)
        
        # test
        dataset['fit']['test'] = {}
        df_fit_test = df_fit.loc[np.isin(df_fit['ICUSTAY_ID'], testing_uniqueID), :]
        # apply normalization
        df_fit_test_scaled, _ = normalization(df_fit_test, scaler = scaler)
        dataset['fit']['test']['X'], dataset['fit']['test']['y'], dataset['fit']['test']['column names'], dataset['fit']['test']['ICUSTAY_ID'] = \
            imputation_on_the_fly(df_fit_test_scaled, series = series, gap = gap, imputation_limit = args.imputation_limit)


        # external
        dataset['external'] = {}
        # apply normalization
        df_ext_scaled, _ = normalization(df_ext, scaler = scaler)
        dataset['external']['X'], dataset['external']['y'], dataset['external']['column names'], dataset['external']['ICUSTAY_ID'] = \
            imputation_on_the_fly(df_ext_scaled, series = series, gap = gap, imputation_limit = args.imputation_limit)
        
        kf = KFold(n_splits=args.nfolds, random_state=args.seed, shuffle=True)
        for i, (train_index, val_index) in zip(range(1, args.nfolds+1), kf.split(trainval_uniqueID)):
            fold = 'fold_' + str(i)
            print(fold)
            dataset['fit'][fold] = {}
            
            # train
            dataset['fit'][fold]['train'] = {}
            ICUSTAY_ID_train = trainval_uniqueID[train_index]
            df_fit_train = df_fit.loc[np.isin(df_fit['ICUSTAY_ID'], ICUSTAY_ID_train), :]
            dataset['fit'][fold]['train']['ICUSTAY_ID_train'] = ICUSTAY_ID_train
            # normalization on trainval
            df_fit_train_scaled, scaler = normalization(df_fit_train)
            # average imputation
            df_fit_train_scaled = averageImputation(df_fit_train_scaled, by='ICUSTAY_ID', columns = ['INPUT_12HR', 'OUTPUT_12HR'])
            # imputation_on_the_fly
            X_fit_train, y_fit_train, dataset['fit'][fold]['train']['column names'], dataset['fit'][fold]['train']['ICUSTAY_ID'] = \
                imputation_on_the_fly(df_fit_train_scaled, series = series, gap = gap, imputation_limit = args.imputation_limit)
            # random oversampling
            dataset['fit'][fold]['train']['X'], dataset['fit'][fold]['train']['y'], dataset['fit'][fold]['train']['ICUSTAY_ID'] = \
                randomOverSampling(X_fit_train, y_fit_train, dataset['fit'][fold]['train']['ICUSTAY_ID'], random_state=args.seed)
        
            # validation
            dataset['fit'][fold]['val'] = {}
            ICUSTAY_ID_val = trainval_uniqueID[val_index]
            df_fit_val = df_fit.loc[np.isin(df_fit['ICUSTAY_ID'], ICUSTAY_ID_val), :]
            # apply normalization
            df_fit_val_scaled, _ = normalization(df_fit_val, scaler)
            # imputation_on_the_fly
            dataset['fit'][fold]['val']['X'], dataset['fit'][fold]['val']['y'], dataset['fit'][fold]['val']['column names'], dataset['fit'][fold]['val']['ICUSTAY_ID'] = \
                imputation_on_the_fly(df_fit_val_scaled, series = series, gap = gap, imputation_limit = args.imputation_limit)
        if save:
            with open(args.result_dir + 'dataset.pickle', 'wb') as f: pickle.dump(dataset, f)
        
    return dataset
