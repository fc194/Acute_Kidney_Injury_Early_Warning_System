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
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import RandomOverSampler
np.warnings.filterwarnings('ignore')
tqdm.pandas()

workdir = ''# change to your working directory
sys.path.append(workdir)
from Pipeline._load_dataset import load_data
    


def applyModel(X, y, model):
    
    X_reshaped = X.reshape(X.shape[0], -1)
    # AUC F1 test
    y_pred = model.predict(X_reshaped)
    y_pred_proba = model.predict_proba(X_reshaped)[:,1]
    
    ros = RandomOverSampler(random_state=0)
    idx_resampled, y_test_resampled = ros.fit_resample(np.arange(len(y)).reshape(-1,1), y)
    y = y[idx_resampled.reshape(-1)]
    y_pred = y_pred[idx_resampled.reshape(-1)]
    y_pred_proba = y_pred_proba[idx_resampled.reshape(-1)]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    auc_test = auc(fpr, tpr)
    f1_test = f1_score(y, y_pred)
    
    # P, R test
    precision_test = precision_score(y, y_pred)
    recall_test = recall_score(y, y_pred)
    
    # other statistics
    mcc = matthews_corrcoef(y, y_pred)
    TP, FP, TN, FN, sensitivity, specificity, ppv, npv, hitrate = get_evaluation_res(y, y_pred)
    
    nbins = 100
    fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred_proba, n_bins=nbins)
    reg = LinearRegression(fit_intercept=False).fit(X=mean_predicted_value.reshape(-1,1), y=fraction_of_positives.reshape(-1,1))
    calibration_slope = reg.coef_[0][0]
    calibration_in_the_large = reg.intercept_
    
    print("test AUC: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f, sensitivity: %.8f, specificity: %.8f, PPV: %.8f, NPV: %.8f" \
          % (auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv))
    return auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv, calibration_slope, calibration_in_the_large

def get_evaluation_res(y_actual, y_hat):
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           fp += 1
        if y_actual[i]==y_hat[i]==0:
           tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    return tp, fp, tn, fn, sensitivity, specificity, ppv, npv, hitrate
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', default=12, type=int, help='12 hours')
    parser.add_argument('--gap', default=12, type=int, help='12, 24, 48 hours')
    parser.add_argument('--stage', default=23, type=int)
    parser.add_argument('--fit', default='EICU', type = str)
    parser.add_argument('--no_input_output', default=False, action='store_true')
    parser.add_argument('--no_SCr', default=False, action='store_true')
    parser.add_argument('--imputation_limit', default=0, type=int, help='if =0, no limit. if >0, imputation must not exeed certain hours')
    parser.add_argument('--feature_drop_threshold', default=0.05, type=float, help='This is the parameter from step 1.')
    parser.add_argument('--model', default='GBM', type = str)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--testing_ratio', default=0.1, type=float)
    parser.add_argument('--randomOversampling', default=True, action='store_true')
    parser.add_argument('--result_dir', default=workdir+'Results/', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    series = args.series
    gap = args.gap
    randomOversampling = args.randomOversampling
    nfolds = args.nfolds
    mtd = args.model
    seed = args.seed
    stage = args.stage
    feature_drop_threshold = args.feature_drop_threshold
    
    
    if args.fit == 'EICU':
        fit = 'EICU'
        ext = 'MIMIC'
    elif args.fit == 'MIMIC':
        fit = 'MIMIC'
        ext = 'EICU'
    result_dir = args.result_dir + 'fit_%s/feature_drop_threshold_%.2f/stage_%d/Use_All_features/Serie_%d_Gap_%d/random_seed=%d/%s/' \
                        % (fit, feature_drop_threshold, stage, series, gap, seed, mtd)
        
# =============================================================================
#     Prepare dataset and model
# =============================================================================
    dataset = load_data(args)
    with open(result_dir + 'regr_model.pkl', 'rb') as f: regr_model = pickle.load(f)
    

    with open(workdir + 'Processed_Data/data_expand_all_stage_%d.pickle' % (args.stage), 'rb') as f:
        data_expand_all = pickle.load(f)
    
    table = pd.DataFrame()
    
    disease_features = ['UO', 'CR']
    for d_f in disease_features:
        try:
            print('')
            # ICUSTAY_ID subset where UO (or CR) == 1 for positive cohort and all negative cohort
            ICUSTAY_ID_valid = data_expand_all['EICU'].loc[(data_expand_all['EICU']['AKI%d_%s' % (stage, d_f)] == 1) | \
                                        (data_expand_all['EICU']['AKI'] == 0), 'ICUSTAY_ID'].unique()
            # print('[EICU - Entire Cohort]  %s: %d (positive: %d) ----> %d (positive: %d)' %
            #       (d_f,
            #        len(data_expand_all['EICU'].loc[:,'ICUSTAY_ID'].unique()),
            #        sum(data_expand_all['EICU'].drop_duplicates('ICUSTAY_ID')['AKI'] == 1),
            #        len(ICUSTAY_ID_valid),
            #        sum(data_expand_all['EICU'].loc[data_expand_all['EICU']['ICUSTAY_ID'].isin(ICUSTAY_ID_valid)].drop_duplicates('ICUSTAY_ID')['AKI'] == 1)))
            subset_idx = np.isin(dataset['fit']['test']['ICUSTAY_ID'], ICUSTAY_ID_valid) # get the index
            X = dataset['fit']['test']['X'][subset_idx]
            y = dataset['fit']['test']['y'][subset_idx]
            print('[EICU - Testing Set]  %s: %d (positive: %d) ----> %d (positive: %d)' % (d_f, len(subset_idx), sum(dataset['fit']['test']['y']), len(X), sum(y)))
            auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv, calibration_slope, calibration_in_the_large = \
                applyModel(X, y, regr_model)
            table = table.append({
                    'Disease': d_f, 'Cohort': 'EICU test',
                    'Subset patients': len(X), 'Subset positive patients': sum(y),
                    'AUC': auc_test, 'F1': f1_test, 'Precision': precision_test, 'Recall': recall_test,
                    'Sensitivity': sensitivity, 'Specificity': specificity, 'PPV': ppv, 'NPV': npv,
                    'Calibration_slope': calibration_slope,'Calibration-in-the-large': calibration_in_the_large
                    }, ignore_index = True)
    
    
    
            # ICUSTAY_ID subset where UO (or CR) == 1 for positive cohort and all negative cohort
            ICUSTAY_ID_valid = data_expand_all['MIMIC'].loc[(data_expand_all['MIMIC']['AKI%d_%s' % (stage, d_f)] == 1) | \
                                        (data_expand_all['MIMIC']['AKI'] == 0), 'ICUSTAY_ID'].unique()
            # print('[MIMIC - Entire Cohort] %s: %d (positive: %d) ----> %d (positive: %d)' %
            #       (d_f,
            #        len(data_expand_all['MIMIC'].loc[:,'ICUSTAY_ID'].unique()),
            #        sum(data_expand_all['MIMIC'].drop_duplicates('ICUSTAY_ID')['AKI'] == 1),
            #        len(ICUSTAY_ID_valid),
            #        sum(data_expand_all['MIMIC'].loc[data_expand_all['MIMIC']['ICUSTAY_ID'].isin(ICUSTAY_ID_valid)].drop_duplicates('ICUSTAY_ID')['AKI'] == 1)))
            subset_idx = np.isin(dataset['external']['ICUSTAY_ID'], ICUSTAY_ID_valid) # get the index
            X = dataset['external']['X'][subset_idx]
            y = dataset['external']['y'][subset_idx]
            print('[MIMIC - Testing Set] %s: %d (positive: %d) ----> %d (positive: %d)' % (d_f, len(subset_idx), sum(dataset['external']['y']), len(X), sum(y)))
            auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv, calibration_slope, calibration_in_the_large = \
                applyModel(X, y, regr_model)
            table = table.append({
                    'Disease': d_f, 'Cohort': 'MIMIC external',
                    'Subset patients': len(X), 'Subset positive patients': sum(y),
                    'AUC': auc_test, 'F1': f1_test, 'Precision': precision_test, 'Recall': recall_test,
                    'Sensitivity': sensitivity, 'Specificity': specificity, 'PPV': ppv, 'NPV': npv,
                    'Calibration_slope': calibration_slope,'Calibration-in-the-large': calibration_in_the_large
                    }, ignore_index = True)
        except:
            continue

    
    
    disease_features = ['CHF','SEP','DIA','RES']
    
    for d_f in disease_features:
        try:
            print('')
            mask = dataset['fit']['test']['X'][:,:,d_f == dataset['fit']['test']['column names']].reshape(dataset['fit']['test']['X'].shape[0],-1)
            subset_idx = np.min(mask, axis = 1) == 1 # get the index where disease feature == 1 for all series
            X = dataset['fit']['test']['X'][subset_idx]
            y = dataset['fit']['test']['y'][subset_idx]
            ICUSTAY_ID = dataset['fit']['test']['ICUSTAY_ID']
            print('[EICU]  %s: %d (positive: %d) ----> %d (positive: %d)' % (d_f, len(subset_idx), sum(dataset['fit']['test']['y']), len(X), sum(y)))
            auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv, calibration_slope, calibration_in_the_large = \
                applyModel(X, y, regr_model)
            table = table.append({
                    'Disease': d_f, 'Cohort': 'EICU test',
                    'Subset patients': len(X), 'Subset positive patients': sum(y),
                    'AUC': auc_test, 'F1': f1_test, 'Precision': precision_test, 'Recall': recall_test,
                    'Sensitivity': sensitivity, 'Specificity': specificity, 'PPV': ppv, 'NPV': npv,
                    'Calibration_slope': calibration_slope,'Calibration-in-the-large': calibration_in_the_large
                    }, ignore_index = True)
    
            
            mask = dataset['external']['X'][:,:,d_f == dataset['external']['column names']].reshape(dataset['external']['X'].shape[0],-1)
            subset_idx = np.min(mask, axis = 1) == 1 # get the index where disease feature == 1 for all series
            X = dataset['external']['X'][subset_idx]
            y = dataset['external']['y'][subset_idx]
            ICUSTAY_ID = dataset['external']['ICUSTAY_ID']
            print('[MIMIC] %s: %d (positive: %d) ----> %d (positive: %d)' % (d_f, len(subset_idx), sum(dataset['external']['y']), len(X), sum(y)))
            auc_test, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv, calibration_slope, calibration_in_the_large = \
                applyModel(X, y, regr_model)
            table = table.append({
                    'Disease': d_f, 'Cohort': 'MIMIC external',
                    'Subset patients': len(X), 'Subset positive patients': sum(y),
                    'AUC': auc_test, 'F1': f1_test, 'Precision': precision_test, 'Recall': recall_test,
                    'Sensitivity': sensitivity, 'Specificity': specificity, 'PPV': ppv, 'NPV': npv,
                    'Calibration_slope': calibration_slope,'Calibration-in-the-large': calibration_in_the_large
                    }, ignore_index = True)
        except:
            continue
    
    
    table.to_csv(result_dir + 'predict_disease_table.csv')
    
    
    
