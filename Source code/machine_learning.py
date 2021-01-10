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
from Pipeline._load_dataset import load_data
    

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
    parser.add_argument('--feature_drop_threshold', default=0.05, type=float, help='This is the parameter from step 1.')
    parser.add_argument('--no_input_output', default=False, action='store_true')
    parser.add_argument('--no_SCr', default=False, action='store_true')
    parser.add_argument('--model', default='GBM', type = str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--testing_ratio', default=0.1, type=float)
    parser.add_argument('--randomOversampling', default=True, action='store_true')
    parser.add_argument('--imputation_limit', default=0, type=int, help='if =0, no limit. if >0, imputation must not exeed certain hours')
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
    
        
    if args.fit == 'EICU':
        fit_dataset, external_dataset = 'EICU', 'MIMIC'
    elif args.fit == 'MIMIC':
        fit_dataset, external_dataset = 'MIMIC', 'EICU'
    if args.imputation_limit > 0:
        print('Now performing imputation with limit = %d' % args.imputation_limit)
        args.result_dir = workdir+'Results_limit='+str(args.imputation_limit)+'/'
    
    args.result_dir = args.result_dir[:-1] + '/fit_%s/feature_drop_threshold_%.2f/stage_%s/' % \
                    (args.fit, args.feature_drop_threshold, str(args.stage))
    
    TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    
    mtd = args.model
    if args.no_input_output and not args.no_SCr :
        args.result_dir += 'No_input_output/Serie_' + str(args.series) + '_Gap_' + str(args.gap) + '/'
    elif args.no_SCr and not args.no_input_output :
        args.result_dir += 'No_SCr/Serie_' + str(args.series) + '_Gap_' + str(args.gap) + '/'
    elif args.no_input_output and args.no_SCr:
        args.result_dir += 'No_input_output_SCr/Serie_' + str(args.series) + '_Gap_' + str(args.gap) + '/'
    else:
        args.result_dir += 'Use_All_features/Serie_' + str(args.series) + '_Gap_' + str(args.gap) + '/'
    
    results_dir_dataset = args.result_dir + 'random_seed=%d/%s/' % (args.seed, mtd)
        
    if not os.path.exists(results_dir_dataset):
        os.makedirs(results_dir_dataset)

    # create logger
    logger = logging.getLogger(TIMESTRING)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(results_dir_dataset+'mainlog.log', mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
        
# =============================================================================
#     Prepare dataset
# =============================================================================
    dataset = load_data(args)
# =============================================================================
#     Machine Learning
# =============================================================================
    print('Start machine learning ...')
    if randomOversampling:
        class_weight = None
    
    if args.model == 'all':
#        mtds = ["logit_l1", "logit_l2", "NN", "NN_l2", "DT", "RF", "AdaBoost", "GBM"]
        mtds = ["logit_l1", "DT", "RF", "AdaBoost", "GBM"]
    else:
        mtds = [args.model]
    for mtd in mtds:
        print(mtd)
        if mtd == "logit_l1": # around 8 mins for all folds
            regr_list = []
            hyperparam_list = [0.5, 1, 1.5, 2, 2.5, 3]
            for c in hyperparam_list:
                regr_list.append(LogisticRegression(penalty='l1', n_jobs = -1, C=c, solver='saga', class_weight = class_weight))
        if mtd == "DT":
            regr_list = []
            hyperparam_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            for c in hyperparam_list:
                regr_list.append(DecisionTreeClassifier(max_depth = c, class_weight = class_weight))
        if mtd == "RF":
            regr_list = []
            hyperparam_list = [16, 64, 128]
            for c in hyperparam_list:
                regr_list.append(RandomForestClassifier(n_estimators = c, class_weight = class_weight))
        if mtd == "AdaBoost":
            regr_list = []
            hyperparam_list = [16, 64, 128]
            for c in hyperparam_list:
                regr_list.append(AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, class_weight = class_weight), n_estimators = c))
        if mtd == "GBM":
            regr_list = []
            hyperparam_list = [16, 64, 128]
            for c in hyperparam_list:
                regr_list.append(GradientBoostingClassifier(n_estimators = c))


    # =============================================================================
    #             Find Hyper-parameters
    # =============================================================================
        f1_val_list = []
        for idx, regr in enumerate(regr_list):
                        
            auc_val, f1_val, precision_val, recall_val, mcc_val, sensitivity, specificity, ppv, npv, hitrate = 0,0,0,0,0,0,0,0,0,0
            for i in range(1, nfolds+1):
                print("%d fold CV -- %d/%d" % (nfolds, i, nfolds))
                logger.log(logging.INFO, "%d fold CV -- %d/%d" % (nfolds, i, nfolds))
                curr_data = dataset['fit']['fold_' + str(i)]
                
                X_train = curr_data['train']['X']
                y_train = curr_data['train']['y']
                X_val = curr_data['val']['X']
                y_val = curr_data['val']['y']
                
                X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
                X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
                regr.fit(X_train_reshaped, y_train)
                y_pred = regr.predict(X_val_reshaped)
                y_pred_proba = regr.predict_proba(X_val_reshaped)[:,1]
                fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
                auc_val += auc(fpr, tpr)/nfolds
                f1_val += f1_score(y_pred, y_val, average = 'macro')/nfolds
                # P, R test
                precision_val += precision_score(y_val, y_pred, average = 'macro')/nfolds
                recall_val += recall_score(y_val, y_pred, average = 'macro')/nfolds
                # other statistics
                mcc_val += matthews_corrcoef(y_val, y_pred)/nfolds
                TP, FP, TN, FN, sensitivity_temp, specificity_temp, ppv_temp, npv_temp, hitrate_temp = get_evaluation_res(y_val, y_pred)
                sensitivity += sensitivity_temp/nfolds
                specificity += specificity_temp/nfolds
                ppv += ppv_temp/nfolds
                npv += npv_temp/nfolds
                hitrate += hitrate_temp/nfolds
                
            f1_val_list.append(f1_val)
            print("Current model: %s, current hyper-parameter: %s, current AUC: %s, current F1 score: %.8f; current sensitivity: %.8f" % \
                  (mtd, str(hyperparam_list[idx]), auc_val, f1_val, sensitivity) )
            logger.log(logging.INFO, "Current model: %s, current hyper-parameter: %s, current AUC: %s, current F1 score: %.8f; current sensitivity: %.8f" % \
                       (mtd, str(hyperparam_list[idx]), auc_val, f1_val, sensitivity) )
        
        # Choose the optimal regr
        print("Best hyper-parameter: %s" % str(hyperparam_list[np.argmax(f1_val_list)]))
        logger.log(logging.INFO, "Best hyper-parameter: %s" % str(hyperparam_list[np.argmax(f1_val_list)]))
        print("Model:")
        logger.log(logging.INFO, "Model:")
        regr = regr_list[np.argmax(f1_val_list)]
        print(regr)
        logger.log(logging.INFO, regr)
        
    # =============================================================================
    #             Training after searching hyper-parameters
    # =============================================================================
        
        X_train = dataset['fit']['trainval']['X']
        y_train = dataset['fit']['trainval']['y']
        X_test = dataset['fit']['test']['X']
        y_test = dataset['fit']['test']['y']
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        regr.fit(X_train_reshaped, y_train)
        
        # AUC F1 train
        y_pred = regr.predict(X_train_reshaped)
        y_pred_proba = regr.predict_proba(X_train_reshaped)[:,1]
        fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
        auc_train = auc(fpr, tpr)
        f1_train = f1_score(y_pred, y_train, average = 'macro')
        
        # AUC F1 test
        y_pred = regr.predict(X_test_reshaped)
        y_pred_proba = regr.predict_proba(X_test_reshaped)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_test = auc(fpr, tpr)
        f1_test = f1_score(y_test, y_pred, average = 'macro')
        
        # P, R test
        precision_test = precision_score(y_test, y_pred, average = 'macro')
        recall_test = recall_score(y_test, y_pred, average = 'macro')
        
        # other statistics
        mcc = matthews_corrcoef(y_test, y_pred)
        TP, FP, TN, FN, sensitivity, specificity, ppv, npv, hitrate = get_evaluation_res(y_test, y_pred)

        print("[%s] train AUC: %.8f, test AUC: %.8f, train F1: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f, sensitivity: %.8f, specificity: %.8f, PPV: %.8f, NPV: %.8f" \
              % (fit_dataset, auc_train, auc_test, f1_train, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv))
        logger.log(logging.INFO, "[%s] train AUC: %.8f, test AUC: %.8f, train F1: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f, sensitivity: %.8f, specificity: %.8f, PPV: %.8f, NPV: %.8f" \
              % (fit_dataset, auc_train, auc_test, f1_train, f1_test, precision_test, recall_test, sensitivity, specificity, ppv, npv))
    
        with open(results_dir_dataset + 'regr_model.pkl', 'wb') as handle:
            pickle.dump(regr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + 'outputs_test_proba_%s.pkl' % fit_dataset, 'wb') as handle:
            pickle.dump(y_pred_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + 'outputs_test_bin_%s.pkl' % fit_dataset, 'wb') as handle:
            pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + 'outputs_test_true_%s.pkl' % fit_dataset, 'wb') as handle:
            pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + 'column_names.pkl', 'wb') as handle:
            pickle.dump(dataset['fit']['test']['column names'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        res_table = pd.DataFrame([auc_train, auc_test, f1_train, f1_test, precision_test, recall_test, \
                                  sensitivity, specificity, ppv, npv, hitrate, mcc])
        res_table.index = ['auc_train', 'auc_test', 'f1_train', 'f1_test', 'precision_test', 'recall_test', \
                           'sensitivity', 'specificity', 'ppv', 'npv', 'hitrate', 'MCC']
        res_table.to_csv(results_dir_dataset + 'performance_%s.csv' % fit_dataset)
        
        plt.figure(figsize=(8,4))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='%s test set (AUC = %0.4f%%)' % (mtd, 100*auc(fpr, tpr)))
        plt.axes().set_aspect('equal')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig(results_dir_dataset + "AUC_test_%s.png" % fit_dataset, dpi=300)
    
        # =============================================================================
        #             External validation
        # =============================================================================
        
        if args.imputation_limit == 0:
            X_test = dataset['external']['X']
            y_test = dataset['external']['y']
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            
            # AUC F1 test
            y_pred = regr.predict(X_test_reshaped)
            y_pred_proba = regr.predict_proba(X_test_reshaped)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc_test = auc(fpr, tpr)
            f1_test = f1_score(y_test, y_pred, average = 'macro')
            
            # P, R test
            precision_test = precision_score(y_test, y_pred, average = 'macro')
            recall_test = recall_score(y_test, y_pred, average = 'macro')
            
            # other statistics
            mcc = matthews_corrcoef(y_test, y_pred)
            TP, FP, TN, FN, sensitivity, specificity, ppv, npv, hitrate = get_evaluation_res(y_test, y_pred)
            
            print("[%s] test AUC: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f" \
                  % (external_dataset, auc_test, f1_test, precision_test, recall_test))
            logger.log(logging.INFO, "[%s] test AUC: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f" \
                  % (external_dataset, auc_test, f1_test, precision_test, recall_test))
        
            with open(results_dir_dataset + 'outputs_external_proba_%s.pkl' % external_dataset, 'wb') as handle:
                pickle.dump(y_pred_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(results_dir_dataset + 'outputs_external_bin_%s.pkl' % external_dataset, 'wb') as handle:
                pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(results_dir_dataset + 'outputs_external_true_%s.pkl' % external_dataset, 'wb') as handle:
                pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            res_table = pd.DataFrame([auc_test, f1_test, precision_test, recall_test, \
                                      sensitivity, specificity, ppv, npv, hitrate, mcc])
            res_table.index = ['auc_test', 'f1_test', 'precision_test', 'recall_test', \
                               'sensitivity', 'specificity', 'ppv', 'npv', 'hitrate', 'MCC']
            res_table.to_csv(results_dir_dataset + 'performance_%s.csv' % external_dataset)
            
            plt.figure(figsize=(8,4))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='%s test set (AUC = %0.4f%%)' % (mtd, 100*auc(fpr, tpr)))
            plt.axes().set_aspect('equal')
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
            plt.savefig(results_dir_dataset + "AUC_external_validation_%s.png" % external_dataset, dpi=300)
        
    
    
    
    
    
    
    
    
    
    
    
    
