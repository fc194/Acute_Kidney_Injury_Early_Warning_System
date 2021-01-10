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
from tqdm import tqdm
import gc, logging, copy, pickle, math, random, argparse, time
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import scipy.stats
from imblearn.over_sampling import RandomOverSampler
import matplotlib
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.font_manager
from matplotlib import rcParams
import matplotlib.pyplot as plt


from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression

from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score

np.warnings.filterwarnings('ignore')
tqdm.pandas()

    
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
    

workdir = ''# change to your working directory
    

# =============================================================================
# cm = plt.get_cmap('OrRd_r')
# possible colormap are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, 
# BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys,
# Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1,
# Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r,
# PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
# RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r,
# Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r,
# YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg,
# brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper,
# copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray,
# gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r,
# gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r,
# gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r,
# nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism,
# prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r,
# tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r,
# twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r
# =============================================================================
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data).astype(float)
    a = a[~np.isnan(a)]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    ci = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, ci

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_drop_threshold', default=0.05, type=float)
    parser.add_argument('--stage', default=23, type=int)
    parser.add_argument('--fit', default='EICU', type = str)
    parser.add_argument('--result_dir', default=workdir+'Results/', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fit == 'EICU':
        fit = 'EICU'
        ext = 'MIMIC'
    elif args.fit == 'MIMIC':
        fit = 'MIMIC'
        ext = 'EICU'
        
    nbins = 10
    args.result_dir = args.result_dir + 'fit_'+fit+'/feature_drop_threshold_' + str(args.feature_drop_threshold) + '/stage_'+str(args.stage)+'/Use_All_features/'
    exp_dirs = [args.result_dir + s + '/' for s in os.listdir(args.result_dir) if 'Serie' in s]
    
    methods = ['logit_l1', 'DT', 'RF', 'AdaBoost', 'GBM']
    seeds = [1,2,3,4,5]
    
    df_fit = pd.DataFrame()
    df_ext = pd.DataFrame()
    df_disease = pd.DataFrame()
    y = {}
    model = {}
    for d in tqdm(exp_dirs):
        serie = int(d.split('Serie_')[1].split('_Gap')[0])
        gap = int(d.split('Gap_')[1].split('/')[0])
        y[(serie, gap)] = {}
        model[(serie, gap)] = {}
        for seed in seeds:
            y[(serie, gap)][seed] = {}
            model[(serie, gap)][seed] = {}
            print('Seed = %d' % seed)
            for mtd in methods:
                y[(serie, gap)][seed][mtd] = {}
                y[(serie, gap)][seed][mtd]['fit'] = {}
                y[(serie, gap)][seed][mtd]['ext'] = {}
                model[(serie, gap)][seed][mtd] = {}
                
                result_dir = d + 'random_seed=%d/%s/' % (seed, mtd)
                # fit
                df = pd.read_csv(result_dir + '/performance_%s.csv' % fit, index_col = 0).transpose()
                df['serie'], df['gap'], df['model'], df['seed'] = serie, gap, mtd, seed
                with open(result_dir + 'outputs_test_true_%s.pkl' % fit,'rb') as f: y[(serie, gap)][seed][mtd]['fit']['true'] = pickle.load(f)
                with open(result_dir + 'outputs_test_bin_%s.pkl' % fit,'rb') as f: y[(serie, gap)][seed][mtd]['fit']['pred'] = pickle.load(f)
                with open(result_dir + 'outputs_test_proba_%s.pkl' % fit,'rb') as f: y[(serie, gap)][seed][mtd]['fit']['pred_proba'] = pickle.load(f)
                with open(result_dir + 'regr_model.pkl','rb') as f: model[(serie, gap)][seed][mtd] = pickle.load(f)
                # calibration
                y_test = y[(serie, gap)][seed][mtd]['fit']['true']
                y_pred_proba = y[(serie, gap)][seed][mtd]['fit']['pred_proba']
                y_pred = y[(serie, gap)][seed][mtd]['fit']['pred']
                
                ros = RandomOverSampler(random_state=0)
                idx_resampled, y_test_resampled = ros.fit_resample(np.arange(len(y_test)).reshape(-1,1), y_test)
                y_pred_proba_resampled = y_pred_proba[idx_resampled.reshape(-1)]
                y_pred_resampled = y_pred[idx_resampled.reshape(-1)]
                fpr, tpr, thresholds = roc_curve(y_test_resampled, y_pred_proba_resampled)
                df['auc_test'] = auc(fpr, tpr)
                df['precision_test'] = precision_score(y_test_resampled, y_pred_resampled)
                df['recall_test'] = recall_score(y_test_resampled, y_pred_resampled)
                df['f1_test'] = f1_score(y_test_resampled, y_pred_resampled)
                tp, fp, tn, fn, df['sensitivity'], df['specificity'], df['ppv'], df['npv'], df['hitrate'] = get_evaluation_res(y_test_resampled, y_pred_resampled)
                
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test_resampled, y_pred_proba_resampled, n_bins=nbins)
                reg = LinearRegression(fit_intercept=False).fit(X=mean_predicted_value.reshape(-1,1), y=fraction_of_positives.reshape(-1,1))
                df['Calibration slope'] = reg.coef_[0][0]
                df['Calibration-in-the-large'] = reg.intercept_
                df_fit = df_fit.append(df, ignore_index = True)
                
                # external
                df = pd.read_csv(result_dir + '/performance_%s.csv' % ext, index_col = 0).transpose()
                df['serie'], df['gap'], df['model'], df['seed'] = serie, gap, mtd, seed
                with open(result_dir + 'outputs_external_true_%s.pkl' % ext,'rb') as f: y[(serie, gap)][seed][mtd]['ext']['true'] = pickle.load(f)
                with open(result_dir + 'outputs_external_bin_%s.pkl' % ext,'rb') as f: y[(serie, gap)][seed][mtd]['ext']['pred'] = pickle.load(f)
                with open(result_dir + 'outputs_external_proba_%s.pkl' % ext,'rb') as f: y[(serie, gap)][seed][mtd]['ext']['pred_proba'] = pickle.load(f)
                # calibration
                y_test = y[(serie, gap)][seed][mtd]['ext']['true']
                y_pred_proba = y[(serie, gap)][seed][mtd]['ext']['pred_proba']
                y_pred = y[(serie, gap)][seed][mtd]['ext']['pred']
                
                ros = RandomOverSampler(random_state=0)
                idx_resampled, y_test_resampled = ros.fit_resample(np.arange(len(y_test)).reshape(-1,1), y_test)
                y_pred_proba_resampled = y_pred_proba[idx_resampled.reshape(-1)]
                y_pred_resampled = y_pred[idx_resampled.reshape(-1)]
                fpr, tpr, thresholds = roc_curve(y_test_resampled, y_pred_proba_resampled)
                df['auc_test'] = auc(fpr, tpr)
                df['precision_test'] = precision_score(y_test_resampled, y_pred_resampled)
                df['recall_test'] = recall_score(y_test_resampled, y_pred_resampled)
                df['f1_test'] = f1_score(y_test_resampled, y_pred_resampled)
                tp, fp, tn, fn, df['sensitivity'], df['specificity'], df['ppv'], df['npv'], df['hitrate'] = get_evaluation_res(y_test_resampled, y_pred_resampled)
                
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test_resampled, y_pred_proba_resampled, n_bins=nbins)
                reg = LinearRegression(fit_intercept=False).fit(X=mean_predicted_value.reshape(-1,1), y=fraction_of_positives.reshape(-1,1))
                df['Calibration slope'] = reg.coef_[0][0]
                df['Calibration-in-the-large'] = reg.intercept_
                df_ext = df_ext.append(df, ignore_index = True)
                
                # disease
                df = pd.read_csv(result_dir + '/predict_disease_table.csv', index_col = 0).transpose()
                df = df.transpose()
                df['serie'], df['gap'], df['model'], df['seed'] = serie, gap, mtd, seed
                df_disease = df_disease.append(df, ignore_index = True)
    
# =============================================================================
#     Table summary
# =============================================================================
    
    metrics = ['auc_test', 'f1_test', 'precision_test', \
               'recall_test', 'sensitivity', 'specificity', 'ppv', 'npv',\
               'Calibration slope','Calibration-in-the-large']
        
    series_gaps = ['12_12', '12_24', '12_48']
    columns = [['%s | %s' % (s,m) for m in methods] for s in series_gaps]
    columns = list(itertools.chain.from_iterable(columns))
    index = metrics
    table_fit = pd.DataFrame(index = index, columns = columns)
    table_ext = pd.DataFrame(index = index, columns = columns)
    
    for idx, col in itertools.product(index, columns):
        metric = idx
        mdl = col.split(' | ')[1]
        s_g = col.split(' | ')[0]
        s, g = s_g.split('_')
        s, g = int(s), int(g)

        df = df_fit.loc[(df_fit['model'] == mdl) & \
                        (df_fit['serie'] == s) & \
                        (df_fit['gap'] == g)]
        mean, ci = mean_confidence_interval(df[metric])
        table_fit.loc[idx, col] = '%.2f (%.2f-%.2f)' % (mean, mean-ci, mean+ci)
        
        df = df_ext.loc[(df_ext['model'] == mdl) & \
                        (df_ext['serie'] == s) & \
                        (df_ext['gap'] == g)]
        mean, ci = mean_confidence_interval(df[metric])
        table_ext.loc[idx, col] = '%.2f (%.2f-%.2f)' % (mean, mean-ci, mean+ci)

    table_fit.insert(5, '', '')
    table_fit.insert(11, '', '', allow_duplicates = True)

    table_ext.insert(5, '', '')
    table_ext.insert(11, '', '', allow_duplicates = True)


    table_fit.to_csv(args.result_dir + 'table_fit_95%CI.csv')
    table_ext.to_csv(args.result_dir + 'table_ext_95%CI.csv')
    
    
# =============================================================================
#     Table summary: Disease-specific
# =============================================================================
    methods = ['logit_l1', 'GBM', 'RF']
    diseases = ['UO', 'CR', 'CHF', 'SEP', 'DIA', 'RES']
    metrics = ['AUC', 'F1', 'NPV', 'PPV', 'Precision', 'Recall', 'Sensitivity', 'Specificity','Calibration_slope','Calibration-in-the-large']
    series_gaps = ['12_12', '12_24', '12_48']
    columns = [['%s | %s' % (s,m) for m in methods] for s in series_gaps]
    columns = list(itertools.chain.from_iterable(columns))
    index = metrics
    
    for disease in diseases:
        table_fit = pd.DataFrame(index = index, columns = columns)
        table_ext = pd.DataFrame(index = index, columns = columns)
        
        for idx, col in itertools.product(index, columns):
            metric = idx
            mdl = col.split(' | ')[1]
            s_g = col.split(' | ')[0]
            s, g = s_g.split('_')
            s, g = int(s), int(g)
    
            df = df_disease.loc[(df_disease['model'] == mdl) & \
                                (df_disease['serie'] == s) & \
                                (df_disease['gap'] == g) & \
                                (df_disease['Cohort'] == 'EICU test') & \
                                (df_disease['Disease'] == disease)]
            mean, ci = mean_confidence_interval(df[metric])
            table_fit.loc[idx, col] = '%.2f (%.2f-%.2f)' % (mean, mean-ci, mean+ci)
            
            df = df_disease.loc[(df_disease['model'] == mdl) & \
                                (df_disease['serie'] == s) & \
                                (df_disease['gap'] == g) & \
                                (df_disease['Cohort'] == 'MIMIC external') & \
                                (df_disease['Disease'] == disease)]
            mean, ci = mean_confidence_interval(df[metric])
            table_ext.loc[idx, col] = '%.2f (%.2f-%.2f)' % (mean, mean-ci, mean+ci)
            
            table_fit.to_csv(args.result_dir + 'disease_%s_table_fit_95%%CI.csv' % disease)
            table_ext.to_csv(args.result_dir + 'disease_%s_table_ext_95%%CI.csv' % disease)
    
        
# =============================================================================
#         ROC curve + calibration
# =============================================================================
        
    methods_names = {'logit_l1': 'logistic regression',
                     'DT': 'decision tree',
                     'RF': 'random forest',
                     'AdaBoost': 'AdaBoost',
                     'GBM': 'gradient boosting machine'}
    for d in tqdm(exp_dirs):
        serie = int(d.split('Serie_')[1].split('_Gap')[0])
        gap = int(d.split('Gap_')[1].split('/')[0])
        fig, ax = plt.subplots(figsize=(10,10), nrows=2, ncols=2, constrained_layout=True)
        ax[0,0].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Chance')
        ax[0,0].set_aspect('equal')
        ax[0,1].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Perfectly calibrated')
        ax[0,1].set_aspect('equal')
        ax[1,0].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Chance')
        ax[1,0].set_aspect('equal')
        ax[1,1].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Perfectly calibrated')
        ax[1,1].set_aspect('equal')
        color_main = {'GBM':'b','logit_l1':'g','RF':'r'}
        color_2 = {'GBM':'lightblue','logit_l1':'lightgreen','RF':'pink'}
        for mtd in ['GBM','logit_l1']:
            # fit test
            tprs_fit = []
            fprs_fit = []
            auc_fit = []
            calibration_slope_fit = []
            # external
            tprs_ext = []
            fprs_ext = []
            auc_ext = []
            calibration_slope_ext = []
            y_test_fit, y_pred_proba_fit = [], []
            y_test_ext, y_pred_proba_ext = [], []
            for seed in seeds:
                # fit test
                y_test = y[(serie, gap)][seed][mtd]['fit']['true']
                y_pred_proba = y[(serie, gap)][seed][mtd]['fit']['pred_proba']
                ros = RandomOverSampler(random_state=0)
                y_pred_proba_resampled, y_test_resampled = ros.fit_resample(y_pred_proba.reshape(-1,1), y_test)
                y_test, y_pred_proba = y_test_resampled, y_pred_proba_resampled.reshape(-1)
                
                y_test_fit += list(y_test)
                y_pred_proba_fit += list(y_pred_proba)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                tprs_fit += list(tpr)
                fprs_fit += list(fpr)
                auc_fit.append(auc(fpr, tpr))
                ax[0,0].plot(fpr, tpr, color=color_2[mtd], alpha=0.5, lw=1)#, label='Random state %d' % seed)
                # calibration
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=nbins)
                ax[0,1].plot(mean_predicted_value, fraction_of_positives, color=color_2[mtd], alpha=0.5, lw=1)
                # The calibration slope, which is calculated by regressing the observed outcome on the predicted probabilities.
                reg = LinearRegression(fit_intercept=False).fit(X=mean_predicted_value.reshape(-1,1), y=fraction_of_positives.reshape(-1,1))
                calibration_slope_fit.append(reg.coef_[0][0])
                
                # external
                y_test = y[(serie, gap)][seed][mtd]['ext']['true']
                y_pred_proba = y[(serie, gap)][seed][mtd]['ext']['pred_proba']
                ros = RandomOverSampler(random_state=0)
                y_pred_proba_resampled, y_test_resampled = ros.fit_resample(y_pred_proba.reshape(-1,1), y_test)
                y_test, y_pred_proba = y_test_resampled, y_pred_proba_resampled.reshape(-1)
                
                y_test_ext += list(y_test)
                y_pred_proba_ext += list(y_pred_proba)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                tprs_ext += list(tpr)
                fprs_ext += list(fpr)
                auc_ext.append(auc(fpr, tpr))
                ax[1,0].plot(fpr, tpr, color=color_2[mtd], alpha=0.5, lw=1)#, label='Random state %d' % seed)
                # calibration
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba.reshape(-1), n_bins=nbins)
                ax[1,1].plot(mean_predicted_value, fraction_of_positives, color=color_2[mtd], alpha=0.5, lw=1)
                # The calibration slope, which is calculated by regressing the observed outcome on the predicted probabilities.
                reg = LinearRegression(fit_intercept=False).fit(X=mean_predicted_value.reshape(-1,1), y=fraction_of_positives.reshape(-1,1))
                calibration_slope_ext.append(reg.coef_[0][0])
                
            mean, std = mean_confidence_interval(auc_fit)
            ax[0,0].plot(np.sort(fprs_fit), np.sort(tprs_fit), color=color_main[mtd], alpha=1, lw=2, \
                         label='%s\n= %0.3f $\pm$ %0.4f' % (methods_names[mtd], mean, std))
            # calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test_fit, y_pred_proba_fit, n_bins=nbins)
            # The calibration slope, which is calculated by regressing the observed outcome on the predicted probabilities.
            mean, std = mean_confidence_interval(calibration_slope_fit)
            ax[0,1].plot(mean_predicted_value, fraction_of_positives, color=color_main[mtd], alpha=1, lw=2, \
                         label='%s\n= %0.3f $\pm$ %0.4f' % (methods_names[mtd], mean, std))
            
            mean, std = mean_confidence_interval(auc_ext)
            ax[1,0].plot(np.sort(fprs_ext), np.sort(tprs_ext), color=color_main[mtd], alpha=1, lw=2, \
                         label='%s\n= %0.3f $\pm$ %0.4f' % (methods_names[mtd], mean, std))
            # calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test_ext, y_pred_proba_ext, n_bins=nbins)
            # The calibration slope, which is calculated by regressing the observed outcome on the predicted probabilities.
            mean, std = mean_confidence_interval(calibration_slope_ext)
            ax[1,1].plot(mean_predicted_value, fraction_of_positives, color=color_main[mtd], alpha=1, lw=2, \
                         label='%s\n= %0.3f $\pm$ %0.4f' % (methods_names[mtd], mean, std))
            
            ax[0,0].set(xlim=[0,1], ylim=[0,1])
            ax[1,0].set(xlim=[0,1], ylim=[0,1])
            ax[0,1].set(xlim=[0,1], ylim=[0,1])
            ax[1,1].set(xlim=[0,1], ylim=[0,1])
            
            ax[0,0].set_title('(A) EICU testing set: AUC', loc = 'left')
            ax[0,1].set_title('(B) EICU testing set: calibration plot', loc = 'left')
            ax[1,0].set_title('(C) MIMIC validation set: AUC', loc = 'left')
            ax[1,1].set_title('(D) MIMIC validation set: calibration plot', loc = 'left')
            
            ax[0,0].legend(loc="lower right")
            ax[1,0].legend(loc="lower right")
            ax[0,1].legend(loc="upper left")
            ax[1,1].legend(loc="upper left")
            
            ax[0,0].set_xlabel("1-Specificity")
            ax[1,0].set_xlabel("1-Specificity")
            ax[0,0].set_ylabel("Sensitivity")
            ax[1,0].set_ylabel("Sensitivity")
            
            ax[0,1].set_xlabel("Mean predicted value")
            ax[1,1].set_xlabel("Mean predicted value")
            ax[0,1].set_ylabel("Fraction of positives")
            ax[1,1].set_ylabel("Fraction of positives")
        fig.suptitle('Performances in AKI-%d with Model %d-%d' % (args.stage, serie, gap), size=16)
        fig.savefig(d+'AUC.pdf', dpi = 300)
        
        
        
        
