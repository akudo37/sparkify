#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics calculation of classification model and plot of precision-recall curve
"""
# import libraries
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.sql.functions import col, udf, asc, desc, min, max, count, avg, stddev_pop, countDistinct, last, first, when
from pyspark.sql.functions import isnan, lit, greatest, round
from pyspark.sql.functions import sum as Fsum
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, StandardScaler, SQLTransformer

import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as Fauc
from sklearn.metrics import average_precision_score


# prepare function for metric calculation
def calculate_classification_metrics(model_name, df_test_for_model, output=True):
    '''
    INPUT:
    model_name - (string) classification model name
    df_test_for_model - (pyspark dataframe) transformed test dataframe including prediction and label
    output - (bool) whether to print metrics to stdout
    
    OUTPUT:
    metrics - (dictionary) dictionary storing TP, TN, FP, FN, Precision, Recall, and F1
    
    DESCRIPTION:
    Print out and return TP, TN, FP, FN, Precision, Recall and F1
    '''
    # Count True Positive, True Negative, False Positive, False Negative in test data result
    sqlTrans = SQLTransformer(statement=" \
        SELECT \
            SUM(CASE WHEN label = 1 AND prediction = 1 THEN 1 ELSE 0 END) AS TP, \
            SUM(CASE WHEN label = 0 AND prediction = 0 THEN 1 ELSE 0 END) AS TN, \
            SUM(CASE WHEN label = 0 AND prediction = 1 THEN 1 ELSE 0 END) AS FP, \
            SUM(CASE WHEN label = 1 AND prediction = 0 THEN 1 ELSE 0 END) AS FN \
            FROM __THIS__")

    counts = sqlTrans.transform(df_test_for_model).collect()

    # calculate precision, recall and f1 score by definition
    TP, TN, FP, FN = counts[0].TP, counts[0].TN, counts[0].FP, counts[0].FN
    if (TP + FP) > 0:
        Precision = TP / (TP + FP)
    else:
        Precision = 0
        print('[INFO: TP + FP = 0, and Precision is set 0.]')
        
    if (TP + FN) > 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0
        print('[INFO: TP + FN = 0, and Recall is set 0.]')
        
    if (Recall + Precision) > 0:
        F1_score = 2 * Recall * Precision / (Recall + Precision)
    else:
        F1_score = 0
        print('[INFO: Recall + Precision = 0, and F1 is set 0.]')

    if output:
        print(model_name)
        print('precision:{:.4f}, recall:{:.4f}, f1:{:.4f}'.format(Precision, Recall, F1_score))
        print('(TP:{}, TN:{}, FP:{}, FN:{})'.format(TP, TN, FP, FN))
    
    metrics = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN, 'Precision':Precision, 'Recall':Recall, 'F1':F1_score}
    return metrics


# Plot precision-recall curve and output F1, AUC, and AP
def plot_precision_recall_curve(df, title_addition=None, ax=None):
    '''
    INPUT:
    df - (pyspark dataframe) dataset transformed by model, including 'label', 'probability' and 'prediction' columns
    title_addition - (str) additional text string to chart title
    ax - (object) Axes object or array of Axes objects.
    
    DESCRIPTION:
    Plot precision-recall curve from transformed dataset.
    Also output F1 score, AUC (area under curve), and AP (average precision)
    In case ax is privided, plt.show() will not be called inside this function. (plt:matplotlib.pyplot)
    '''
    # change to pandas dataframe
    label_proba = df.select('label', 'probability', 'prediction').toPandas()
    
    # extract probability for 1
    label_proba['proba'] = label_proba['probability'].apply(lambda x: x[1])
    
    testy = label_proba['label']
    probs = label_proba['proba']
    yhat = label_proba['prediction']

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(testy, probs)

    # calculate F1 score
    f1 = f1_score(testy, yhat)

    # calculate precision-recall AUC
    auc = Fauc(recall, precision)  # Fauc is an alias of sklearn.metrics auc

    # calculate average precision score
    ap = average_precision_score(testy, probs)

    print('F1={:.4f}, AUC={:.4f}, AP={:.4f} {}'.format(f1, auc, ap, title_addition))

    if ax is None:
        # plot no skill
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')

        # plot the roc curve for the model
        plt.plot(recall, precision, marker='.')
        if title_addition is not None:
            plt.title(title_addition)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.ylim(-0.01, 1.05)
        
        # show the plot
        plt.show()
    else:
        # plot no skill
        ax.plot([0, 1], [0.5, 0.5], linestyle='--')

        # plot the roc curve for the model
        ax.plot(recall, precision, marker='.')
        if title_addition is not None:
            ax.set_title(title_addition)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_ylim(-0.01, 1.05)
        