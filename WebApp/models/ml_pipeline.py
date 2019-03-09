#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML pipeline that trains classifier and saves
"""
# import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# own scripts
from classification_metrics import calculate_classification_metrics

import argparse


def build_model():
    '''
    OUTPUT:
    gbt_cv - classification pipeline

    DESCRIPTION:
    Build machine learning model.
    Standard scale features, and then classify with GBT Classifier.
    Grid search is applied with three folds cross validation.
    '''
    # Standard scaling for feature vector
    stdScaler = StandardScaler(withMean=True, withStd=True,
                               inputCol='features', outputCol='scaledFeatures')

    # Gradient Boosted-Tree Classifier (training with default parameters)
    gbt = GBTClassifier(featuresCol='scaledFeatures', labelCol='label')

    # pipeline with standard scaler
    gbt_pipeline = Pipeline(stages=[stdScaler, gbt])

    # grid search parameters
    gbt_grid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [10, 20, 50]) \
        .build()

    # evaluator
    # precision-recall-curve
    gbt_evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')

    # grid search with 3 folds cross validation
    gbt_cv = CrossValidator(estimator=gbt_pipeline,
                            estimatorParamMaps=gbt_grid,
                            evaluator=gbt_evaluator,
                            numFolds=3)

    return gbt_cv


def main():
    '''
    INPUT:
    data_path - (string) filepath of parquet file to load cleaned and
        transformed data from
    model_save_path - (string) folderpath of classifier to save trained
        classifier to

    DESCRIPTION:
    ML pipeline standalone mode main routine. It trains classifier and saves.
    Loaded data will be split into train and test sets.
    Metrics will be calculated for both train set and test set.
    SparkSession creation line needs to be manually modified depending on
        Spark environment.
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="filepath of parquet file to load \
                        cleaned and transformed data from")
    parser.add_argument("model_save_path", help="folderpath of classifier to \
                        save trained classifier to")

    args = parser.parse_args()

    # create a Spark session (in case of local workspace)
    '''please modify for actual Spark environment'''
    spark = SparkSession \
        .builder \
        .appName("Sparkify") \
        .master("local") \
        .getOrCreate()

    # load processed data
    print('Loading data...\n    DATASET: {}'.format(args.data_path))
    data = spark.read.load(args.data_path)

    # train, evaluate and save model
    df_train, df_test = data.randomSplit([0.7, 0.3], seed=24)

    print('Building model...')
    cv = build_model()

    print('Training model...')
    model = cv.fit(df_train)

    print('Evaluating model...')
    _ = calculate_classification_metrics('GBT Classifier (train set)',
                                         model.transform(df_train),
                                         output=True)
    _ = calculate_classification_metrics('GBT Classifier (test set)',
                                         model.transform(df_test),
                                         output=True)

    print('Saving model...\n    MODEL: {}'.format(args.model_save_path))
    model.bestModel.write().overwrite().save(args.model_save_path)
    print('Trained model saved!')


if __name__ == '__main__':
    main()
