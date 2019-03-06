#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL pipeline that cleans and transforms data, and stores in JSON file
"""
# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.sql.functions import col, udf, asc, desc, min, max, count, avg, stddev_pop, countDistinct, last, first, when
from pyspark.sql.functions import isnan, lit, greatest, round
from pyspark.sql.functions import sum as Fsum
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, StandardScaler, SQLTransformer
from pyspark.ml.pipeline import Pipeline, PipelineModel

import numpy as np
import datetime

import argparse

# Data Cleaner (subroutine)
def add_data_cleaner():
    '''
    OUTPUT:
    stages - (list) list of transformer to be used as 'stages' argument of pyspark Pipeline() constructor
    
    DESCRIPTION:
    This is a subroutine of create_preprocess_pipeline() function.
    Stages added by this function will clean raw pyspark dataframe for next steps.
    '''
    stages = []  # pipeline stage list
    
    # filter rows with userId==Null or sessionId==Null, just in case
    sqlTrans = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE userId IS NOT NULL AND sessionId IS NOT NULL")
    stages.append(sqlTrans)

    # drop empty user id row
    sqlTrans = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE userId != ''")
    stages.append(sqlTrans)

    # drop 'Logged-Out' state and 'Guest' state
    sqlTrans = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE auth != 'Logged Out' AND auth != 'Guest'")
    stages.append(sqlTrans)

    # exclude rows with user who has only one song play or less
    sqlTrans = SQLTransformer(statement=" \
        SELECT * \
        FROM __THIS__ \
        WHERE userId NOT IN ( \
            SELECT DISTINCT userId \
            FROM \
            (SELECT userId, page, \
                COUNT(CASE WHEN page = 'NextSong' THEN page END) \
                OVER(PARTITION BY userId) AS songCount \
            FROM __THIS__) AS user_page_count \
            WHERE user_page_count.songCount < 2)")
    stages.append(sqlTrans)

    return stages


# Label Maker (subroutine)
def add_label_maker(stages):
    '''
    INPUT:
    stages - (list) list of transformer to be used as 'stages' argument of pyspark Pipeline() constructor
                It should be an output of 'create_data_cleaner()' function.
    
    OUTPUT:
    stages - (list) list of transformer to be used as 'stages' argument of pyspark Pipeline() constructor
    
    DESCRIPTION:
    This is a subroutine of create_preprocess_pipeline() function.
    Stages added by this function will make label column in target pyspark dataframe.
    It also drops rows which the label column directly depends on.
    '''
    # 'churn_event'
    # add a column to store churn event as integer
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            CASE WHEN page = 'Cancellation Confirmation' THEN 1 ELSE 0 END AS churn_event \
        FROM __THIS__")
    stages.append(sqlTrans)

    # 'Churn'
    # add a column to store cumulative sum of churn flag
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            MAX(churn_event) OVER ( \
                PARTITION BY userId \
            ) AS Churn \
        FROM __THIS__")
    stages.append(sqlTrans)

    return stages


# Feature Creater (subroutine)
def add_features_maker(stages):
    '''
    INPUT:
    stages - (list) list of transformer to be used as 'stages' argument of pyspark Pipeline() constructor
                It must be an output of 'create_label_maker()' function.
    
    OUTPUT:
    stages - (list) list of transformer to be used as 'stages' argument of pyspark Pipeline() constructor
    feature_labels - (list) list of feature column names for utility
    
    DESCRIPTION:
    This is a subroutine of create_preprocess_pipeline() function.
    Stages added by this function will make feature columns in target pyspark dataframe.
    '''
    # 'event_name'
    # replace whitespace of page column with underbar and put into a new column
    sqlTrans = SQLTransformer(statement=" \
        SELECT userId, Churn AS label, ts, registration, level, event_name \
        FROM ( \
            SELECT *, REPLACE(page, ' ', '_') AS event_name \
            FROM __THIS__)")
    
    stages.append(sqlTrans)

    # 'event_name' elements
    event_names = [
         'About',
         'Add_Friend',
         'Add_to_Playlist',
         #'Cancel',
         #'Cancellation_Confirmation',
         'Downgrade',
         'Error',
         'Help',
         'Home',
         'Logout',
         'NextSong',
         'Roll_Advert',
         'Save_Settings',
         'Settings',
         'Submit_Downgrade',
         'Submit_Upgrade',
         'Thumbs_Down',
         'Thumbs_Up',
         'Upgrade']
    
    # 'eventInterval'
    # add a column to store event intervals (in seconds)
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            ((FIRST_VALUE(ts) OVER ( \
                PARTITION BY userId, event_name \
                ORDER BY ts DESC \
                ROWS BETWEEN 1 PRECEDING AND CURRENT ROW \
            ) / 1000) - (LAST_VALUE(ts) OVER ( \
                PARTITION BY userId, event_name \
                ORDER BY ts DESC \
                ROWS BETWEEN 1 PRECEDING AND CURRENT ROW \
            ) / 1000)) AS eventInterval \
        FROM __THIS__")
    
    stages.append(sqlTrans)

    # 'lastTS'
    # add a column to store the last TS for each user
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            (FIRST_VALUE(ts) OVER ( \
                PARTITION BY userId, event_name \
                ORDER BY ts DESC \
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW \
            )) AS lastTS \
        FROM __THIS__")
    
    stages.append(sqlTrans)

    # 'trueInterval'
    # set the last TS row's interval value to Null
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            CASE WHEN ts == lastTS THEN NULL ELSE eventInterval END AS trueInterval \
        FROM __THIS__")
    
    stages.append(sqlTrans)

    # 'trueInterval'(update), 'pageCount', 'paidCount', 'songCount'
    # group by userId and page
    # we get average of interval for NextSong, and count for other events
    # we also count paid songs, and total songs
    sqlTrans = SQLTransformer(statement=" \
        SELECT label, userId, event_name, \
            AVG(trueInterval) AS trueInterval, \
            COUNT(event_name) AS pageCount, \
            COUNT(CASE WHEN event_name = 'NextSong' AND level = 'paid' THEN event_name END) AS paidCount, \
            COUNT(CASE WHEN event_name = 'NextSong' THEN event_name END) AS songCount \
        FROM __THIS__ \
        GROUP BY label, userId, event_name")
    
    stages.append(sqlTrans)
    
    # 'songInterval'
    # add a column to store interval when page is NextSong
    sqlTrans = SQLTransformer(statement=" \
        SELECT *, \
            CASE WHEN event_name == 'NextSong' THEN trueInterval END AS songInterval \
        FROM __THIS__")
    
    stages.append(sqlTrans)
    
    # 'songInterval'(update), 'paidRatio', element of event_names list as new columns
    # group by userId, average song intervals, and count other events and vidide the sum by songCount

    # loop event names to create sql lines and concatenate them
    sql_line = ''.join(['(COUNT(CASE WHEN event_name == "{}" THEN pageCount END) / SUM(songCount)) AS {},' \
                        .format(name, name) for name in event_names])[:-1]
    
    sqlTrans = SQLTransformer(statement=" \
        SELECT label, userId, \
            MAX(songInterval) AS songInterval, \
            (MAX(paidCount) / MAX(songCount)) AS paidRatio, \
            {} \
        FROM __THIS__ \
        GROUP BY label, userId".format(sql_line))

    stages.append(sqlTrans)
    
    # 'featureVec'
    # assemble feature columns into a vector column
    event_names.remove('NextSong')
    feature_columns = ['songInterval', 'paidRatio'] + event_names
    
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='featureVec')
    
    stages.append(assembler)

    # store feature labels for utility
    feature_labels = assembler.getInputCols()

    return stages, feature_labels


# Creating Preprocessing Pipeline (Data Cleaning + Feature Engineering)
def create_preprocess_pipeline():
    '''
    OUTPUT:
    preprocess_pipeline - (pyspark Pipeline object)
    feature_labels - (list) string labels for corresponding feature vector elements
    
    DESCRIPTION:
    This function creates a pipeline for data cleaning and preprocessing.
    It must be first 'fit' with data to create a pipeline model.
    Then the model can 'transform' the data.
    
    Example:
    > preprocess_pipeline, feature_labels = create_preprocess_pipeline()
    > preprocess_model = preprocess_pipeline.fit(data)
    > data = preprocess_model.transform(data) 
    '''
    # clean data
    stages = add_data_cleaner()

    # make label
    stages = add_label_maker(stages)

    # make features
    stages, feature_labels = add_features_maker(stages)
    
    # select only necessary columns, 'userId', 'label', 'features'
    sqlTrans = SQLTransformer(statement=" \
        SELECT userId, label, featureVec AS features \
        FROM __THIS__")
    stages.append(sqlTrans)

    # create preprocessing pipeline
    preprocess_pipeline = Pipeline(stages=stages)
    
    return preprocess_pipeline, feature_labels    


def main():    
    '''
    INPUT:
    data_path - (string, mandatory) filepath of event log JSON file
    save_path - (string, mandatory) filepath of parquet file to save cleaned and transformed data to
    pipeline_save_path - (string, optional) folderpath of pipeline to save fit pipeline to
    pipeline_load_path - (string, optional) folderpath of pipeline to load fit pipeline from
        *Note: pipeline_save_path and pipeline_load_path cannot be set at the same time
        
    DESCRIPTION:
    ETL pipeline sandalone mode main routine. 
    It cleans and transforms data, and stores in JSON file.
    SparkSession creation line needs to be manually modified depending on Spark environment.    
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="filepath of event log JSON file")
    parser.add_argument("save_path", help="filepath of parquet file to save cleaned and transformed data to")
    
    # optional: either save or load fit pipeline
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-ps", "--pipeline_save_path", type=str, help="folderpath of pipeline to save fit pipeline to")
    group.add_argument("-pl", "--pipeline_load_path", type=str, help="folderpath of pipeline to load fit pipeline from")
        
    args = parser.parse_args()
    
    # create a Spark session (in case of local workspace) 
    ##### please modify for actual Spark environment #####
    spark = SparkSession \
        .builder \
        .appName("Sparkify") \
        .master("local") \
        .getOrCreate()
    ######################################################

    # load data subset
    print('Loading data...\n    EVENT LOG: {}'.format(args.data_path))
    data = spark.read.json(args.data_path)

    # clean and transform
    print('Cleaning and transforming data...')
    if args.pipeline_load_path:
        # load preprocess pipeline from folder and transform data
        prepro_model = PipelineModel.load(args.pipeline_load_path)
        processed_data = prepro_model.transform(data)
    else:
        # or create preprocessing pipeline, and fit and transform
        preprocess_pipeline, feature_labels = create_preprocess_pipeline()
        preprocess_model = preprocess_pipeline.fit(data)
        processed_data = preprocess_model.transform(data)

    # save data to a file
    print('Saving data...\n    DATASET: {}'.format(args.save_path))
    processed_data.write.mode('overwrite').save(args.save_path)
    print('Cleaned and transformed data saved to parquet file!')

    if args.pipeline_save_path:
        # save fit pipeline to a folder
        preprocess_model.write().overwrite().save(args.pipeline_save_path)
        print('Preprocess pipeline saved to a folder!')


if __name__ == '__main__':
    main()