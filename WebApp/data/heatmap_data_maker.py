from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, last
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml.pipeline import Pipeline

import argparse
import pickle
import pandas as pd

# own script
from etl_pipeline import add_data_cleaner, add_label_maker


def main():
    '''
    INPUT:
    data_path - (string) filepath of event log JSON file
    id_list_path = (string) filepath of user ID CSV file
    heatmap_save_path - (string) filepath of pickle file to save heatmap data
        and event name list to

    DESCRIPTION:
    Make heatmap data for visualization into pandas dataframe and save it into
        a pickle file.
    Event name list is saved together with the heatmap data as dictionary.
    Dictionary key to access heatmap data is 'heatmap', and key for event name
        is 'labels'.
    SparkSession creation line needs to be manually modified depending on Spark
        environment.
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="filepath of event log JSON file")
    parser.add_argument("id_path", help="filepath of user ID CSV file")
    parser.add_argument("heatmap_save_path",
                        help="filepath of pickle file to save heatmap data \
                        and event name list to")

    args = parser.parse_args()

    # create a Spark session (in case of local workspace)
    '''please modify for actual Spark environment'''
    spark = SparkSession \
        .builder \
        .appName("Sparkify") \
        .master("local") \
        .getOrCreate()

    # load data set
    print('Loading data...\n    EVENT LOG: {}'.format(args.data_path))
    data = spark.read.json(args.data_path)

    # load user ID
    print('Loading id list...\n    USER IDS: {}'.format(args.id_path))
    userIdList = pd.read_csv(args.id_path)
    del userIdList['Unnamed: 0']
    userIds = userIdList['userId'].values.tolist()

    print('Transforming to heatmap data...')
    # create SQL tables of 'data''
    data.createOrReplaceTempView('t_data')
    # extract test set users from data
    test_data = spark.sql(" \
        SELECT * \
        FROM t_data \
        WHERE userId IN ({})".format(', '.join([str(i) for i in userIds])))

    # extract last n_last_events of each user
    n_last_events = 0  #1000
    if n_last_events > 0:  # if 0, take all data
        user_descTs_follow_n_window = Window \
            .partitionBy('userId') \
            .orderBy(desc('ts')) \
            .rowsBetween(Window.currentRow, n_last_events)

        test_data = test_data.withColumn('beforeNeventsTS', last('ts') \
            .over(user_descTs_follow_n_window))

        test_data = test_data.filter(col('ts') >= col('beforeNeventsTS'))

    # clean and define label (no feature engineering yet)
    stages = add_data_cleaner()
    stages = add_label_maker(stages)
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(test_data)
    df = model.transform(test_data)

    # index page colum
    indexer = StringIndexer(inputCol='page', outputCol='event')
    indexerModel = indexer.fit(df)
    df_event = indexerModel.transform(df)
    labels = indexerModel.labels

    # change ts into second unit
    df_event = df_event.withColumn('ts', col('ts') / 1000)

    # select columns and save as pandas dataframe
    df_event = df_event.select('ts', 'userId', 'event')
    pd_event = df_event.toPandas()
    print('Saving heatmap data...\n    HEATMAP: {}'.format(
            args.heatmap_save_path))
    package = {'heatmap': pd_event, 'labels': labels}
    with open(args.heatmap_save_path, 'wb') as f:
        pickle.dump(package, f)
    print('Heatmap data is saved!')


if __name__ == '__main__':
    main()
