from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.pipeline import Pipeline

import argparse
import pickle

# own script
from etl_pipeline import add_data_cleaner, add_label_maker


def main():
    '''
    INPUT:
    data_path - (string, mandatory) filepath of event log JSON file
    heatmap_save_path - (string, mandatory) filepath of pickle file to save
        heatmap data and to

    DESCRIPTION:
    Make heatmap data for visualization into pandas dataframe and save it into
        a pickle file.
    Event name list is saved together with the heatmap data as dictionary.
    Dictionary key to access heatmap data is 'heatmap', and key for event name
        is 'labels'.
    (Actual visualization should sample limited number of userId from it.)
    SparkSession creation line needs to be manually modified depending on Spark
        environment.
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="filepath of event log JSON file")
    parser.add_argument("heatmap_save_path",
                        help="filepath of pickle file to save heatmap data to")

    args = parser.parse_args()

    # create a Spark session (in case of local workspace)
    '''please modify for actual Spark environment'''
    spark = SparkSession \
        .builder \
        .appName("Sparkify") \
        .master("local") \
        .getOrCreate()

    # load data subset
    print('Loading data...\n    EVENT LOG: {}'.format(args.data_path))
    data = spark.read.json(args.data_path)

    print('Transforming to heatmap data...')
    # clean and define label (no feature engineering yet)
    stages = add_data_cleaner()
    stages = add_label_maker(stages)
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(data)
    df = model.transform(data)

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
