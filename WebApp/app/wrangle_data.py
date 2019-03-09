# import libraries
import numpy as np
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.pipeline import PipelineModel

import pickle


# create a Spark session (in case of local workspace)
'''please modify for actual Spark environment'''
spark = SparkSession \
    .builder \
    .appName("Sparkify") \
    .master("local") \
    .getOrCreate()

# load heatmap data and labels from pickle file
heatmap_data_path = './data/micro_sparkify_heatmap_full.pickle'
with open(heatmap_data_path, 'rb') as f:
    package = pickle.load(f)
pd_event = package['heatmap']
labels = package['labels']

# load processed data
feature_data_path = './data/mini_sparkify_features.parquet'
print('Loading data...\n    DATASET: {}'.format(feature_data_path))
feature_data = spark.read.load(feature_data_path)

# load trained Gradient Boosted-Tree Classifier model from folder
model_path = './models/webGbtModel'
print('Loading model...\n    MODEL: {}'.format(model_path))
classifierModel = PipelineModel.load(model_path)


def return_prediction(sample_id):
    '''
    INPUT:
    sample_id - (list) list of sampled userId

    OUTPUT:
    actuals - (list) list of actual churn status (0 or 1)
    probas - (list) list of churn probabilities
    preds - (list) list of churn prediction (0 or 1)

    DESCRIPTION:
    Make prediction on sampled data and return labels, probabilities and
    predictions.
    '''
    global spark, pd_event, labels, feature_data, classifiedModel

    # extract subset of data for sample id
    print('Extracting samples...')
    sample_data = feature_data.filter(col('userId').isin(sample_id))

    # transform with classification pipeline (churn prediction)
    print('Classifying data...')
    classifiedData = classifierModel.transform(sample_data)
    pd_classified = classifiedData.select('userId', 'label', 'probability',
                                          'prediction').toPandas()
    print('Data classified!')

    # sort by userId (as integer)
    pd_classified['int_id'] = pd_classified['userId'].apply(lambda x: int(x))
    pd_classified = pd_classified.sort_values('int_id')
    pd_classified = pd_classified.drop('int_id', axis=1)

    actuals = [x for x in pd_classified['label']]
    probas = [round(x[1], 3) for x in pd_classified['probability']]
    preds = [x for x in pd_classified['prediction']]

    return actuals, probas, preds


def return_heatmap(n_sample=10):
    '''
    INPUT:
    heatmap_data_path - (string) filapath to pickle file to load heatmap data
        and event labels from
    n_sample - (int) number of users to sample for heatmap

    OUTPUT:
    figures - (list) list containing plotly heatmap visualization
    y - (list) list of sampled userId

    DESCRIPTION:
    Creates plotly heatmap visualization.
    '''
    global spark, pd_event, labels, feature_data, classifiedModel

    # sort data by timestamp and change timestamp to integer
    pd_event = pd_event.sort_values('ts')
    pd_event['ts'] = pd_event['ts'].astype(int)

    # sample n users
    user_list = pd_event['userId'].unique()
    np.random.shuffle(user_list)
    sample_id = user_list[:n_sample]
    pd_event_sample = pd_event[pd_event['userId'].isin(sample_id)]

    # change to userId x ts table (event index as matrix values)
    pd_heatmap = pd_event_sample.groupby('ts').max()
    pd_heatmap = pd_heatmap.pivot(columns='userId', values='event').transpose()

    # sort by userId (as integer)
    pd_heatmap = pd_heatmap.reset_index()
    pd_heatmap['int_id'] = pd_heatmap['userId'].apply(lambda x: int(x))
    pd_heatmap = pd_heatmap.sort_values('int_id', ascending=False)
    pd_heatmap = pd_heatmap.drop('int_id', axis=1)
    pd_heatmap = pd_heatmap.set_index('userId')

    # create heatmap data
    x = pd_heatmap.columns.values.tolist()
    y = pd_heatmap.index.values.tolist()
    z = pd_heatmap.values.tolist()

    # create heatmap hovertext
    text = pd_heatmap.applymap(lambda x: labels[int(x)] if x >= 0 else x)

    # create heatmap colorscale
    # refer to 'tab20' colormap of matplotlib
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_cm.py
    _tab20_data = (
        (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
        (0.6823529411764706,  0.7803921568627451,  0.9098039215686274  ),  # aec7e8
        (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
        (1.0,                 0.7333333333333333,  0.47058823529411764 ),  # ffbb78
        (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
        (0.596078431372549,   0.8745098039215686,  0.5411764705882353  ),  # 98df8a
        (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
        (1.0,                 0.596078431372549,   0.5882352941176471  ),  # ff9896
        (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
        (0.7725490196078432,  0.6901960784313725,  0.8352941176470589  ),  # c5b0d5
        (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b
        (0.7686274509803922,  0.611764705882353,   0.5803921568627451  ),  # c49c94
        (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2
        (0.9686274509803922,  0.7137254901960784,  0.8235294117647058  ),  # f7b6d2
        (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f
        (0.7803921568627451,  0.7803921568627451,  0.7803921568627451  ),  # c7c7c7
        (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22
        (0.8588235294117647,  0.8588235294117647,  0.5529411764705883  ),  # dbdb8d
        (0.09019607843137255, 0.7450980392156863,  0.8117647058823529  ),  # 17becf
        (0.6196078431372549,  0.8549019607843137,  0.8980392156862745),    # 9edae5
    )

    # compose to plotly colorspace format (up to 19 colors)
    _tab19_rgb = ['rgb({}, {}, {})'.format(int(255 * r),
                                           int(255 * g),
                                           int(255 * b)
                                           ) for (r, g, b) in _tab20_data[:-1]]

    _tab19_colorscale = []
    for level, rgb in zip(np.arange(0, 1+1/19, 1/19), _tab19_rgb):
        _tab19_colorscale.append([round(level, 3), rgb])
        _tab19_colorscale.append([round(level + 1/19, 3), rgb])

    # create heatmap
    graph_one = [
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            zmin=0,
            zmax=19,
            text=text,
            hoverinfo='text',
            colorscale=_tab19_colorscale,
            colorbar=dict(tickmode='array',
                          tickvals=np.arange(0.5, 19, 1),
                          ticktext=labels,
                          ticks='outside',
                          tickfont=dict(size=10))
        )
    ]

    # heatmap layout
    layout_one = dict(title='Sparkify last 1000 user event patterns in time series',
                      xaxis=dict(title='Timestamp (second)',
                                 ticks='',
                                 type='category',
                                 tickfont=dict(size=8)),
                      yaxis=dict(title='User ID',
                                 ticks='',
                                 type='category'
                                 ),
                      margin=dict(b=80, t=50))

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))

    return figures, y
