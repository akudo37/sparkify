# import libraries
from app import app
import json
import plotly
from flask import render_template, request
from app.wrangle_data import return_heatmap, return_prediction

# global variables
figures = None
sample_id = None


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    global figures, sample_id

    if request.method == 'POST' and 'predict' in request.form and sample_id\
        is not None:

        actuals, probas, preds = return_prediction(sample_id)
        judges = [actual == pred for (actual, pred) in zip(actuals, preds)]
        actuals = [(lambda x: 'Churned' if x == 1 else 'In-Service'
                    )(x) for x in actuals]
        preds = [(lambda x: 'Churn' if x == 1 else 'Stay')(x) for x in preds]
        table_data = zip(sample_id, actuals, probas, preds, judges)

        button_for_new_sample = True
    else:  # GET or 'sample' case
        figures, sample_id = return_heatmap(n_sample=5)
        sample_id = sample_id[::-1]

        # empty table
        actuals = [''] * len(sample_id)
        probas = [''] * len(sample_id)
        preds = [''] * len(sample_id)
        judges = [-1] * len(sample_id)
        table_data = zip(sample_id, actuals, probas, preds, judges)

        button_for_new_sample = False

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           button_for_new_sample=button_for_new_sample,
                           table_data=table_data,
                           figuresJSON=figuresJSON)
