import flask
import os
import pandas as pd
import numpy as np
import re
from webapp.forms import VisualizeRouteForm, VisualizeStopForm, VisualizeStopRouteForm, PredictionRouteForm, \
    PredictionStopForm, PredictionStopRouteForm, RetryForm
from werkzeug.contrib.cache import SimpleCache
from werkzeug.utils import secure_filename
from project.lstm_regression import LSTM_REG, run
from project.multivariate_lstm_regresson import run_multivariate_lstm, MV_LSTM
from project.data_preprocessing import read_traffic_csv, read_text_file,\
    aggregate_dataframe, cast_time_interval,\
    aggregate_double_attribute, aggregate_stop_attribute, aggregate_route_attribute, \
    get_statistics, date_slice_extent, sort_routes, heatmap_stop
from webapp.helper_function import create_prediction_list, set_predict_form, set_visualize_form, \
    check_predict_model
from datetime import timedelta

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = '33723372'
app.send_file_max_age_default = timedelta(seconds=1)
cache = SimpleCache()
PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}')
UPLOAD_PATH = os.path.normcase(os.path.join(os.path.dirname(__file__),'datasets'))
TIMESTEP = 96

"""
The route to the upload file page as the entrance
"""
@app.route('/', methods=['GET', 'POST'])
def upload():
    if flask.request.method == 'GET':
        return flask.render_template('upload.html')

    if flask.request.method == 'POST':
        # Read the CSV file by data preprocessing model
        upload_dataset = flask.request.files.get('file')
        cache.set('dataframe', read_traffic_csv(upload_dataset), timeout = 0)
        cache.set('start date', flask.request.form.get('date range start'), timeout = 0)
        cache.set('end date', flask.request.form.get('date range end'), timeout = 0)
        # Check path and save the file
        isExists = os.path.exists(UPLOAD_PATH)
        filename = secure_filename(upload_dataset.filename)
        if not isExists:
            os.makedirs(UPLOAD_PATH)
        upload_dataset.save(os.path.normcase(os.path.join(UPLOAD_PATH, filename)))
        cache.set('uploading', 'on')
        cache.set('route chart', 'off')
        cache.set('stop route chart', 'off')
        return flask.redirect( flask.url_for('visualize'))

"""
The route to the visualization page
"""
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    # get the input values from the flask form
    form_stop = VisualizeStopForm()
    form_route = VisualizeRouteForm()
    form_stop_route = VisualizeStopRouteForm()
    # get the visualization mode
    if flask.request.method == 'POST':
        if form_stop.submit_stop.data:
            cache.set('submit item', 'stop')
            if form_stop.validate_on_submit():
                cache.set('form_stop', form_stop.data)
        elif form_route.submit_route.data:
            cache.set('route chart', 'on')
            cache.set('submit item', 'route')
            if form_route.validate_on_submit():
                cache.set('form_route', form_route.data)
        elif form_stop_route.submit_stop_route.data:
            cache.set('stop route chart', 'on')
            cache.set('submit item', 'stop_route')
            if form_stop_route.validate_on_submit():
                cache.set('form_stop_route', form_stop_route.data)
    return flask.render_template('visualize.html', form_stop=form_stop, form_route=form_route,
                                 form_stop_route=form_stop_route)

"""
The route for asynchronous loading using JSON. It is to load the stop passenger numbers 
"""
@app.route("/stop_transfer", methods=["GET"])
def stop_transfer():
    if flask.request.method == "GET":
        if (cache.get('form_stop')['stop_id'] != None) & \
                (cache.get('form_stop')['stop_time_interval'] != None):
            # parameters contains the passenger number list, date list, and the stop name.
            parameters, start_date, end_date, timing = set_visualize_form(cache, 'stop')
            dataframe = cache.get('dataframe')
            stop_frame = aggregate_stop_attribute(dataframe, start_date, end_date, parameters['title'],
                                                  parameters['time_interval'], timing)
            parameters['time_interval'] = cast_time_interval(parameters['time_interval'])
            parameters['dates'] = stop_frame['Date'].values.tolist()
            parameters['list'] = stop_frame['Passengers'].values.tolist()
            # get the statistics for the passenger flow
            parameters = get_statistics(parameters)
            return flask.jsonify(parameters)

"""
The route for asynchronous loading using JSON. It is to load the route passenger numbers 
"""
@app.route("/route_transfer", methods=["GET"])
def route_transfer():
    if flask.request.method == "GET":
        if (cache.get('form_route')['route_id'] != None) & \
                (cache.get('form_route')['route_time_interval'] != None):
            parameters, start_date, end_date, timing = set_visualize_form(cache, 'route')
            switch = cache.get('route chart')
            if (switch == 'on'):
                dataframe = cache.get('dataframe')
                route_frame = aggregate_route_attribute(dataframe, start_date, end_date, parameters['title'],
                                                       parameters['time_interval'] , timing)
                parameters['time_interval'] = cast_time_interval(parameters['time_interval'])
                parameters['dates'] = route_frame['Date'].values.tolist()
                parameters['list'] = route_frame['Passengers'].values.tolist()
                parameters = get_statistics(parameters)
            return flask.jsonify(parameters)

"""
The route for asynchronous loading using JSON. It is to load the passenger numbers with stop and route constraints 
"""
@app.route("/stop_route_transfer", methods=["GET"])
def stop_route_transfer():
    if flask.request.method == "GET":
        parameters, start_date, end_date, timing = set_visualize_form(cache, 'stop_route')
        if (parameters['title_route'] != None) & (parameters['time_interval']!= None):
            if (cache.get('stop route chart') == 'on'):
                dataframe = cache.get('dataframe')
                stop_route_frame= aggregate_double_attribute(dataframe, start_date, end_date, parameters['title_stop'],
                                                             parameters['title_route'], parameters['time_interval'], timing)
                parameters['time_interval'] = cast_time_interval(parameters['time_interval'])
                parameters['dates'] = stop_route_frame['Date'].values.tolist()
                parameters['list'] = stop_route_frame['Passengers'].values.tolist()
                parameters = get_statistics(parameters)
        return flask.jsonify(parameters)

"""
The route for asynchronous loading using JSON. It is to load the visualize mode 
"""
@app.route("/submit_item", methods=["GET"])
def submit_item():
    parameters = {}
    if cache.get('submit item') == None:
        cache.set('submit item', 'None')
        parameters['submit item'] = cache.get('submit item')
    else:
        parameters['submit item'] = cache.get('submit item')
    return flask.jsonify(parameters)

"""
The route for asynchronous loading using JSON. It is to load the passenger overview 
"""
@app.route("/initialize", methods=["GET"])
def initialize():
    if flask.request.method == "GET":
        parameters = {}
        if cache.get('uploading') == 'on':
            parameters['title'] = "Traffic Flow Overview"
            dataframe = aggregate_dataframe(cache.get('dataframe'),
                                            cache.get('start date'), cache.get('end date'), 1440, 'Alighting')
            parameters['dates'] = dataframe['Date'].values.tolist()
            parameters['list'] = dataframe['Passengers'].values.tolist()
            parameters = get_statistics(parameters)
            cache.set('uploading', 'off')
        return flask.jsonify(parameters)

"""
The route for the prediction page
"""
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form_stop = PredictionStopForm()
    form_route = PredictionRouteForm()
    form_stop_route = PredictionStopRouteForm()
    if flask.request.method == 'GET':
        return flask.render_template('prediction.html', form_stop=form_stop, form_route=form_route,
                                     form_stop_route=form_stop_route)
    # get the prediction mode
    if flask.request.method == 'POST':
        if form_stop.submit_stop_predict.data:
            if form_stop.validate_on_submit():
                cache.set('predict model', 'stop')
                cache.set('pre_form_stop', form_stop.data)
        elif form_route.submit_route_predict.data:
            if form_route.validate_on_submit():
                cache.set('predict model', 'route')
                cache.set('pre_form_route', form_route.data)
        elif form_stop_route.submit_stop_route_predict.data:
            if form_stop_route.validate_on_submit():
                cache.set('predict model', 'stop route')
                cache.set('pre_form_stop_route', form_stop_route.data)
        return flask.redirect(flask.url_for('prediction_result'))

"""
The route for the prediction result page
"""
@app.route('/prediction_result', methods=['GET', 'POST'])
def prediction_result():
    form = RetryForm()
    if flask.request.method == 'GET':
        src = ''
        if cache.get('predict model') == 'stop': # predict stop passenger numbers
            dataframe, future_steps, proportion, start_date, \
            end_date, epoch, stop_id, timing = set_predict_form(cache, 'stop')
            destination = pd.read_csv('Destination.csv')
            stop = aggregate_stop_attribute(dataframe, start_date, end_date, stop_id, 1440/TIMESTEP, timing)
            dates = stop['Date'].values.tolist()
            set = np.array(stop['Passengers'].values)
            stop_pred = aggregate_stop_attribute(dataframe, cache.get('start date'), cache.get('end date'), stop_id, 1440/TIMESTEP, timing)
            extend_dates = stop_pred['Date'].values.tolist()
            set_pred = np.array(stop_pred['Passengers'].values)
            model = check_predict_model(cache, 'stop')
            if model == MV_LSTM:
                src, real_list, prediction_list, mape, rmse = \
                    run_multivariate_lstm(stop, epoch=epoch, proportion=proportion, pred_set=stop_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)
            else:
                src, real_list, prediction_list, mape, rmse = run(set, model_switch='off', model_type=model,
                                                                  epoch=epoch, proportion=proportion, step=future_steps,
                                                                  pred_set=set_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)

        elif cache.get('predict model') == 'route':  # predict route passenger numbers
            destination = pd.read_csv('Destination.csv')
            dataframe, future_steps, proportion, start_date, \
            end_date, epoch, route_id, timing = set_predict_form(cache, 'route')
            route = aggregate_route_attribute(dataframe, start_date, end_date, route_id, 1440/TIMESTEP, timing)
            dates = route['Date'].values.tolist()
            set = np.array(route['Passengers'].values)
            route_pred = aggregate_route_attribute(dataframe, cache.get('start date'), cache.get('end date'), route_id,
                                                 1440 / TIMESTEP, timing)
            extend_dates = route_pred['Date'].values.tolist()
            set_pred = np.array(route_pred['Passengers'].values)
            model = check_predict_model(cache, 'route')
            if model == MV_LSTM:
                src, real_list, prediction_list, mape, rmse =\
                    run_multivariate_lstm(route, epoch=epoch, proportion=proportion, pred_set=route_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)
            else:
                src, real_list, prediction_list, mape, rmse = run(set, model_switch='off', model_type=model,
                                                      epoch=epoch, proportion=proportion, step=future_steps,
                                                                  pred_set=set_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)

        elif cache.get('predict model') == 'stop route': # predict stop and route passenger numbers
            destination = pd.read_csv('Destination.csv')
            dataframe, future_steps, proportion, start_date, \
            end_date, epoch, stop_id, route_id, timing = set_predict_form(cache, 'stop_route')
            stop_route = aggregate_double_attribute(dataframe, start_date,
                                                    end_date, stop_id, route_id, 1440/TIMESTEP, timing)
            dates = stop_route['Date'].values.tolist()
            set = np.array(stop_route['Passengers'].values)
            stop_route_pred = aggregate_double_attribute(dataframe, cache.get('start date'), cache.get('end date'), stop_id,
                                                    route_id, 1440 / TIMESTEP, timing)
            extend_dates = stop_route_pred['Date'].values.tolist()
            set_pred = np.array(stop_route_pred['Passengers'].values)
            model = check_predict_model(cache, 'stop_route')
            if model == MV_LSTM:
                src, real_list, prediction_list, mape, rmse = \
                    run_multivariate_lstm(stop_route, epoch=epoch, proportion=proportion, pred_set=stop_route_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)
            else:
                src, real_list, prediction_list, mape, rmse = run(set, model_switch='off', model_type=model,
                                                                  epoch=epoch, proportion=proportion, step=future_steps,
                                                                  pred_set=set_pred)
                create_prediction_list(cache, dates, extend_dates, real_list,
                                       prediction_list)
        return flask.render_template('prediction_result.html', src=src, form=form, mape=mape, rmse=rmse)

    if flask.request.method == 'POST':
        return  flask.redirect(flask.url_for('prediction'))

"""
The route for asynchronous loading using JSON. It is to load the future prediction outcome 
"""
@app.route('/prediction_dynamic', methods=['GET', 'POST'])
def prediction_dynamic():
    if flask.request.method == 'GET':
        parameters = {'list': cache.get('predict list'), 'dates': cache.get('predict dates'),
                      'real': cache.get('real list'), 'extent dates': cache.get('extent dates')}
        return flask.jsonify(parameters)

"""
The route for travel advice page
"""
@app.route('/advice', methods=['GET', 'POST'])
def advice():
    if flask.request.method == 'GET':
        return flask.render_template('travel_advice.html')

"""
The route for asynchronous loading using JSON. It is to load the top busy route
"""
@app.route('/transfer_top_routes', methods=['GET', 'POST'])
def transfer_top_routes():
    top_routes, passenger_list, date_list, total = sort_routes(cache.get('dataframe'),
                                            cache.get('start date'), cache.get('end date'))
    stop_info = heatmap_stop( cache.get('dataframe'), cache.get('start date'), cache.get('end date'), 1440)
    parameters = {'top_routes': top_routes, 'passenger_list': passenger_list,
                  'date_list': date_list, 'total': total, 'stop_info': stop_info}
    return flask.jsonify(parameters)

if __name__ == '__main__':
    app.run(debug=True)


