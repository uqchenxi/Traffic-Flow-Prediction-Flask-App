from project.rnn_regression import RNN
from project.gru_regression import GRU_REG
from project.lstm_regression import LSTM_REG
from project.multivariate_lstm_regresson import MV_LSTM

def create_prediction_list(cache, dates, extend_dates, real_list, prediction_list):
    cache.set('real list', real_list)
    cache.set('extent dates', extend_dates)
    cache.set('predict list', prediction_list)
    cache.set('predict dates', dates)

def set_visualize_form(cache, type):
    if type == 'stop_route':
        parameters = {'title_route': cache.get('form_stop_route')['stop_route_id_route'],
                      'title_stop': cache.get('form_stop_route')['stop_route_id_stop'],
                      'time_interval': cache.get('form_stop_route')['stop_route_time_interval']}
    else:
        parameters = {'title': cache.get('form_' + type)[type + '_id'],
                      'time_interval': cache.get('form_' + type)[type + '_time_interval']}
    start_date = cache.get('form_' + type)[type + '_date_start']
    end_date = cache.get('form_' + type)[type + '_date_end']
    if cache.get('form_' + type)[type + '_timing'] == 1:
        timing = 'Alighting'
    else:
        timing = 'Boarding'
    return parameters, start_date, end_date, timing

def set_predict_form(cache, type):
    dataframe = cache.get('dataframe')
    future_steps = int(cache.get('pre_form_' + type)['predict_' + type + '_future_steps'])
    proportion = cache.get('pre_form_' + type)['predict_' + type + '_train'] / 100
    start_date = cache.get('pre_form_' + type)['predict_' + type + '_date_start']
    end_date = cache.get('pre_form_' + type)['predict_' + type + '_date_end']
    epoch = cache.get('pre_form_' + type)['predict_' + type + '_epoch']
    if cache.get('pre_form_' + type)['predict_' + type + '_timing'] == 1:
        timing = 'Alighting'
    else:
        timing = 'Boarding'
    if type == 'stop_route':
        stop_id = cache.get('pre_form_' + type)['predict_' + type + '_id_stop']
        route_id = cache.get('pre_form_' + type)['predict_' + type + '_id_route']
        return dataframe, future_steps, proportion, start_date,\
           end_date, epoch, stop_id, route_id, timing
    else:
        id = cache.get('pre_form_' + type)['predict_' + type + '_id']
        return dataframe, future_steps, proportion, start_date,\
           end_date, epoch, id, timing

def check_predict_model(cache, type):
    if cache.get('pre_form_' + type)['predict_' + type + '_model'] == 1:
        model = MV_LSTM
    elif cache.get('pre_form_' + type)['predict_' + type + '_model'] == 2:
        model = LSTM_REG
    elif cache.get('pre_form_' + type)['predict_' + type + '_model'] == 3:
        model = RNN
    else:
        model = GRU_REG
    return model
