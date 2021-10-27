import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from project.data_preprocessing import read_traffic_csv, read_text_file,\
    aggregate_double_attribute, aggregate_route_attribute, aggregate_dataframe, aggregate_stop_attribute
from project.lstm_regression import run, aggregate_stop_attribute, aggregate_route_attribute
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    dataframe = read_traffic_csv('May 2019 TransactionReport-314828-1.csv')
    # dataframe = read_traffic_csv('April 2019 TransactionReport-314829-1.csv')
    # stop_UQ = aggregate_stop_attribute(dataframe, '2019/05/01 06:00', '2019/05/30 23:59', '1882', 120, 'Alighting')
    stop_UQ = aggregate_dataframe(dataframe, '2019/05/01 06:00', '2019/05/30 23:59', 120, 'Alighting')
    set = np.array(stop_UQ['Passengers'].values)
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(set, lags=20, ax=ax1)  # 自相关
    # ax1.xaxis.set_ticks_position('bottom')
    # fig.tight_layout()
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(set, lags=20, ax=ax2)  # 偏自相关
    # ax2.xaxis.set_ticks_position('bottom')
    # fig.tight_layout()

    train_size = int(len(set) * 0.7)
    train, test = set[0 : train_size], set[train_size : ]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 3))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
