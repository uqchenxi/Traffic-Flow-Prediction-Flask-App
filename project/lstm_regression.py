import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from matplotlib.animation import ImageMagickWriter
from io import BytesIO
from torch import nn
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from project.data_preprocessing import read_traffic_csv, read_text_file,\
    aggregate_double_attribute, aggregate_route_attribute, aggregate_dataframe, aggregate_stop_attribute

NUMLIST = [str(i) for i in range(10)] # the number str from 1 to 9
TIMESTEPS = 12 # training size for each
LR = 0.02  # learning rate

"""
The function to extract hours and minutes from Time Interval str

Parameters:
    time_str: The time string to be extracted:
Returns:
    s_num: The time string to be derived like 09:00
"""
def get_time_period(time_str):
    status = False
    s_num = ''
    for i, num in enumerate(time_str):
        if num == ' ':
            status = True
        if status == True:
            if num in NUMLIST:
                s_num += num
        if len(s_num) == 4:
            status = False
    return s_num

"""
The function to cast the date string into date

Parameters:
    date_string: The date string to be converted
Returns:
    date: The date in format of datetime
"""
def casting_date(date_string):
    date = pd.to_datetime(date_string, format = '%Y%m%d %H:%M')
    return date

"""
Construct a function to split the dataframe into input and output for time series.

Parameters:
    raw_data: The dataset to be processed
    time_interval_num: The input size of dataset for each time step
"""
def create_sequence(raw_data, time_interval_num, input_size=1):
    sequence_in, sequence_out = [], []
    # for the single input flow
    if input_size == 1:
        for i in range(len(raw_data) - time_interval_num):
            element = raw_data[i:(i + time_interval_num)]
            sequence_in.append(element)
            sequence_out.append(raw_data[i + time_interval_num])
    # for the multi-dimensional input with event label
    else:
        for i in range(len(raw_data) - time_interval_num):
            element = raw_data[i:(i + time_interval_num)]
            sequence_in.append(element)
            sequence_out.append(raw_data[i + time_interval_num][0])
    return (np.array(sequence_in), np.array(sequence_out))

"""
The function to normalize data from 0 to 1

Parameters:
    dataset: The dataset to be normalized
Returns:
    dataset: The dataset has been normalized
"""
def normalize_dataset(dataset):
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    dataset = (dataset - min_value) / (max_value - min_value)
    return dataset

"""
Construct a function to separate the dataset into training set and testing set

Parameters:
    input: The input to be seperated
    output: The output to be seperated
    proporiton: The proportion of training set
    timestamp: The timestamp to be reshape
Returns:
    train_X: The training input of the dataset
    train_Y: The output of the dataset
    train_size: The size of training set
"""
def seperate_sequence(input, output, proportion, timestamp):
    train_size = int(len(input) * proportion)
    train_x = input[: train_size]
    train_y = output[: train_size]
    # Transfer the dataset into tensor object
    train_X = torch.from_numpy(train_x.reshape(-1, 1, timestamp))
    train_Y = torch.from_numpy(train_y.reshape(-1, 1, 1))
    return train_size, train_X, train_Y

"""
The function to calculate generate_mape

Parameters:
    output: The expected output
    prediction: The prediction
Returns:
    generate_mape
"""
def generate_mape(output, prediction):
    output, prediction = np.array(output), np.array(prediction)
    delete_index = []
    # please note this part is to avoid zero values
    for index, element in enumerate(output):
        if (element < 200):
            delete_index.append(index)
    # get the processed value list without zero
    output = np.delete(output, delete_index, axis=0)
    prediction = np.delete(prediction, delete_index, axis=0)
    return np.mean(np.abs((output - prediction) / output)) * 100

"""
Construct a function to invert dataset from tensor to numpy

Parameters:
    model: The expected output
    prediction: The prediction to be inverted
    timestamp: The timestamp of the model
    scaler: The scaler to be used in inversion
Returns:
    prediction_invert: The prediction values has been inverted
"""
def scale_invert(prediction, timestamp, scaler):
    prediction = np.concatenate((np.zeros(timestamp), prediction))
    prediction_invert = scaler.inverse_transform(prediction.reshape(-1, 1))
    prediction_invert = prediction_invert.reshape(-1)
    return prediction_invert

"""
The function to restore the net

Parameter:
    model_name: The name of the model
Returns:
    model: The model has been trained in the training process
"""
def restore_net(model_name):
    model = torch.load(model_name)

    return model

"""
Construct an LSTM module class to make passenger flow prediction 
"""
class LSTM_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=3, dropout=0.5):
        super(LSTM_REG, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout) # add the LSTM layer
        self.reg = nn.Linear(hidden_size, output_size) # add the linear regression layer

    def forward(self, _x):
        x, _ = self.rnn(_x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

"""
Construct a function to train the LSTM model and get the relevant values for visualization

Parameter:
    set: The dataset to be processed
    model_switch: The switch of restoring LSTM model
    epoch: The epoch in the training process
    proportion: The proportion of training size
    pred_set: The dataframe for the future prediction. Here, we must use 
    stream data set for the future prediction.
    step: The time steps for future prediction. Please note this parameters is only available when stream data is available
Return:
    src: The source link of the image outcome
    real_list: The real values of the passenger flow 
    predict_list: The predicted the result for passenger flow
    mape: The mape for benchmark
    rmse: The rmse for benchmark
"""
def run(set, model_switch, model_type, epoch, proportion,step, pred_set):
    global optimizer, model, loss_fun
    # normalize dataset with min max scalar
    scalar = MinMaxScaler()
    scaled_set = scalar.fit_transform(set.reshape(-1, 1))
    scaled_set = scaled_set.reshape(-1)
    scalar_pred = MinMaxScaler()
    # normalized future prediction dataset with min max scalar
    scaled_set_pred = scalar_pred.fit_transform(pred_set.reshape(-1, 1))
    scaled_set_pred = scaled_set_pred.reshape(-1)
    input_pred_np, output_pred_np = create_sequence(scaled_set_pred, TIMESTEPS)
    input_np, output_np = create_sequence(scaled_set, TIMESTEPS)
    train_size, train_X, train_Y = seperate_sequence(input_np, output_np, proportion, TIMESTEPS)

    # deploy deep learning model
    if model_switch == 'on':
        model = restore_net('./Model/LSTM/net_Adam_April_01-30_dropout_log.pkl')
        model.cuda()
    elif model_switch == 'off':
        model = model_type(TIMESTEPS, 64)
        model.cuda()
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    # training process
    fig = plt.figure(figsize=(12, 5))
    metadata = dict(title='Darren Movie Test', artist='Matplotlib',
                    comment='Darren Movie support!')
    writer = ImageMagickWriter(fps=15, metadata=metadata)
    if model_switch == 'off':
        # get the gif file for training process
        with writer.saving(fig, r'C:\Users\15078\Desktop\Thesis\Code\Script\webapp\static\prediction.gif', 100):
            for i in range(epoch):
                out = model(train_X.cuda())
                loss = loss_fun(out, train_Y.cuda())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (i + 1) % 50 == 0:
                    print('Epoch: {}, Loss:{:.6f}'.format(i + 1, loss.item()))
                if (i + 1) % 20 == 0:
                    steps = np.linspace(0, 1, len(train_X), dtype=np.float32, endpoint=False)
                    plt.plot(steps, train_Y.flatten(), 'r', label='Real Passenger Flow')
                    plt.plot(steps, out.cpu().data.numpy().flatten(), 'b', label='LSTM Predicted Passenger Flow')
                    plt.ylabel('Passenger Number')
                    writer.grab_frame()
                    plt.clf()

        torch.save(model, 'net_Adam_April_01-30_dropout_log.pkl') # replace with derived model name
    plt.close()
    model = model.eval()
    model.cuda()

    # reshape the input values
    input = input_np.reshape(-1, 1, TIMESTEPS)
    input = torch.from_numpy(input)
    # reshape the input values for future prediction
    input_pred = input_pred_np.reshape(-1, 1, TIMESTEPS)
    input_pred = torch.from_numpy(input_pred)
    # predict the values for the input
    prediction = model(input.cuda())
    # predict the values for future prediction
    predict_np = prediction.cpu().view(-1).data.numpy()
    future_plot = model(input_pred.cuda())
    future_plot = future_plot.cpu().view(-1).data.numpy()
    # invert the prediction result
    prediction_invert = scale_invert(predict_np, TIMESTEPS, scalar)
    future_plot_invert = scale_invert(future_plot, TIMESTEPS, scalar_pred)
    predict_list = future_plot_invert.tolist() # the list of predicted passenger values
    real_list = set.tolist() # the list of real passenger values

    # calculate RMSE and MAPE
    assert len(prediction_invert) == len(set)
    rmse = sqrt(mean_squared_error(set[train_size:], prediction_invert[train_size:]))
    mape = generate_mape(set[TIMESTEPS:], prediction_invert[TIMESTEPS:])
    print('Test RMSE: %.3f' % rmse)
    print('Test MAPE: %.3f' % mape)

    # plot the figure of the outcome
    height = set.max()
    length = len(set)
    plt.figure(figsize=(12, 5))
    plt.plot(prediction_invert[TIMESTEPS:].flatten(), 'b', label='Predicted Passenger Flow')
    plt.plot(set[TIMESTEPS:].flatten(), 'r', label='Real Passenger Flow')
    plt.ylabel('Passenger Number')
    plt.text(length * 0.1, height, 'RMSE = %.3f' % rmse, fontdict={'size': 10, 'color': 'green'})
    plt.text(length * 0.1, height * 0.9, 'MAPE = %.3f' % mape, fontdict={'size': 10, 'color': 'green'})
    plt.plot((train_size, train_size), (0, height), 'g--')
    plt.legend(loc='best')
    # plt.show()
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()
    # return the source link of the image to front page
    src = 'data:png;base64,' + str(data)
    # The returned parameters are used for the front-end page.
    return src, real_list, predict_list, 'MAPE = %.3f' % mape, 'RMSE = %.3f' % rmse

