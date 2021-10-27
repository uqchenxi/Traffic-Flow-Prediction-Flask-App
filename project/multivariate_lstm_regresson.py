import numpy as np
import torch
import matplotlib.pyplot as plt
import base64
from matplotlib.animation import ImageMagickWriter
from io import BytesIO
from torch import nn
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from project.data_preprocessing import read_traffic_csv
from project.lstm_regression import aggregate_stop_attribute, \
    create_sequence, generate_mape, scale_invert, aggregate_route_attribute

FEATURES= 2 # this is number of parallel inputs
TIME_STEPS = 12 # this is number of timesteps

# split a multivariate sequence into samples
def seperate_sequence(input, output, proportion):
    train_size = int(len(input) * proportion)
    train_x = input[: train_size]
    train_y = output[: train_size]
    train_X = torch.tensor(train_x, dtype=torch.float32)
    train_Y = torch.tensor(train_y, dtype=torch.float32)
    return train_size, train_X, train_Y

"""
Construct an multrvariate LSTM module class to make passenger flow prediction 
"""
class MV_LSTM(nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.FEATURES= n_features
        self.seq_len = seq_length
        self.n_hidden = 64 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
        self.dropout = 0.5
        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True,
                                dropout=self.dropout)
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)

    def forward(self, x): # the forward function for LSTM model
        batch_size, seq_len, _ = x.size()
        lstm_out, hidden = self.l_lstm(x)
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

"""
Construct a function to train the LSTM model and get the relevant values for visualization

Parameter:
    set: The dataset to be processed
    epoch: The epoch in the training process
    proportion: The proportion of training size
    pred_set: The dataframe for the future prediction. Here, we must use 
    stream data set for the future prediction.
Return:
    src: The source link of the image outcome
    real_list: The real values of the passenger flow 
    predict_list: The predicted the result for passenger flow
    mape: The mape for benchmark
    rmse: The rmse for benchmark
"""
def run_multivariate_lstm(set, epoch, proportion, pred_set):
    passengers = np.array(set['Passengers'].values) # get the passenger flow
    holiday = np.array(set['Is_holiday'].values) # get the public event label
    passengers_pred = np.array(pred_set['Passengers'].values) # get the passenger flow for future prediction
    holiday_pred = np.array(pred_set['Is_holiday'].values) # get the label events for future prediction

    # normalize predicted dataset and separate into train and test set
    scalar = MinMaxScaler()
    scaled_passengers = scalar.fit_transform(passengers.reshape(-1, 1))
    scaled_passengers = scaled_passengers.reshape(-1)
    scaled_passengers = scaled_passengers.reshape((len(passengers), 1))
    holiday = holiday.reshape((len(holiday), 1))
    dataset = np.hstack((scaled_passengers, holiday))
    input_np, output_np = create_sequence(dataset, TIME_STEPS, 2)
    train_size, train_X, train_Y = seperate_sequence(input_np, output_np, proportion)

    # this is for future prediction result to generate train and test set
    scalar_pred = MinMaxScaler()
    scaled_passengers_pred = scalar_pred.fit_transform(passengers_pred.reshape(-1, 1))
    scaled_passengers_pred = scaled_passengers_pred.reshape(-1)
    scaled_passengers_pred = scaled_passengers_pred.reshape((len(scaled_passengers_pred), 1))
    holiday_pred = holiday_pred.reshape((len(holiday_pred), 1))
    dataset_pred = np.hstack((scaled_passengers_pred, holiday_pred))
    input_pred_np, output_pred_np = create_sequence(dataset_pred, TIME_STEPS, 2)

    # create NN
    mv_net = MV_LSTM(FEATURES, TIME_STEPS)
    mv_net.cuda()
    criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=0.02)
    # training process
    fig = plt.figure(figsize=(12, 5))
    metadata = dict(title='Darren Movie Test', artist='Matplotlib',
                    comment='Darren Movie support!')
    writer = ImageMagickWriter(fps=15, metadata=metadata)
    with writer.saving(fig, r'C:\Users\15078\Desktop\Thesis\Code\Script\webapp\static\prediction.gif', 100):
        for t in range(epoch):
            output = mv_net(train_X.cuda())
            loss = criterion(output.view(-1), train_Y.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (t + 1) % 50 == 0:
                print('Epoch: {}, Loss:{:.6f}'.format(t + 1, loss.item()))
            if (t + 1) % 20 == 0:
                steps = np.linspace(0, 1, len(train_X), dtype=np.float32, endpoint=False)
                plt.plot(steps, train_Y.flatten(), 'r', label='Real Passenger Flow')
                plt.plot(steps, output.cpu().data.numpy().flatten(), 'b', label='LSTM Predicted Passenger Flow')
                plt.ylabel('Passenger Number')
                writer.grab_frame()
                plt.clf()
        torch.save(mv_net, 'net_Adam_April_01-30_dropout_log.pkl') # please replace the model name
    plt.close()

    # evaluate the model
    mv_net = mv_net.eval()
    input = torch.tensor(input_np, dtype=torch.float32)
    prediction = mv_net(input.cuda())
    predict_np = prediction.cpu().view(-1).data.numpy()

    # invert the prediction value and return the list for the front page
    predict_invert = scale_invert(predict_np, TIME_STEPS, scalar)
    input_pred = torch.tensor(input_pred_np, dtype=torch.float32)
    future = mv_net(input_pred.cuda())
    future_np = future.cpu().view(-1).data.numpy()
    future_invert = scale_invert(future_np , TIME_STEPS, scalar_pred)
    predict_list = future_invert.tolist() # the predicted future passenger values
    real_list = passengers.tolist()  # the real list of passenger values

    # generate benchmark
    assert len(predict_invert) == len(set)
    rmse = sqrt(mean_squared_error(passengers[train_size:], predict_invert[train_size:]))
    mape = generate_mape(passengers[TIME_STEPS:], predict_invert[TIME_STEPS:])
    print('Test RMSE: %.3f' % rmse)
    print('Test MAPE: %.3f' % mape)

    # visualize the outcome
    height = passengers.max()
    length = len(passengers)
    plt.figure(figsize=(12, 5))
    plt.plot(predict_invert[TIME_STEPS:].flatten(), 'b', label='Predicted Passenger Flow')
    plt.plot(passengers[TIME_STEPS:].flatten(), 'r', label='Real Passenger Flow')
    plt.ylabel('Passenger Number')
    # plt.text(length * 0.1, height, 'RMSE = %.3f' % rmse, fontdict={'size': 10, 'color': 'green'})
    # plt.text(length * 0.1, height * 0.9, 'MAPE = %.3f' % mape, fontdict={'size': 10, 'color': 'green'})
    plt.plot((train_size, train_size), (0, height), 'g--')
    plt.legend(loc='best')
    # plt.show()
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:png;base64,' + str(data)
    return src,  real_list, predict_list, 'MAPE = %.3f' % mape, 'RMSE = %.3f' % rmse

if __name__ == '__main__':
    dataframe = read_traffic_csv('April 2019 TransactionReport-314829-1.csv')
    # dataframe = read_traffic_csv('December 2018 TransactionReport-314840-1.csv')
    stop_UQ = aggregate_stop_attribute(dataframe, '2019/04/01 06:00', '2019/04/30 23:59', '1882', 60, 'Alighting')
    run_multivariate_lstm(stop_UQ, 50, 0.7, stop_UQ)
    # stop_UQ = aggregate_dataframe(dataframe, '2019/05/01 06:00', '2019/05/30 23:59', 120, 'Alighting')
    # stop_UQ = aggregate_dataframe(dataframe, '2019/04/01 06:00', '2019/04/07 23:59', 15, 'Alighting')

