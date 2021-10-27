import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

NUMLIST = [str(i) for i in range(10)]

"""
Construct a function to extract hours and minutes from Time Interval str

Parameters:
    time_str: the time string to be extract:
Returns:
    s_num: the expected number of string
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



def casting_date(date):
    date = pd.to_datetime(date, format = '%Y%m%d %H:%M')
    return date

"""
Construct a function to extract passenger flow and corresponding date from a stop

Parameters:
    dataframe: The traffic dataframe to be extract
    stop_id: The stop to be analyzed
Return:
    stop: the customized data only contains time interval and passengers
"""
def create_stop_data(stop_id, dataframe):
    stop = dataframe[dataframe['stop_id'] == stop_id]
    stop = stop[['Time Interval', 'Passengers']]
    stop['Time Interval'] = stop['Time Interval'].apply(lambda x: get_time_period(x))
    stop = stop.astype('float32')
    return stop

"""
Construct a function to to normalize data from 0 to 1

Parameters:
    dataset: The dataset to be normalized
Returns:
    dataset: The normalized data
"""
def normalize_dataset(dataset):
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    dataset = (dataset - min_value) / (max_value - min_value)
    return dataset

"""
Construct a linear module class to make passenger flow prediction 
"""
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)


    def forward(self, x):
        x = F.softmax(self.hidden(x))
        x = self.predict(x)
        return x

# read the processed data from passenger flow file and destination file
dataframe = pd.read_csv('./Processed_Result/Aggregate_Passenger_Flow_April_01.csv')
dataframe = dataframe.dropna()
destination = pd.read_csv('./Processed_Result/Destination.csv')
dataframe['stop_id'] = dataframe['stop_id'].astype(str)

# combine the passenger flow file and destination file
combined_data = pd.merge(destination, dataframe, on = 'stop_id')

# extract the dataset of UQ lake on April 1st, 2019
stop_UQ = create_stop_data('10795', dataframe)
train_set_time = np.array(stop_UQ['Time Interval'].values)
train_set_passenger = np.array(stop_UQ['Passengers'].values)
train_set_passenger = normalize_dataset(train_set_passenger)
train_set_time = normalize_dataset(train_set_time)
torch_time = torch.from_numpy(train_set_time[:, np.newaxis])
torch_passenger = torch.from_numpy(train_set_passenger[:, np.newaxis])

net = Net(1, 128, 1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(1000):

    prediction = net(torch_time)
    loss = loss_func(prediction, torch_passenger)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (t + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(t + 1, loss.item()))
    plt.cla()
    plt.plot(train_set_time.data, prediction.data.numpy(), 'r-')
    plt.plot(train_set_time.data, train_set_passenger, 'b-')
    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color': 'red'})
    plt.title('Passenger Flow for Boggo Road station [10795] during April 1st, 2019')
    plt.xlabel('Time Interval : 15 minutes / Duration: April 1st')
    plt.ylabel('Passenger Number')
    plt.pause(0.1)
plt.ioff()


net = net.eval()
pred_test = net(torch_time)
pred_test = pred_test.data.numpy()
plt.plot(torch_time.data.numpy(), torch_passenger.data.numpy(), 'b-', lw = 2, label = 'Real Passenger Flow')
plt.plot(torch_time.data.numpy(), prediction.data.numpy(), 'r-', lw = 2, label = 'Linear Prediction')
plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict = {'size':10, 'color':'red'})
plt.legend(loc='best')
plt.xlabel('Time Interval : 15 minutes / Duration: April 1st')
plt.ylabel('Passenger Number')
#plt.title('Passenger Flow for UQ Lake Station [1882] during April 1st, 2019')
plt.title('Passenger Flow for Boggo Road station [10795] during April 1st, 2019')
plt.plot(train_set_time, pred_test, 'r', label='prediction')
plt.show()