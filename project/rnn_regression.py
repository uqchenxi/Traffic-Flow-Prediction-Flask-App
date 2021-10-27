import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from project.data_preprocessing import read_traffic_csv, read_text_file,\
     aggregate_stop_attribute
from project.lstm_regression import run

NUMLIST = [str(i) for i in range(10)]
TRAINING_TIME_INTERVAL_NUM = 12
LR = 0.02  # learning rate


"""
Construct an RNN module class to make passenger flow prediction 
"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.5):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout) # add an RNN layer
        self.reg = nn.Linear(hidden_size, output_size) # add a linear regression layer
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


