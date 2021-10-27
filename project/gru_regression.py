from torch import nn
LR = 0.02  # learning rate

"""
Construct an GRU module class to make passenger flow prediction 
"""
class GRU_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout = 0.5):
        super(GRU_REG, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout) # add a gru layer
        self.reg = nn.Linear(hidden_size, output_size) # add a linear regression layer

    def forward(self, x): # define the forward function for regression
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

