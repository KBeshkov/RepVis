#RNN class for training on Allen data
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, n_neurons, n_stim, n_outputs, n_layers, batch_sz,rnn_type='RNN'):
        super(RNN, self).__init__()
        #params
        self.n_neurons = n_neurons
        self.n_stim = n_stim
        self.outputs = n_outputs
        self.n_layers = n_layers
        self.batch_sz = batch_sz

        
        #RNN layer
        if rnn_type=='RNN':
            self.rnn = nn.RNN(n_neurons+n_stim,n_neurons,n_layers)
        elif rnn_type=='GRU':
            self.rnn = nn.GRU(n_neurons+n_stim,n_neurons,n_layers)
        elif rnn_type=='LSTM':
            self.rnn = nn.LSTM(n_neurons+n_stim,n_neurons,n_layers)
        else:
            raise SystemExit('invalid RNN type')
            
        self.ff = nn.Linear(n_neurons,n_neurons)
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers,self.batch_sz,self.n_neurons).float()
        rnn_out = self.rnn(x,h0)
        out = self.ff(rnn_out).T
        return out