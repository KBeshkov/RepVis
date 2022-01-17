#RNN class for training on Allen data
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class RNN(nn.Module):
    def __init__(self, n_neurons, n_outputs, n_layers, n_tsteps=1,rnn_type='RNN',dropout_rate=0.1):
        super(RNN, self).__init__()
        #params
        self.n_neurons = n_neurons
        self.outputs = n_outputs
        self.n_layers = n_layers
        self.n_tsteps = n_tsteps
        
        #RNN layer
        if rnn_type=='RNN':
            self.rnn_dict = OrderedDict()
            for i in range(self.n_tsteps):
                self.rnn_dict['rnn'+str(i)] = nn.RNN(n_neurons,n_neurons,n_layers,dropout=dropout_rate)
                self.rnn_dict['rlin'+str(i)] = nn.Linear(n_neurons,n_neurons)
            self.rnn = nn.Sequential(OrderedDict(self.rnn_dict))
        elif rnn_type=='GRU':
            self.rnn = nn.GRU(n_neurons,n_neurons,n_layers,dropout=dropout_rate)
        elif rnn_type=='LSTM':
            self.rnn = nn.LSTM(n_neurons,n_neurons,n_layers,dropout=dropout_rate)
        else:
            raise SystemExit('invalid RNN type')
                    
        
    def forward(self, x):
        out = torch.zeros(x.shape[0],self.n_tsteps,self.n_neurons)
        rnn_out = x.clone()[:,0,:]
        rnn_out = rnn_out[:,None,:]
        for n,layer in enumerate(self.rnn):
            if n%2==0:
                rnn_out = layer(rnn_out)
            else:
                rnn_out = layer(rnn_out[0])
                out[:,int((n-1)/2)] = rnn_out
        return out
    
    def get_params(self):
        param_dict = {}
        for i, layer in enumerate(self.rnn):
            param_dict[str(i)] = layer.weight_ih_l()
        return param_dict

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self. loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        
    def train_step(self, x, y):
        
        self.model.train()
        
        yhat = self.model(x)
        
        loss = self.loss_fn(y, yhat)
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self, train_loader, test_loader, batch_size, n_epochs, n_features):
        for epoch in range(1, n_epochs+1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to("cpu")
                y_batch = y_batch.to("cpu")
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            with torch.no_grad():
                batch_test_losses = []
                for x_test, y_test in test_loader:
                    x_test = x_test.to("cpu")
                    y_test = y_test.to("cpu")
                    self.model.eval()
                    yhat = self.model(x_test)
                    test_loss = self.loss_fn(y_test, yhat).item()
                    batch_test_losses.append(test_loss)
                test_loss = np.mean(batch_test_losses)
                self.test_losses.append(test_loss)
                
            if epoch%5 ==0:
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {test_loss:.4f}")
            
    def test(self, test_loader, batch_size, n_features):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.to("cpu")
                y_test = y_test.to("cpu")
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to("cpu").detach().numpy())
                values.append(y_test.to("cpu").detach().numpy())
                
        return predictions, values
    
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.test_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()

def Model_predict(model,iters,init_cond, cond_set):
    x = torch.zeros([len(init_cond),iters+1,np.shape(init_cond)[2]])
    x[:,0,:] = np.squeeze(init_cond)
    for i in range(iters):
        # inp_stim = cond_set[:,i,:]
        # inp_stim = inp_stim[:,None,:]
        inp_x = x[:,i,:]
        inp_x = inp_x[:,None,:]
        # inp_x = torch.dstack([inp_x,inp_stim])
        x[:,i+1,:] = model(inp_x)
    return x[:,1:,:]
        

