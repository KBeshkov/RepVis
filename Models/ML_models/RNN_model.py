#RNN class for training on Allen data
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, n_neurons, n_stim, n_outputs, n_layers, rnn_type='RNN'):
        super(RNN, self).__init__()
        #params
        self.n_neurons = n_neurons
        self.n_stim = n_stim
        self.outputs = n_outputs
        self.n_layers = n_layers

        
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
        h0 = torch.zeros(self.n_layers,1,self.n_neurons).requires_grad_()
        rnn_out, h0 = self.rnn(x,h0.detach())
        rnn_out = rnn_out[:,-1,:]
        out = self.ff(rnn_out)
        return out
    

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
                x_batch = x_batch.view([batch_size, -1, n_features]).to("cpu")
                y_batch = y_batch.view([batch_size, -1, n_features]).to("cpu")
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            with torch.no_grad():
                batch_test_losses = []
                for x_test, y_test in test_loader:
                    x_test = x_test.view([batch_size, -1, n_features]).to("cpu")
                    y_test = y_test.view([batch_size, -1, n_features]).to("cpu")
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
                x_test = x_test.view([batch_size,-1,n_features]).to("cpu")
                y_test = y_test.view([batch_size, -1, n_features]).to("cpu")
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
