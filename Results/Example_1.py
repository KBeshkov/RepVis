#Example training of RNN on Allen data
import os
os.chdir('/home/kosio/Repos/RepVis')
from Analysis.Load_data import *
from Analysis.Preprocessing import *
from Models.ML_models import *

session = Load_data.load_data('/mnt/hdd1/Data/ecephys_cache_dir/session_715093703/session_715093703.nwb')
stim_type = 'drifting_gratings'
region = 'VISp'
block = 2
bin_time = 0.05

dataset = Load_data.open_data_cont(session,stim_type,region,block,bin_time)
stim_encoding = stim_to_one_hot(session,stim_type,block,bin_time)

rates_kern = smooth_rates(dataset,10,bin_time)

#%% Trace plots for different kernels

kern_values = np.linspace(0.5,5,8)#np.logspace(-5,2,8)

#Gaussian kernels
gauss_cells = []
for k in kern_values:
    gauss_cells.append(smooth_rates(dataset,k,bin_time))
    
#Exponential kernels
exp_cells = []
for k in kern_values:
    exp_cells.append(smooth_rates(dataset,k,bin_time,kern_type='Exp'))

#Step kernels
step_kern_values = np.arange(1,23,3)
step_cells = []
for k in step_kern_values:
    step_cells.append(smooth_rates(dataset,k,bin_time,kern_type='Step'))

#%%Plots
plt.figure(dpi=400,figsize=(12,6))
for i in range(8):
    plt.subplot(4,2,i+1)
    plt.plot(gauss_cells[i][26],'g')
    # plt.plot(exp_cells[i][36],'b')
    # plt.plot(step_cells[i][36],'r')
    plt.xticks([])
plt.tight_layout()

#%% RNN training on smoothed data
from torch.utils.data import TensorDataset, DataLoader


model = RNN(len(rates_kern),0,len(rates_kern),1)

X_ = torch.Tensor(split_by_stim(gauss_cells[3],stim_encoding[2],bin_time))
X = X_[:,:,:-1]
y = X_[:,:,1:]
train_ids = np.random.choice(np.arange(0,len(X)),size=int(len(X_)*0.7),replace=False)
test_ids = np.setdiff1d(np.arange(0,len(X)),train_ids)
X_train, X_test = X[train_ids], X[test_ids]
y_train, y_test = y[train_ids], y[test_ids]
data_train = TensorDataset(X_train, y_train)
data_test = TensorDataset(X_test, y_test)
dataloader_train = DataLoader(data_train)
dataloader_test = DataLoader(data_test)

batch_sz = np.shape(X)[2]
n_epoch = 1000
lr = 0.01
crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

opt = Optimization(model=model,loss_fn = crit, optimizer=optimizer)
opt.train(dataloader_train, dataloader_test, batch_size=batch_sz, n_epochs=100, n_features=np.shape(X)[1])
opt.plot_losses()

predicted, actual = opt.test(dataloader_test, batch_size=batch_sz,n_features=np.shape(X)[1])
