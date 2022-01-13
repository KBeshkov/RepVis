#Example training of RNN on Allen data
import os
os.chdir('/home/kosio/Repos/RepVis')
from Analysis.Load_data import *
from Analysis.Preprocessing import *
from Models.ML_models.RNN_model import *

session = load_data('/mnt/hdd1/Data/ecephys_cache_dir/session_715093703/session_715093703.nwb')
stim_type = 'drifting_gratings'
region = 'VISp'
block = 2
bin_time = 0.05

dataset = open_data_cont(session,stim_type,region,block,bin_time)
stim_encoding = stim_to_one_hot(session,stim_type,block,bin_time)

rates_kern = smooth_rates(dataset,10,bin_time)

#%% Trace plots for different kernels

kern_values = np.linspace(0.5,5,8)#np.logspace(-5,2,8)

#Gaussian kernels
gauss_cells = []
for k in kern_values:
    gauss_cells.append(smooth_rates(dataset,k,bin_time))
    
# #Exponential kernels
# exp_cells = []
# for k in kern_values:
#     exp_cells.append(smooth_rates(dataset,k,bin_time,kern_type='Exp'))

# #Step kernels
# step_kern_values = np.arange(1,23,3)
# step_cells = []
# for k in step_kern_values:
#     step_cells.append(smooth_rates(dataset,k,bin_time,kern_type='Step'))

#%%Plots
plt.figure(dpi=400,figsize=(12,6))
for i in range(8):
    plt.subplot(4,2,i+1)
    plt.plot(gauss_cells[i][21],'g')
    # plt.plot(exp_cells[i][36],'b')
    # plt.plot(step_cells[i][36],'r')
    plt.xticks([])
plt.tight_layout()

#%% RNN training on smoothed data
from torch.utils.data import TensorDataset, DataLoader


model = RNN(len(rates_kern),1,len(rates_kern),1,rnn_type='RNN')
datset = np.vstack([gauss_cells[1],stim_encoding[1]])
np.save('/Data/dat_example1.npy')

# X_ = torch.Tensor(split_by_stim(gauss_cells[1],stim_encoding[2],bin_time))
X_ = torch.Tensor(split_by_len(datset,2))
X = X_[:,:-1,:]
y = X_[:,1:,:-1]
train_ids = np.random.choice(np.arange(0,len(X)),size=int(len(X_)*0.7),replace=False)
test_ids = np.setdiff1d(np.arange(0,len(X)),train_ids)
X_train, X_test = X[train_ids], X[test_ids]
y_train, y_test = y[train_ids], y[test_ids]
data_train = TensorDataset(X_train, y_train)
data_test = TensorDataset(X_test, y_test)

batch_sz = 1#np.shape(X)[2]
dataloader_train = DataLoader(data_train,batch_size=batch_sz,shuffle=False)
dataloader_test = DataLoader(data_test,batch_size=batch_sz,shuffle=False)

n_epoch = 50
lr = 0.001
crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

opt = Optimization(model=model,loss_fn = crit, optimizer=optimizer)
opt.train(dataloader_train, dataloader_test, batch_size=batch_sz, n_epochs=n_epoch, n_features=np.shape(X)[1])

predicted, actual = opt.test(dataloader_test, batch_size=batch_sz,n_features=np.shape(X)[1])

#train set predicitons
init_x = X_train[:,0,:]
init_x = init_x[:,None,:]
stim_train = X_train[:,:,-1]
stim_train = stim_train[:,:,None]
rnn_pred_train = Model_predict(model,1,init_x,stim_train)

#test set predictions
init_x = X_test[:,0,:]
init_x = init_x[:,None,:]
stim_test = X_test[:,:,-1]
stim_test = stim_test[:,:,None]
rnn_pred_test = Model_predict(model,1,init_x,stim_test)


#%% Performance eval
opt.plot_losses()

train_set = torch.flatten(y_train,0,1).detach().numpy()
pred_train_set = torch.flatten(rnn_pred_train,0,1).detach().numpy()

test_set = torch.flatten(y_test,0,1).detach().numpy()
pred_test_set = torch.flatten(rnn_pred_test,0,1).detach().numpy()


plt.figure()
plt.subplot(2,1,1)
plt.imshow(train_set.T)
plt.subplot(2,1,2)
plt.imshow(pred_train_set.T)
plt.tight_layout()

plt.figure()
plt.subplot(2,1,1)
plt.imshow(test_set.T)
plt.subplot(2,1,2)
plt.imshow(pred_test_set.T)
plt.tight_layout()

nrn_plt_n = 1
rand_nrns = np.random.choice(np.arange(0,X_.shape[2]),replace=False,size=nrn_plt_n)
plt.figure(dpi=300)
for i in range(nrn_plt_n):
    plt.subplot(2,1,1)
    plt.plot(train_set[:,rand_nrns[i]],'k')
    plt.plot(pred_train_set[:,rand_nrns[i]],'r')
    plt.subplot(2,1,2)
    plt.plot(test_set[:,rand_nrns[i]],'k')
    plt.plot(pred_test_set[:,rand_nrns[i]],'r')
plt.show()
    
    
    