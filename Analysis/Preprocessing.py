#Data processing functions
import numpy as np
from scipy.signal import convolve2d
import torch
from torch.utils.data import Dataset


'''
Smoothing of spikes/rates:
    rates: a neuronXtime array
    kern_width: the width parameter of the convolution kernel
    tstep: the size of a time step in the data
    kern_type: either Gaussian <(1/b*sqrt(2pi))*exp(-(x^2)/(2b))>, exponential <exp(-bx)>,
    box <1> on an interval of lenght kern_width
'''
    
def smooth_rates(rates,kern_width,tstep,kern_type='Gauss',norm=True):
    tot_time = len(rates.T)*tstep
    if kern_type=='Gauss':
        kern_range = np.arange(-tot_time/2,tot_time/2,tstep)
        kern = (1/kern_width*np.sqrt(2*np.pi))*np.exp(-(kern_range**2)/(2*kern_width**2))
        kern = np.expand_dims(kern,0)
        smooth_rates = convolve2d(rates.T,kern.T).T
        smooth_rates = smooth_rates[:,int(len(rates.T)/2)-1:len(smooth_rates.T)-int(len(rates.T)/2)]
    elif kern_type=='Exp':
        kern_range = np.arange(0,tot_time,tstep)
        kern = np.exp(-kern_range/kern_width)
        kern = np.expand_dims(kern,0)
        smooth_rates = convolve2d(rates.T,kern.T).T
        smooth_rates = smooth_rates[:,:len(smooth_rates.T)-int(len(rates.T))]
    elif kern_type=='Step':
        kern_range = np.arange(-tot_time/2,tot_time/2,tstep)
        midpoint = int(len(kern_range)/2)
        kern = np.zeros(len(kern_range))
        kern[midpoint-kern_width:midpoint+kern_width] = 1
        kern = np.expand_dims(kern,0)
        smooth_rates = convolve2d(rates.T,kern.T).T
        smooth_rates = smooth_rates[:,int(len(rates.T)/2)-1:len(smooth_rates.T)-int(len(rates.T)/2)]
    if norm==True:
        smooth_rates = (smooth_rates-np.mean(smooth_rates))/np.std(smooth_rates)
    return smooth_rates

'''
Split data by stimuli:
    dat: dataset to be split
    intervals: a list of lengths for each stimulus
    tstep: the size of a single time step
'''

def split_by_stim(dat,intervals,tstep):
    increment = int(np.mean(intervals)/tstep)
    new_dat = np.zeros([len(intervals),increment,len(dat)])
    curr_step = 0
    for i in range(len(intervals)):
        new_dat[i,:,:] = dat[:,curr_step:curr_step+increment].T
        curr_step = curr_step+increment
    return new_dat


def split_by_len(dat,tlen):
    nsamples = len(dat.T)-tlen
    new_dat = np.zeros([nsamples,tlen,len(dat)])
    rnd_indcs = np.arange(0,nsamples)
    for i in range(nsamples):
        new_dat[i,:,:,] = dat[:,rnd_indcs[i]:rnd_indcs[i]+tlen].T
    return new_dat