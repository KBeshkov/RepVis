#Load Allen data from path
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
import numpy as np

def load_data(path):
    session = EcephysSession.from_nwb_path(path)
    return session


def invalid_trials_ind(sess,stim_condition):
    stim = sess.stimulus_presentations[sess.stimulus_presentations['stimulus_name'] == stim_condition].index.values
    q = []
    if len(sess.invalid_times)>0:
        iv_times_start = sess.invalid_times.start_time.values
        iv_times_stop = sess.invalid_times.stop_time.values
        o = sess.stimulus_presentations
        s_times = o.start_time[stim].values
        for i in range(len(iv_times_start)):
            q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
    return sum(q)

def open_data_by_stim(sess,stim_condition,region,bin_step,bin_strt=0,snr=0,frate=0,inval=True):
    stim = sess.stimulus_presentations[sess.stimulus_presentations['stimulus_name'] == stim_condition].index.values
    decent_snr_unit_ids = sess.units[np.logical_and(sess.units['snr'] >= snr, sess.units['firing_rate'] > frate)]        
    decent_snr_unit_ids = list(decent_snr_unit_ids[decent_snr_unit_ids['ecephys_structure_acronym']==region].index.values)
    if inval==True:
        if len(sess.invalid_times)>0:
            iv_times_start = sess.invalid_times.start_time.values
            iv_times_stop = sess.invalid_times.stop_time.values
            o = sess.stimulus_presentations
            s_times = o.start_time[stim].values
            q = []
            for i in range(len(iv_times_start)):
                q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
            stim = stim[np.where(sum(q)==0)]
    durations = sess.stimulus_presentations.duration[stim]
    bins = []
    for i in range(len(durations)):
        bins.append(np.arange(bin_strt,np.max(durations.values),bin_step))
    stim_labels = list(sess.stimulus_presentations.stimulus_condition_id[stim])
    spikes_per_stim = []
    count = 0
    for i in np.unique(stim_labels):
        spike_counts_da = sess.presentationwise_spike_counts(
            bin_edges=bins[count],
            stimulus_presentation_ids=stim[stim_labels==i],
            unit_ids=decent_snr_unit_ids,
            binarize=False
        )
       
        spikes_per_stim.append(spike_counts_da.data)
        count = count + 1
    return [spikes_per_stim,durations]

def open_data_cont(sess,stim_condition,region,stim_block,bin_step,snr=0,frate=0,inval=True):
    stims = sess.stimulus_presentations
    stim = stims[np.logical_and(stims['stimulus_name'] == stim_condition,stims['stimulus_block']==stim_block)].index.values
    decent_snr_unit_ids = sess.units[np.logical_and(sess.units['snr'] >= snr, sess.units['firing_rate'] > frate)]        
    decent_snr_unit_ids = list(decent_snr_unit_ids[decent_snr_unit_ids['ecephys_structure_acronym']==region].index.values)
    if inval==True:
        if len(sess.invalid_times)>0:
            iv_times_start = sess.invalid_times.start_time.values
            iv_times_stop = sess.invalid_times.stop_time.values
            s_times = stims.start_time[stim].values
            q = []
            for i in range(len(iv_times_start)):
                q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
            stim = stim[np.where(sum(q)==0)]
    durations = stims.duration[stim]
    bins = []
    for i in range(len(durations)):
        bins.append(np.arange(0,durations.values[i],bin_step))
    stim_labels = list(stims.stimulus_condition_id[stim])
    spikes_per_stim = []
    count = 0
    for i in range(len(stim_labels)):
        spike_counts_da = sess.presentationwise_spike_counts(
            bin_edges=bins[count],
            unit_ids=decent_snr_unit_ids,
            stimulus_presentation_ids=stim_labels[i],binarize=False)
       
        spikes_per_stim.append(np.squeeze(spike_counts_da.data))
        count = count + 1
    return np.vstack(spikes_per_stim).T
    
def stim_to_one_hot(sess,stim_condition,stim_block,bin_step,inval=True):
    stims = sess.stimulus_presentations
    stim = stims[np.logical_and(stims['stimulus_name'] == stim_condition,stims['stimulus_block']==stim_block)].index.values
    if inval==True:
        if len(sess.invalid_times)>0:
            iv_times_start = sess.invalid_times.start_time.values
            iv_times_stop = sess.invalid_times.stop_time.values
            s_times = stims.start_time[stim].values
            q = []
            for i in range(len(iv_times_start)):
                q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
            stim = stim[np.where(sum(q)==0)]
    durations = stims.duration[stim]
    bins = []
    for i in range(len(durations)):
        bins.append(np.arange(0,durations.values[i],bin_step))
    stim_labels = list(stims.stimulus_condition_id[stim])
    stim_labels = np.asarray(stim_labels)-min(stim_labels)
    one_hot = []
    for i in range(len(stim_labels)):
        for j in range(len(bins[i])-1):
            one_hot.append(stim_labels[i])
    one_vect = np.zeros([np.max(stim_labels),len(one_hot)])
    for i in range(len(one_hot)):
        one_vect[one_hot[i]-1,i]=1
    return one_vect, one_hot, durations
    
    
    
    