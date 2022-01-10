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

def open_data(sess,stim_condition,region,bin_step,bin_strt=0,snr=0,frate=0,inval=True):
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
            stim = stim[np.where(sum(q)!=1)]
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