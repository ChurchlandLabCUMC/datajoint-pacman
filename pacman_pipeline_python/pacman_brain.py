import pdb

import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import scipy
import src.neo as neo
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from src.churchland_pipeline_python import lab, acquisition, processing
from src.churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import Any, List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_brain')
# =======
# LEVEL 0
# =======


@schema
class NeuronSpikeRaster(dj.Computed):
    definition = """
    # Aligned neuron single-trial spike raster
    -> processing.Neuron
    -> pacman_processing.EphysTrialAlignment
    ---
    neuron_spike_raster: longblob # neuron trial-aligned spike raster (boolean array)
    neuron_spike_count: int # number of spikes in trial
    """

    key_source = processing.Neuron \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):
        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment')

        # create spike bin edges centered around ephys alignment indices
        spike_bin_edges = np.append(ephys_alignment, ephys_alignment[-1]+1+np.arange(2)).astype(float)
        spike_bin_edges -= 0.5

        # fetch raw spike indices for the full recording
        neuron_spike_indices = (processing.Neuron & key).fetch1('neuron_spike_indices')

        # assign spike indices to bins
        spike_bins = np.digitize(neuron_spike_indices, spike_bin_edges) - 1
        # remove spike bins outside trial bounds
        spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(ephys_alignment))]

        # create trial spike raster
        spike_raster = np.zeros(len(ephys_alignment), dtype=bool)
        spike_raster[spike_bins] = 1

        key.update(neuron_spike_raster=spike_raster, neuron_spike_count=spike_raster.sum())

        # insert spike raster
        self.insert1(key)

    
    def rebin(
        self, 
        fs: int=None, 
        as_raster: bool=False, 
        order_by: str=None
        ) -> (List[dict], np.ndarray):
        """Rebin spike rasters.

        Args:
            fs (int, optional): New sample rate. Defaults to behavior sample rate.
            as_raster (bool, optional): If True, returns output as raster. If False, returns spike indices. Defaults to False.
            order_by (str, optional): Attribute used to order the returned data. If None, returns the data without additional sorting. 

        Returns:
            keys (list): List of key dictionaries to identify each set of rebinned spikes with the original table entry
            spikes (np.ndarray): Array of rebinned spike indices or rasters, depending on as_raster value
        """

        # fetch behavior condition keys (retain only condition time vectors)
        condition_keys = (pacman_acquisition.Behavior.Condition & self).proj('condition_time').fetch(as_dict=True)

        # initialize list of spikes and keys
        keys = []
        spikes = []

        # loop unique conditions
        for cond_key in condition_keys:

            # new time vector (behavior time base by default)
            if fs is None:
                fs_new = (acquisition.BehaviorRecording & cond_key).fetch1('behavior_recording_sample_rate')
                t_new = cond_key['condition_time']
            else:
                fs_new = fs
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs_new)

            # create time bins in new time base
            t_bins = np.concatenate((t_new[:-1,np.newaxis], t_new[1:,np.newaxis]), axis=1).Mean(axis=1)
            t_bins = np.insert(t_bins, 0, t_new[0]-1/(2*fs_new))
            t_bins = np.append(t_bins, t_new[-1]+1/(2*fs_new))

            # resample time vector to ephys time base
            fs_ephys = (acquisition.EphysRecording & cond_key).fetch1('ephys_recording_sample_rate')
            t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs_ephys)

            # rebin spike rasters to new time base
            raster_keys, spike_rasters = (self & cond_key).fetch('KEY', 'neuron_spike_raster')
            new_spikes = [np.digitize(t_ephys[raster], t_bins) - 1 for raster in spike_rasters]

            # convert spike indices to raster
            if as_raster:
                new_spikes = [[True if i in spk_idx else False for i in range(len(t_new))] for spk_idx in new_spikes]

            # append spike rasters and keys to list
            keys.extend(raster_keys)
            spikes.extend(new_spikes)

        # order the data
        if order_by is not None:

            # extract the ordering attribute from the keys
            order_attr = [key[order_by] for key in keys]

            # sort the spike data
            spike_data = [(key, spk) for _, key, spk in sorted(zip(order_attr, keys, spikes))]

            # unpack the keys and spike indices as 
            keys, spikes = map(list, zip(*spike_data))

        return keys, np.array(spikes)

@schema
class NeuronIndex(dj.Manual):
    definition = """
    unit_id = 0 : smallint unsigned # unit id in psth xarray
    ephys_file_id = 0 : smallint unsigned #file id for unit
    neuron_id : smallint unsigned # unit id in datajoint table
    session_date : date
    """


@schema #TODO change this to inherit neuron_id from NeuronIndex primary key
class NeuronJitter(dj.Computed):
    definition = """
    -> NeuronIndex
    -> pacman_processing.ShuffleIds
    jitter: smallint # amount that spike times are jittered by
    ---
    neuron_id: smallint unsigned
    session_date: date
    """

    def make(self, key):
        jitter_amount = np.random.randint(-5000, 5000)
        neuron_id, date = (NeuronIndex & key).fetch1('neuron_id', 'session_date')
        # date = (NeuronIndex & key).fetch1('neuron_id', 'session_date')
        key.update(jitter=jitter_amount, neuron_id=neuron_id, session_date=date)
        self.insert1(key)


@schema #TODO modify to inherit elements from NeuronJitter
class NeuronJitterSpikes(dj.Computed):
    definition = """
    # Aligned jittered neuron single-trial spike raster
    -> processing.Neuron
    -> pacman_processing.EphysTrialAlignment
    -> NeuronJitter
    ---
    neuron_spike_raster: longblob # neuron trial-aligned spike raster (boolean array)
    neuron_spike_count: int # number of spikes in trial
    """

    key_source = processing.Neuron \
                 * pacman_processing.EphysTrialAlignment \
                 & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment') &  NeuronJitter

    def make(self, key):
        # fetch ephys alignment indices for the current trial
        jitter_amount, shuffle_id, unit_id = (NeuronJitter & key).fetch1('jitter', 'shuffle_id', 'unit_id')
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment')

        # create spike bin edges centered around ephys alignment indices
        spike_bin_edges = np.append(ephys_alignment, ephys_alignment[-1] + 1 + np.arange(2)).astype(float)
        spike_bin_edges -= 0.5

        # fetch raw spike indices for the full recording
        neuron_spike_indices = (processing.Neuron & key).fetch1('neuron_spike_indices')
        jittered_spike_indices = neuron_spike_indices + jitter_amount
        # assign spike indices to bins
        spike_bins = np.digitize(jittered_spike_indices, spike_bin_edges) - 1
        # remove spike bins outside trial bounds
        spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(ephys_alignment))]

        # create trial spike raster
        spike_raster = np.zeros(len(ephys_alignment), dtype=bool)
        spike_raster[spike_bins] = 1



        key.update(neuron_spike_raster=spike_raster, neuron_spike_count=spike_raster.sum(), jitter=jitter_amount, shuffle_id=shuffle_id,
                   unit_id=unit_id)

        # insert spike raster

        self.insert1(key)

    def rebin(
            self,
            fs: int = None,
            as_raster: bool = False,
            order_by: str = None
    ) -> (List[dict], np.ndarray):
        """Rebin spike rasters.

        Args:
            fs (int, optional): New sample rate. Defaults to behavior sample rate.
            as_raster (bool, optional): If True, returns output as raster. If False, returns spike indices. Defaults to False.
            order_by (str, optional): Attribute used to order the returned data. If None, returns the data without additional sorting.

        Returns:
            keys (list): List of key dictionaries to identify each set of rebinned spikes with the original table entry
            spikes (np.ndarray): Array of rebinned spike indices or rasters, depending on as_raster value
        """

        # fetch behavior condition keys (retain only condition time vectors)
        condition_keys = (pacman_acquisition.Behavior.Condition & self).proj('condition_time').fetch(as_dict=True)

        # initialize list of spikes and keys
        keys = []
        spikes = []

        # loop unique conditions
        for cond_key in condition_keys:

            # new time vector (behavior time base by default)
            if fs is None:
                fs_new = (acquisition.BehaviorRecording & cond_key).fetch1('behavior_recording_sample_rate')
                t_new = cond_key['condition_time']
            else:
                fs_new = fs
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs_new)

            # create time bins in new time base
            t_bins = np.concatenate((t_new[:-1, np.newaxis], t_new[1:, np.newaxis]), axis=1).Mean(axis=1)
            t_bins = np.insert(t_bins, 0, t_new[0] - 1 / (2 * fs_new))
            t_bins = np.append(t_bins, t_new[-1] + 1 / (2 * fs_new))

            # resample time vector to ephys time base
            fs_ephys = (acquisition.EphysRecording & cond_key).fetch1('ephys_recording_sample_rate')
            t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs_ephys)

            # rebin spike rasters to new time base
            raster_keys, spike_rasters = (self & cond_key).fetch('KEY', 'neuron_spike_raster')
            new_spikes = [np.digitize(t_ephys[raster], t_bins) - 1 for raster in spike_rasters]

            # convert spike indices to raster
            if as_raster:
                new_spikes = [[True if i in spk_idx else False for i in range(len(t_new))] for spk_idx in new_spikes]

            # append spike rasters and keys to list
            keys.extend(raster_keys)
            spikes.extend(new_spikes)

        # order the data
        if order_by is not None:
            # extract the ordering attribute from the keys
            order_attr = [key[order_by] for key in keys]

            # sort the spike data
            spike_data = [(key, spk) for _, key, spk in sorted(zip(order_attr, keys, spikes))]

            # unpack the keys and spike indices as
            keys, spikes = map(list, zip(*spike_data))

        return keys, np.array(spikes)


@schema
class StabilityThreshold(dj.Manual):
    definition = """
    trial_stability_threshold = .5 : float # lower values correspond to outlier trials
    """

# =======
# LEVEL 1
# =======
@schema
class NeuronStability(dj.Computed):
    definition = """
    # neural stability measure
    -> NeuronSpikeRaster
    -> NeuronIndex
    ___
    neuron_stability : float # overall neuron stability 
    trial_stability : float # individual trial stability
    """
    _key_source = NeuronIndex.proj(
    'session_date', 'neuron_id'
    )
    def make(self, key, std_bin: int = 15):
        neuron_table = (NeuronSpikeRaster() * NeuronIndex) & key
        condition_id, alignment_id = neuron_table.fetch('condition_id', 'alignment_params_id')
        condition_id = np.asarray(condition_id)
        conditions = np.unique(neuron_table.fetch('condition_id'))
        trial_numbers = [None] * conditions.shape[0]
        spikes_in = [None] * conditions.shape[0]
        for idx, condition in enumerate(conditions):
            spikes, trial_number = (neuron_table & {'condition_id': condition}).fetch('neuron_spike_raster', 'trial')
            spikes = np.stack(spikes)
            spikes_per_trial = spikes.sum(axis=1)
            spikes_in[idx] = spikes_per_trial
            trial_numbers[idx] = trial_number
        trial_numbers = np.hstack(trial_numbers)[:, np.newaxis]
        spikes_in = np.hstack(spikes_in)[:, np.newaxis]
        moving_std = np.zeros(trial_numbers.shape[0])
        ordered_trial_idx = np.argsort(trial_numbers, axis=0)
        # align_sorted = alignment_id[ordered_trial_idx]
        condition_sorted = condition_id[ordered_trial_idx]
        ordered_trials = np.sort(trial_numbers, axis=0)
        ordered_spikes = spikes_in[ordered_trial_idx, 0]
        moving_std_insert = np.asarray([ordered_spikes[i:i + std_bin].std()
                                 for i in range(trial_numbers.shape[0] - std_bin)])
        moving_std[:-std_bin] = moving_std_insert
        # for start in range(trial_numbers.shape[0] - std_bin):
        #     moving_std[start] = ordered_spikes[start: start + std_bin].std()
        moving_std[-std_bin:] = moving_std[-2*std_bin: -std_bin]
        moving_std = gaussian_filter1d(moving_std, sigma=std_bin)
        ys = ordered_spikes - ordered_spikes.mean(axis=0)
        xs = ordered_trials - ordered_trials.mean(axis=0)
        unweighted_slope = ((np.linalg.inv(xs.T @ xs)) @ xs.T @ ys)[0, 0]
        probabilities = np.exp(-1/2 * (ys[:, 0] / moving_std)**2)
        stability_score = np.abs(unweighted_slope)/ ordered_spikes.mean()
        key.update({'neuron_stability': stability_score})
        small_table = ((neuron_table) & {'trial': ordered_trials[0, 0]})
        fetched_dict = small_table.fetch(as_dict=True)[0]
        restricted_dict = {key_name: fetched_dict[key_name] for key_name in self.primary_key}
        key.update(restricted_dict)
        key.pop('condition_id')
        all_keys = [dict(key, **{'trial_stability': probability, 'trial': trial, 'condition_id': condition}) for probability, trial, condition in
                    zip(probabilities, ordered_trials[:, 0], condition_sorted[:, 0])]
        self.insert(all_keys, skip_duplicates=True)
        # pdb.set_trace()
        # for probability, trial in zip( probabilities, ordered_trials[:, 0]):
        #     key.update({'trial': trial})
        #     small_table = ((neuron_table) & {'trial': trial})
        #     fetched_dict = small_table.fetch(as_dict=True)[0]
        #     restricted_dict = { key_name: fetched_dict[key_name] for key_name in self.primary_key}
        #     pdb.set_trace()
        #     key.update(restricted_dict)
        #     key.update({'trial_stability': probability})
        #     self.insert1(key, skip_duplicates=True)

# =======
# LEVEL 2
# =======

@schema
class NeuronRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rate
    -> NeuronSpikeRaster
    -> NeuronStability
    -> StabilityThreshold
    -> pacman_processing.FilterParams
    valid_data = 1 : tinyint unsigned # whether this split actually contains data
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    neuron_stability : float # overall neuron stability 
    trial_stability : float # individual trial stability
    """

    # process per neuron/condition
    key_source = processing.Neuron * StabilityThreshold \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & NeuronSpikeRaster

    def make(self, key):
        trial_stab_thresh = key['trial_stability_threshold']
        # fetch behavior and ephys sample rates
        fs_beh = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))
        fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

        # fetch condition time (behavior time base)
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')
        n_samples = len(t_beh)

        # make condition time in ephys time base
        t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

        # fetch spike rasters (ephys time base)
        spike_raster_keys = (((NeuronSpikeRaster * NeuronStability
                              ) & key & f'trial_stability >= {trial_stab_thresh}'
                             ) * StabilityThreshold ).fetch(as_dict=True)

        if spike_raster_keys:
            # rebin spike raster to behavior time base
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
                for spike_raster_key in spike_raster_keys]
            spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

            for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
                spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
                spike_raster_key['neuron_spike_raster'][spk_bins] = 1

            # get filter kernel
            filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1()
            filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
            filter_part = next(part for part in filter_parts if part & filter_key)
            (filter_part & filter_key)
            # filter rebinned spike raster
            neuron_rate_keys = spike_raster_keys.copy()
            [
                neuron_rate_key.update(
                    filter_params_id = key['filter_params_id'],
                    neuron_rate = fs_beh * (filter_part & filter_key).filt(y=neuron_rate_key['neuron_spike_raster'],
                                                                           fs=fs_beh),
                )
                for neuron_rate_key in neuron_rate_keys
            ]
            # remove spike rasters
            [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys]

            # insert neuron rates
            self.insert(neuron_rate_keys, skip_duplicates=True)
        else:
            spike_raster_keys = ((NeuronSpikeRaster * NeuronStability
                                   * StabilityThreshold) & key).fetch(as_dict=True)
            # rebin spike raster to behavior time base
            time_bin_edges = np.append(t_beh, t_beh[-1] + (1 + np.arange(2)) / fs_beh) - 1 / (2 * fs_beh)
            spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
                          for spike_raster_key in spike_raster_keys]
            spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

            for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
                spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
                spike_raster_key['neuron_spike_raster'][spk_bins] = 1

            # get filter kernel
            filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1()
            filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
            filter_part = next(part for part in filter_parts if part & filter_key)
            (filter_part & filter_key)
            # filter rebinned spike raster
            neuron_rate_keys = spike_raster_keys.copy()
            [
                neuron_rate_key.update(
                    filter_params_id=key['filter_params_id'],
                    neuron_rate=np.nan,
                    valid_data=0,
                )
                for neuron_rate_key in neuron_rate_keys
            ]
            # remove spike rasters
            [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys]

            # insert neuron rates
            self.insert(neuron_rate_keys, skip_duplicates=True)



@schema
class JitterRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rate
    -> NeuronJitterSpikes
    -> pacman_processing.FilterParams
    -> pacman_processing.ShuffleIds
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron * pacman_processing.ShuffleIds\
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & NeuronJitterSpikes

    def make(self, key):
        # fetch behavior and ephys sample rates
        fs_beh = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))
        fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

        # fetch condition time (behavior time base)
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')
        n_samples = len(t_beh)

        # make condition time in ephys time base
        t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

        # fetch spike rasters (ephys time base)
        spike_raster_keys = (NeuronJitterSpikes & key).fetch(as_dict=True)

        if spike_raster_keys:
            # rebin spike raster to behavior time base
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
                for spike_raster_key in spike_raster_keys]
            spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

            for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
                spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
                spike_raster_key['neuron_spike_raster'][spk_bins] = 1

            # get filter kernel
            filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1()
            filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
            filter_part = next(part for part in filter_parts if part & filter_key)
            (filter_part & filter_key)
            # filter rebinned spike raster
            neuron_rate_keys = spike_raster_keys.copy()
            [
                neuron_rate_key.update(
                    filter_params_id = key['filter_params_id'],
                    neuron_rate = fs_beh * (filter_part & filter_key).filt(y=neuron_rate_key['neuron_spike_raster'],
                                                                           fs=fs_beh),
                )
                for neuron_rate_key in neuron_rate_keys
            ]
            # remove spike rasters
            [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys]
            [neuron_rate_key.pop('neuron_spike_count') for neuron_rate_key in neuron_rate_keys]

            # insert neuron rates
            self.insert(neuron_rate_keys)
        else:
            print('No spike rasters')

# =======
# LEVEL 2
# =======


@schema
class NeuronPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.Neuron
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    -> StabilityThreshold    
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    neuron_psth_var: longblob # neuron firing rate variance (spikes/s)^2
    trial_count: smallint unsigned
    """

    key_source = (processing.Neuron * pacman_processing.AlignmentParams * pacman_processing.BehaviorBlock
        * pacman_processing.BehaviorQualityParams
        * pacman_processing.FilterParams
        * StabilityThreshold) & ((NeuronRate * pacman_processing.GoodTrial) & 'good_trial' & 'valid_data')

    def make(self, key):
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        if not len(rates) == 0:
            rates = np.stack(rates)
            if not np.isnan(rates).any():
                # update key with psth and standard error
                key.update(
                    neuron_psth=rates.mean(axis=0),
                    neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0]),
                    neuron_psth_var=rates.var(axis=0),
                    trial_count=rates.shape[0]
                )

                # insert neuron PSTH
                self.insert1(key)
            else:
                pdb.set_trace()


    def fetch_psths(
        self,
        fs: int=None,
        soft_normalize: int=None,
        mean_center: bool=False,
        output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch PSTHs.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples PSTHs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the firing rate range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options: 
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per neuron/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            neuron_keys (List[dict]): List of neuron keys in the dataset
        """

        # ensure that there is one PSTH per neuron/condition
        neuron_condtion_keys = processing.Neuron.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(neuron_condtion_keys))
        
        n_psths_per_condition = dj.U(*neuron_condtion_keys).aggr(self, count='count(*)')
        assert not(n_psths_per_condition & 'count > 1'), 'More than one PSTH per neuron and condition. Check ' \
            + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get neuron keys
        neuron_keys = (processing.Neuron & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('neuron_psth')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join psth table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            psths = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch psth data
                psth_data = [(self & cond_key & unit_key).fetch1() for unit_key in neuron_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(neuron_psth=np.interp(t_new, X['t_old'], X['neuron_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['neuron_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & unit_key).fetch1('neuron_psth') for unit_key in neuron_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, psths)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            rate_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X/(rate_range + soft_normalize) for X in psths]

        # Mean-center
        if mean_center:
            rate_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psths = [X - rate_mean for X in psths]
        
        # format output
        if output_format == 'array':

            # stack output across conditions and times
            psths = np.hstack(psths)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            psth_data = []
            for cond_key, X in zip(condition_keys, psths):
                for unit_key, Xi in zip(neuron_keys, X):
                    psth_data.append(dict(cond_key, **unit_key, neuron_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, neuron_keys


@schema
class NeuronJitterPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.Neuron
    -> NeuronJitter
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    """

    key_source = (processing.Neuron * NeuronJitter * pacman_processing.AlignmentParams * pacman_processing.BehaviorBlock
                  * pacman_processing.BehaviorQualityParams
                  * pacman_processing.FilterParams) & ((JitterRate * pacman_processing.GoodTrial) & 'good_trial')

    def make(self, key):
        rates = (JitterRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        try:
            rates = np.stack(rates)
        except:
            pdb.set_trace()

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0)
        )

        # insert neuron PSTH
        self.insert1(key)

    def fetch_psths(
            self,
            fs: int = None,
            soft_normalize: int = None,
            mean_center: bool = False,
            output_format: str = 'array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch PSTHs.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples PSTHs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the firing rate range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options:
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per neuron/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            neuron_keys (List[dict]): List of neuron keys in the dataset
        """

        # ensure that there is one PSTH per neuron/condition
        neuron_condtion_keys = processing.Neuron.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(neuron_condtion_keys))

        n_psths_per_condition = dj.U(*neuron_condtion_keys).aggr(self, count='count(*)')
        assert not (n_psths_per_condition & 'count > 1'), 'More than one PSTH per neuron and condition. Check ' \
                                                          + (', '.join(['{}'] * len(remaining_keys))).format(
            *remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self,
                                                                                    include=['label', 'rank', 'time',
                                                                                             'force'])

        # get neuron keys
        neuron_keys = (processing.Neuron & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('neuron_psth')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join psth table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            psths = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch psth data
                psth_data = [(self & cond_key & unit_key).fetch1() for unit_key in neuron_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(neuron_psth=np.interp(t_new, X['t_old'], X['neuron_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['neuron_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & unit_key).fetch1('neuron_psth') for unit_key in neuron_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'],) * X.shape[1] for cond_key, X in zip(condition_keys, psths)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            rate_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X / (rate_range + soft_normalize) for X in psths]

        # Mean-center
        if mean_center:
            rate_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psths = [X - rate_mean for X in psths]

        # format output
        if output_format == 'array':

            # stack output across conditions and times
            psths = np.hstack(psths)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            psth_data = []
            for cond_key, X in zip(condition_keys, psths):
                for unit_key, Xi in zip(neuron_keys, X):
                    psth_data.append(dict(cond_key, **unit_key, neuron_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, neuron_keys


@schema
class NeuronEvenPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram generated from even trials
    -> processing.Neuron
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    neuron_psth_var: longblob # neuron firing rate variance (spikes/s)**2
    """

    # limit conditions with good trials
    key_source = processing.Neuron \
                 * pacman_processing.AlignmentParams \
                 * pacman_processing.BehaviorBlock \
                 * pacman_processing.BehaviorQualityParams \
                 * pacman_processing.FilterParams \
                 & (NeuronRate() - 'trial % 2') \
                 & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        if rates:
            rates = np.stack(rates)

            # update key with psth and standard error
            key.update(
                neuron_psth=rates.mean(axis=0),
                neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0]),
                neuron_psth_var=rates.var(axis=0)
            )

            # insert neuron PSTH
            self.insert1(key)


    def fetch_psths(
            self,
            fs: int=None,
            soft_normalize: int=None,
            mean_center: bool=False,
            output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch PSTHs.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples PSTHs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the firing rate range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options:
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per neuron/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            neuron_keys (List[dict]): List of neuron keys in the dataset
        """

        # ensure that there is one PSTH per neuron/condition
        neuron_condtion_keys = processing.Neuron.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(neuron_condtion_keys))

        n_psths_per_condition = dj.U(*neuron_condtion_keys).aggr(self, count='count(*)')
        assert not(n_psths_per_condition & 'count > 1'), 'More than one PSTH per neuron and condition. Check ' \
                                                         + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get neuron keys
        neuron_keys = (processing.Neuron & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('neuron_psth')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join psth table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            psths = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch psth data
                psth_data = [(self & cond_key & unit_key).fetch1() for unit_key in neuron_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(neuron_psth=np.interp(t_new, X['t_old'], X['neuron_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['neuron_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & unit_key).fetch1('neuron_psth') for unit_key in neuron_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, psths)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            rate_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X/(rate_range + soft_normalize) for X in psths]

        # Mean-center
        if mean_center:
            rate_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psths = [X - rate_mean for X in psths]

        # format output
        if output_format == 'array':

            # stack output across conditions and times
            psths = np.hstack(psths)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            psth_data = []
            for cond_key, X in zip(condition_keys, psths):
                for unit_key, Xi in zip(neuron_keys, X):
                    psth_data.append(dict(cond_key, **unit_key, neuron_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, neuron_keys


@schema
class NeuronOddPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram generated from odd trials
    -> processing.Neuron
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    neuron_psth_var: longblob # neuron firing rate variance (spikes/s)**2
    """

    # limit conditions with good trials
    key_source = processing.Neuron \
                 * pacman_processing.AlignmentParams \
                 * pacman_processing.BehaviorBlock \
                 * pacman_processing.BehaviorQualityParams \
                 * pacman_processing.FilterParams \
                 & (NeuronRate() & 'trial % 2') \
                 & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0),
            neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0]),
            neuron_psth_var=rates.var(axis=0)
        )

        # insert neuron PSTH
        self.insert1(key, skip_duplicates = True)


    def fetch_psths(
            self,
            fs: int=None,
            soft_normalize: int=None,
            mean_center: bool=False,
            output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch PSTHs.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples PSTHs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the firing rate range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options:
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per neuron/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            neuron_keys (List[dict]): List of neuron keys in the dataset
        """

        # ensure that there is one PSTH per neuron/condition
        neuron_condtion_keys = processing.Neuron.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(neuron_condtion_keys))

        n_psths_per_condition = dj.U(*neuron_condtion_keys).aggr(self, count='count(*)')
        assert not(n_psths_per_condition & 'count > 1'), 'More than one PSTH per neuron and condition. Check ' \
                                                         + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get neuron keys
        neuron_keys = (processing.Neuron & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('neuron_psth')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join psth table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            psths = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch psth data
                psth_data = [(self & cond_key & unit_key).fetch1() for unit_key in neuron_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(neuron_psth=np.interp(t_new, X['t_old'], X['neuron_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['neuron_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & unit_key).fetch1('neuron_psth') for unit_key in neuron_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, psths)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            rate_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X/(rate_range + soft_normalize) for X in psths]

        # Mean-center
        if mean_center:
            rate_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psths = [X - rate_mean for X in psths]

        # format output
        if output_format == 'array':

            # stack output across conditions and times
            psths = np.hstack(psths)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            psth_data = []
            for cond_key, X in zip(condition_keys, psths):
                for unit_key, Xi in zip(neuron_keys, X):
                    psth_data.append(dict(cond_key, **unit_key, neuron_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, neuron_keys


@schema
class NeuronPsthShuffled(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.Neuron
    -> pacman_processing.ReliabilityShuffleIds
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    -> StabilityThreshold    
    ---
    neuron_psth_group1:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_group2:     longblob # neuron trial-averaged firing rate (spikes/s)
    trial_count: smallint unsigned # number of trials
    """

    key_source = (processing.Neuron * pacman_processing.AlignmentParams * pacman_processing.BehaviorBlock
                  * pacman_processing.BehaviorQualityParams
                  * pacman_processing.FilterParams
                  * StabilityThreshold * pacman_processing.ReliabilityShuffleIds) & (
            (NeuronRate * pacman_processing.GoodTrial) & 'good_trial' & 'valid_data')

    def make(self, key):
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        if not len(rates) == 0:
            rates = np.stack(rates)
            if not np.isnan(rates).any():

                n_trials = rates.shape[0]
                indices = np.random.permutation(np.arange(n_trials))
                if n_trials > 15:
                    g1_idx = indices[::2]
                    g2_idx = indices[1::2]
                elif n_trials >1:
                    g1_idx = np.random.choice(np.arange(n_trials), round(n_trials/2))
                    g2_idx = np.random.choice(np.arange(n_trials), round(n_trials/2))
                else:
                    g1_idx = np.arange(n_trials)
                    g2_idx = np.arange(n_trials)
                rates1 = rates[g1_idx]
                rates2 = rates[g2_idx]

                key.update(
                    neuron_psth_group1=rates1.mean(axis=0),
                    neuron_psth_group2=rates2.mean(axis=0),
                    trial_count=round(n_trials/2)
                )

                self.insert1(key, skip_duplicates='True')

        else:
            pdb.set_trace()


@schema
class NeuronReliability(dj.Manual):
    definition = """
            # Peri-stimulus time histogram generated from even trials
            -> lab.Monkey
            -> pacman_processing.ShuffleIds 
            nconditions = 12: smallint unsigned
            grouped = 0: tinyint unsigned       
            sample = 0: tinyint unsigned
            inverted_gain = 0: tinyint
            conditions: varchar(255)
            ---
            reliability:     longblob # neuron trial-averaged firing rate (spikes/s)
            """
