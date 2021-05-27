import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import scipy
import neo
import matplotlib.pyplot as plt
from churchland_pipeline_python import lab, acquisition, processing
from churchland_pipeline_python.utilities import datajointutils
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

        key.update(neuron_spike_raster=spike_raster)

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
            t_bins = np.concatenate((t_new[:-1,np.newaxis], t_new[1:,np.newaxis]), axis=1).mean(axis=1)
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


# =======
# LEVEL 1
# =======

@schema
class NeuronRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rate
    -> NeuronSpikeRaster
    -> pacman_processing.FilterParams
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & NeuronSpikeRaster

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
        spike_raster_keys = (NeuronSpikeRaster & key).fetch(as_dict=True)

        # rebin spike raster to behavior time base 
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
            for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['neuron_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        neuron_rate_keys = spike_raster_keys.copy()
        [
            neuron_rate_key.update(
                filter_params_id = key['filter_params_id'],
                neuron_rate = fs_beh * filter_rel().filt(neuron_rate_key['neuron_spike_raster'], fs_beh)
            )
            for neuron_rate_key in neuron_rate_keys
        ];

        # remove spike rasters
        [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys];

        # insert neuron rates
        self.insert(neuron_rate_keys, skip_duplicates=True)


@schema
class NeuronEvenRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rates for even trials.
    -> NeuronSpikeRaster
    -> pacman_processing.FilterParams
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron \
                 * pacman_acquisition.Behavior.Condition \
                 * pacman_processing.FilterParams \
                 & NeuronSpikeRaster & (pacman_acquisition.Behavior.Trial & dj.Not('trial % 2'))

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
        spike_raster_keys = (NeuronSpikeRaster & key).fetch(as_dict=True)

        # rebin spike raster to behavior time base
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
                      for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['neuron_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        neuron_rate_keys = spike_raster_keys.copy()
        [
            neuron_rate_key.update(
                filter_params_id = key['filter_params_id'],
                neuron_rate = fs_beh * (filter_rel & filter_key).filt(neuron_rate_key['neuron_spike_raster'], fs_beh)
            )
            for neuron_rate_key in neuron_rate_keys
        ];

        # remove spike rasters
        [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys];

        # insert neuron rates
        self.insert(neuron_rate_keys, skip_duplicates=True)


@schema
class NeuronOddRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rates for odd numbered trials.
    -> NeuronSpikeRaster
    -> pacman_processing.FilterParams
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron \
                 * pacman_acquisition.Behavior.Condition \
                 * pacman_processing.FilterParams \
                 & NeuronSpikeRaster & (pacman_acquisition.Behavior.Trial & 'trial % 2')

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
        spike_raster_keys = (NeuronSpikeRaster & key).fetch(as_dict=True)

        # rebin spike raster to behavior time base
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
                      for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['neuron_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        neuron_rate_keys = spike_raster_keys.copy()
        [
            neuron_rate_key.update(
                filter_params_id = key['filter_params_id'],
                neuron_rate = fs_beh * filter_rel().filt(neuron_rate_key['neuron_spike_raster'], fs_beh)
            )
            for neuron_rate_key in neuron_rate_keys
        ];

        # remove spike rasters
        [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys];

        # insert neuron rates
        self.insert(neuron_rate_keys, skip_duplicates=True)

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
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    """

    # limit conditions with good trials
    key_source = processing.Neuron \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & NeuronRate \
        & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0),
            neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0])
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
            mean_center (bool, optional): Whether to subtract the cross-condition mean from the responses. Defaults to False.
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

        # mean-center
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
    """

    # limit conditions with good trials
    key_source = processing.Neuron \
                 * pacman_processing.AlignmentParams \
                 * pacman_processing.BehaviorBlock \
                 * pacman_processing.BehaviorQualityParams \
                 * pacman_processing.FilterParams \
                 & NeuronEvenRate \
                 & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (NeuronEvenRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0),
            neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0])
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
            mean_center (bool, optional): Whether to subtract the cross-condition mean from the responses. Defaults to False.
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

        # mean-center
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
