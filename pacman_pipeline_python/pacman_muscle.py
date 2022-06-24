import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import scipy
import neo
import matplotlib.pyplot as plt
from src.churchland_pipeline_python import lab, acquisition, processing, reference
from src.churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import Any, List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_muscle')

# =======
# LEVEL 0
# =======

@schema
class Emg(dj.Imported):
    definition = """
    # raw, trialized, and aligned EMG data
    -> acquisition.EmgChannelGroup.Channel
    -> pacman_processing.EphysTrialAlignment
    ---
    emg_signal: longblob # EMG voltage signal
    """

    # process per channel group
    key_source = acquisition.EmgChannelGroup \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch channel keys
        channel_keys = (acquisition.EmgChannelGroup.Channel & key).fetch('KEY')

        # read channel indices from keys
        channel_indices = [chan_key['ephys_channel_idx'] for chan_key in channel_keys]

        # fetch ephys alignment indices
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment').astype(int)

        # fetch local ephys recording file path
        ephys_file_path = (acquisition.EphysRecording.File & key).proj_file_path().fetch1('ephys_file_path')

        # ensure local path
        ephys_file_path = reference.EngramTier.ensure_local(ephys_file_path)

        # read NSx file
        reader = neo.rawio.BlackrockRawIO(ephys_file_path)
        reader.parse_header()

        # read raw emg signals and transpose to horizontal
        emg_signals = reader.get_analogsignal_chunk(
            block_index=0, 
            seg_index=0, 
            i_start=ephys_alignment[0], 
            i_stop=1+ephys_alignment[-1], 
            channel_indexes=channel_indices
        ).T

        # update key with channel data
        keys = [dict(key, **chan_key, emg_signal=emg_signal)
            for chan_key, emg_signal in zip(channel_keys, emg_signals)]

        # insert emg signal keys
        self.insert(keys)


@schema
class MotorUnitSpikeRaster(dj.Computed):
    definition = """
    # Aligned motor unit single-trial spike raster
    -> processing.MotorUnit
    -> pacman_processing.EphysTrialAlignment
    ---
    motor_unit_spike_raster: longblob # motor unit trial-aligned spike raster (boolean array)
    """

    key_source = processing.MotorUnit \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment')

        # create spike bin edges centered around ephys alignment indices
        spike_bin_edges = np.append(ephys_alignment, ephys_alignment[-1]+1+np.arange(2)).astype(float)
        spike_bin_edges -= 0.5

        # fetch raw spike indices for the full recording
        motor_unit_spike_indices = (processing.MotorUnit & key).fetch1('motor_unit_spike_indices')

        # assign spike indices to bins
        spike_bins = np.digitize(motor_unit_spike_indices, spike_bin_edges) - 1

        # remove spike bins outside trial bounds
        spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(ephys_alignment))]

        # create trial spike raster
        spike_raster = np.zeros(len(ephys_alignment), dtype=bool)
        spike_raster[spike_bins] = 1

        key.update(motor_unit_spike_raster=spike_raster)

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
            raster_keys, spike_rasters = (self & cond_key).fetch('KEY', 'motor_unit_spike_raster')
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
class EmgEnvelope(dj.Computed):
    definition = """
    # Rectified and filtered EMG data
    -> Emg
    -> pacman_processing.FilterParams
    ---
    emg_envelope: longblob # EMG envelope
    """

    def make(self, key):

        # fetch ephys sample rates
        fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

        # fetch condition time (behavior time base)
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # make condition time in ephys time base
        t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

        # fetch raw emg data
        emg_attributes = (Emg & key).fetch(as_dict=True)

        # highpass filter and rectify raw emg signals
        [emg_attr.update(emg_envelope=abs(processing.Filter.Butterworth().filt(emg_attr['emg_signal'], fs_ephys, order=2, low_cut=40)))
            for emg_attr in emg_attributes];

        # remove raw emg signal
        [emg_attr.pop('emg_signal') for emg_attr in emg_attributes];

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # smooth rectified emg signals to construct envelope and remove 
        [emg_attr.update(
            filter_params_id=key['filter_params_id'],
            emg_envelope=filter_rel().filt(emg_attr['emg_envelope'], fs_ephys)
        ) for emg_attr in emg_attributes];

        # resample emg to behavior time base
        [emg_attr.update(
            emg_envelope=np.interp(t_beh, t_ephys, emg_attr['emg_envelope'])
        ) for emg_attr in emg_attributes];

        # insert emg envelopes
        self.insert(emg_attributes)


@schema
class MotorUnitRate(dj.Computed):
    definition = """
    # Aligned motor unit single-trial firing rate
    -> MotorUnitSpikeRaster
    -> pacman_processing.FilterParams
    ---
    motor_unit_rate: longblob # motor unit trial-aligned firing rate (spikes/s)
    """

    # process per motor unit/condition
    key_source = processing.MotorUnit \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & MotorUnitSpikeRaster

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
        spike_raster_keys = (MotorUnitSpikeRaster & key).fetch(as_dict=True)

        # rebin spike raster to behavior time base 
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['motor_unit_spike_raster']], time_bin_edges) - 1 \
            for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['motor_unit_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['motor_unit_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        motor_unit_rate_keys = spike_raster_keys.copy()
        [
            motor_unit_rate_key.update(
                filter_params_id = key['filter_params_id'],
                motor_unit_rate = fs_beh * filter_rel().filt(motor_unit_rate_key['motor_unit_spike_raster'], fs_beh)
            )
            for motor_unit_rate_key in motor_unit_rate_keys
        ];

        # remove spike rasters
        [motor_unit_rate_key.pop('motor_unit_spike_raster') for motor_unit_rate_key in motor_unit_rate_keys];

        # insert motor unit rates
        self.insert(motor_unit_rate_keys, skip_duplicates=True)


# =======
# LEVEL 2
# =======

@schema
class EmgEnvelopeMean(dj.Computed):
    definition = """
    # Trial-averaged rectified and filtered EMG data
    -> acquisition.EmgChannelGroup.Channel
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    emg_envelope_mean: longblob # trial-averaged EMG envelope (au)
    emg_envelope_sem:  longblob # EMG envelope standard error across trials (au)
    """

    # limit conditions with good trials
    key_source = acquisition.EmgChannelGroup.Channel \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & EmgEnvelope \
        & (pacman_processing.GoodTrial & 'good_trial')

    def make(self, key):

        # fetch single-trial emg envelopes
        emg_envelope = (EmgEnvelope & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('emg_envelope')
        emg_envelope = np.stack(emg_envelope)

        # update key with Mean and standard error
        key.update(
            emg_envelope_mean=emg_envelope.mean(axis=0),
            emg_envelope_sem=emg_envelope.std(axis=0, ddof=(1 if emg_envelope.shape[0] > 1 else 0))/np.sqrt(emg_envelope.shape[0])
        )

        # insert emg envelope means
        self.insert1(key)


    def fetch_emgs(
        self,
        fs: int=None,
        soft_normalize: int=None,
        mean_center: bool=False,
        output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch trial-averaged EMG envelopes.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples EMGs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the signal range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options: 
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per emg channel/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            emgs (Any): EMGs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            emg_channel_keys (List[dict]): List of emg channel keys in the dataset
        """

        # ensure that there is one EMG envelop per channel/condition
        emg_condtion_keys = acquisition.EmgChannelGroup.Channel.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(emg_condtion_keys))
        
        n_emgs_per_condition = dj.U(*emg_condtion_keys).aggr(self, count='count(*)')
        assert not(n_emgs_per_condition & 'count > 1'), 'More than one EMG per emg channel and condition. Check ' \
            + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get emg channel keys
        emg_channel_keys = (acquisition.EmgChannelGroup.Channel & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('emg_envelope_mean')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join emg table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            emgs = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch emg data
                emg_data = [(self & cond_key & chan_key).fetch1() for chan_key in emg_channel_keys]

                # interpolate emgs to new timebase as needed
                if fs is not None:
                    [X.update(emg_envelope_mean=np.interp(t_new, X['t_old'], X['emg_envelope_mean'])) for X in emg_data];

                # extract emgs and append to list
                emgs.append(np.array([X['emg_envelope_mean'] for X in emg_data]))

        else:
            # fetch emgs and stack across units
            emgs = []
            for cond_key in condition_keys:
                emgs.append(np.stack(
                    [(self & cond_key & chan_key).fetch1('emg_envelope_mean') for chan_key in emg_channel_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, emgs)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            signal_range = np.hstack(emgs).ptp(axis=1, keepdims=True)
            emgs = [X/(signal_range + soft_normalize) for X in emgs]

        # Mean-center
        if mean_center:
            signal_mean = np.hstack(emgs).mean(axis=1, keepdims=True)
            emgs = [X - signal_mean for X in emgs]
        
        # format output
        if output_format == 'array':

            # stack output across conditions and times
            emgs = np.hstack(emgs)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            emg_data = []
            for cond_key, X in zip(condition_keys, emgs):
                for chan_key, Xi in zip(emg_channel_keys, X):
                    emg_data.append(dict(cond_key, **chan_key, emg_envelope_mean=Xi))

            emgs = emg_data

        return emgs, condition_ids, condition_times, condition_keys, emg_channel_keys


@schema
class MotorUnitPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.MotorUnit
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    motor_unit_psth:     longblob # motor unit trial-averaged firing rate (spikes/s)
    motor_unit_psth_sem: longblob # motor unit firing rate standard error (spikes/s)
    motor_unit_psth_var: longblob # motor unit firing rate variance (spikes/s)**2
    """

    # limit conditions with good trials
    key_source = processing.MotorUnit \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & MotorUnitRate \
        & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (MotorUnitRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('motor_unit_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            motor_unit_psth=rates.mean(axis=0),
            motor_unit_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0]),
            motor_unit_psth_var=rates.var(axis=0),
        )

        # insert motor unit PSTH
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
                * 'dict' (list of dictionaries per motor unit/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            motor_chan_keys (List[dict]): List of motor unit keys in the dataset
        """

        # ensure that there is one PSTH per motor unit/condition
        motor_unit_condtion_keys = processing.MotorUnit.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(motor_unit_condtion_keys))
        
        n_psths_per_condition = dj.U(*motor_unit_condtion_keys).aggr(self, count='count(*)')
        assert not(n_psths_per_condition & 'count > 1'), 'More than one PSTH per motor unit and condition. Check ' \
            + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get motor unit keys
        motor_chan_keys = (processing.MotorUnit & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('motor_unit_psth')

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
                psth_data = [(self & cond_key & chan_key).fetch1() for chan_key in motor_chan_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(motor_unit_psth=np.interp(t_new, X['t_old'], X['motor_unit_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['motor_unit_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & chan_key).fetch1('motor_unit_psth') for chan_key in motor_chan_keys]
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
                for chan_key, Xi in zip(motor_chan_keys, X):
                    psth_data.append(dict(cond_key, **chan_key, motor_unit_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, motor_chan_keys