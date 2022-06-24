import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from src.churchland_pipeline_python import lab, acquisition, processing
from src.churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import Any, List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_behavior')

# =======
# LEVEL 0
# =======

@schema
class Force(dj.Computed):
    definition = """
    # Single trial force
    -> pacman_processing.BehaviorTrialAlignment
    -> pacman_processing.FilterParams
    ---
    force_raw:  longblob # raw (online), aligned force signal (V)
    force_filt: longblob # filtered, aligned, and calibrated force (N)
    """

    # batch process trials
    key_source = pacman_acquisition.Behavior.Condition \
        * pacman_processing.AlignmentParams \
        * pacman_processing.FilterParams \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # trial source
        trial_source = (pacman_processing.BehaviorTrialAlignment & 'valid_alignment') \
            * pacman_processing.FilterParams & key

        # convert raw force signal to Newtons
        trial_rel = pacman_acquisition.Behavior.Trial & trial_source
        force_data = trial_rel.process_force(data_type='raw', apply_filter=False, keep_keys=True)

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter raw data
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        [frc.update(force_filt_offline=filter_rel().filt(frc['force_raw_online'], fs)) for frc in force_data];

        # fetch alignment indices and cast as integers
        behavior_alignment = (trial_source).fetch('behavior_alignment', order_by='trial')
        behavior_alignment = list(map(lambda x: x.astype(int), behavior_alignment))

        # append key and align raw and filtered forces
        [frc.update(
            force_raw=frc['force_raw_online'][align_idx],
            force_filt=frc['force_filt_offline'][align_idx]
        )
        for frc, align_idx in zip(force_data, behavior_alignment)];

        # pop pre-aligned data
        for f_key in ['force_raw_online', 'force_filt_offline']:
            [frc.pop(f_key) for frc in force_data]

        # merge key with force data
        key = [dict(key, **frc) for frc in force_data]

        # insert aligned forces
        self.insert(key)


# =======
# LEVEL 1
# =======

@schema
class ForceMean(dj.Computed):
    definition = """
    # Trial-averaged forces
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    force_raw_mean:  longblob # trial-averaged raw (online), aligned force signal (V)
    force_raw_sem:   longblob # raw Mean force standard error
    force_filt_mean: longblob # trial-averaged filtered, aligned, and calibrated force (N)
    force_filt_sem:  longblob # filtered Mean force standard error
    """

    # limit conditions with good trials
    key_source = pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & Force \
        & (pacman_processing.GoodTrial & 'good_trial')

    def make(self, key):

        # fetch single-trial forces
        force_raw, force_filt = (Force & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('force_raw', 'force_filt')
        force_raw = np.stack(force_raw)
        force_filt = np.stack(force_filt)

        # update key with Mean and standard error
        key.update(
            force_raw_mean=force_raw.mean(axis=0),
            force_raw_sem=force_raw.std(axis=0, ddof=(1 if force_raw.shape[0] > 1 else 0))/np.sqrt(force_raw.shape[0]),
            force_filt_mean=force_filt.mean(axis=0),
            force_filt_sem=force_filt.std(axis=0, ddof=(1 if force_filt.shape[0] > 1 else 0))/np.sqrt(force_filt.shape[0]),
        )

        # insert forces
        self.insert1(key)


    def fetch_forces(
        self,
        fs: int=None,
        soft_normalize: int=None,
        mean_center: bool=False,
        output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch trial-averaged forces.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples forces to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the force range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition Mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options: 
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per force session/condition)
                * 'list' (list of N x T arrays, one per condition)
                where N, C, and T are the number of sessions, conditions, and time steps.

        Returns:
            forces (Any): Forces in specified output format
            condition_ids (Any): Condition IDs for each sample in forces
            condition_times (Any): Condition time value for each sample in forces
            condition_keys (List[dict]): List of condition keys in the dataset
            session_keys (List[dict]): List of force channel keys in the dataset
        """

        # ensure that there is one force per session/condition
        force_condtion_keys = pacman_acquisition.Behavior.Condition.primary_key
        remaining_keys = list(set(self.primary_key) - set(force_condtion_keys))

        n_forces_per_condition = dj.U(*force_condtion_keys).aggr(self, count='count(*)')
        assert not(n_forces_per_condition & 'count > 1'), 'More than one force per force channel and condition. Check ' \
            + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time'])

        # get session keys
        session_keys = (acquisition.Session & self).fetch('KEY')

        # remove raw forces and standard errors from table
        self = self.proj('force_filt_mean')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join force table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            forces = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch force data
                force_data = [(self & cond_key & sess_key).fetch1() for sess_key in session_keys]

                # interpolate forces to new timebase as needed
                if fs is not None:
                    [X.update(force_filt_mean=np.interp(t_new, X['t_old'], X['force_filt_mean'])) for X in force_data];

                # extract forces and append to list
                forces.append(np.array([X['force_filt_mean'] for X in force_data]))

        else:
            # fetch forces and stack across units
            forces = []
            for cond_key in condition_keys:
                forces.append(np.stack(
                    [(self & cond_key & sess_key).fetch1('force_filt_mean') for sess_key in session_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, forces)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            signal_range = np.hstack(forces).ptp(axis=1, keepdims=True)
            forces = [X/(signal_range + soft_normalize) for X in forces]

        # Mean-center
        if mean_center:
            signal_mean = np.hstack(forces).mean(axis=1, keepdims=True)
            forces = [X - signal_mean for X in forces]
        
        # format output
        if output_format == 'array':

            # stack output across conditions and times
            forces = np.hstack(forces)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            force_data = []
            for cond_key, X in zip(condition_keys, forces):
                for sess_key, Xi in zip(session_keys, X):
                    force_data.append(dict(cond_key, **sess_key, force_filt_mean=Xi))

            forces = force_data

        return forces, condition_ids, condition_times, condition_keys, session_keys