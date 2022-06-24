import datajoint as dj
import os, re, inspect, itertools
import pandas as pd
import numpy as np
import neo
import progressbar
import matplotlib.pyplot as plt
from src.churchland_pipeline_python import lab, acquisition, processing
from src.churchland_pipeline_python.utilities import datasync, datajointutils
from . import pacman_acquisition
from datetime import datetime
from sklearn import decomposition
from decimal import *
from typing import List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_processing')

# =======
# LEVEL 0
# =======

@schema
class AlignmentParams(dj.Manual):
    definition = """
    # Parameters for aligning trials
    -> pacman_acquisition.Behavior
    alignment_params_id:     tinyint unsigned # alignment params ID number
    ---
    -> pacman_acquisition.TaskState
    alignment_max_lag = 0.2: decimal(4,3)     # maximum absolute time lag for shifting each trial (s)
    """
    
    @classmethod
    def populate(self, 
        behavior_rel: pacman_acquisition.Behavior=pacman_acquisition.Behavior(), 
        task_state_rel: pacman_acquisition.TaskState=(pacman_acquisition.TaskState & {'task_state_name': 'InTarget'}), 
        max_lag: float=0.2) -> None:

        # check inputs
        assert isinstance(behavior_rel, pacman_acquisition.Behavior), 'Unrecognized behavior table'
        assert isinstance(task_state_rel, pacman_acquisition.TaskState), 'Unrecognized task state table'

        # construct "key source" from join of behavior and task state tables
        key_source = (behavior_rel * task_state_rel) - (self & {'alignment_max_lag': max_lag})

        behavior_source = behavior_rel & key_source.proj()
        task_state_source = task_state_rel & key_source.proj()

        # insert task state for every behavior
        for beh_key, task_state_key in itertools.product(behavior_source.fetch('KEY'), task_state_source.fetch('KEY')):

            # get filter params ID
            if not self & beh_key:
                new_param_id = 0
            else:
                all_param_id = (self & beh_key).fetch('alignment_params_id')
                new_param_id = next(i for i in range(2+max(all_param_id)) if i not in all_param_id)

            self.insert1(dict(**beh_key, alignment_params_id=new_param_id, **task_state_key, alignment_max_lag=max_lag))


@schema
class BehaviorBlock(dj.Manual):
    definition = """
    # Set of save tags and behavioral recording parameters for conducting analyses
    -> pacman_acquisition.Behavior
    behavior_block_id: tinyint unsigned # behavior block ID number
    ---
    -> pacman_acquisition.ArmPosture
    """
    
    class SaveTag(dj.Part):
        definition = """
        -> master
        -> pacman_acquisition.Behavior.SaveTag
        """

    @classmethod
    def insert_from_file(self,
        monkey: str
    ) -> None:
        """Inserts behavior block entries by reading data from a csv file."""

        # read table from metadata
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'metadata', '')
        behavior_block_df = pd.read_csv(metadata_path + monkey.lower() + '_behavior_block.csv')

        # ensure proper session date format
        behavior_block_df['session_date'] = pd.to_datetime(behavior_block_df['session_date']).dt.strftime('%Y-%m-%d')

        # convert dataframe to behavior block keys (remove secondary attributes)
        new_block_keys = behavior_block_df \
            .drop(['arm_posture_id', 'save_tag'], axis=1) \
            .to_dict(orient='records')

        # prepend dataframe index to key
        new_block_keys = [(idx, key) for idx, key in enumerate(new_block_keys)]

        # filter keys by those in behavior table but not in behavior block table
        old_block_keys = (BehaviorBlock & {'monkey': monkey}).fetch('KEY')
        [key.update(session_date=str(key['session_date'])) for key in old_block_keys];

        new_block_keys = [(idx, key) for idx, key in new_block_keys
            if (pacman_acquisition.Behavior & key) and key not in old_block_keys]

        # insert entries
        for idx, key in new_block_keys:

            # insert behavior block
            self.insert1(dict(key, arm_posture_id=behavior_block_df.loc[idx, 'arm_posture_id']))

            # read save tags for block
            save_tag_str = behavior_block_df.loc[idx,'save_tag']

            if re.match('\d:\d',save_tag_str):
                save_tags = np.arange(int(save_tag_str[0]), 1+int(save_tag_str[-1]))

            else:
                save_tags = eval('[' + save_tag_str + ']')

            # construct save tag keys
            save_tag_keys = [dict(key, save_tag=tag) for tag in save_tags]

            # insert behavior block save tags
            self.SaveTag.insert(save_tag_keys)


@schema
class BehaviorQualityParams(dj.Manual):
    definition = """
    # Parameters for inferring good trials based on behavior
    -> pacman_acquisition.Behavior.Condition
    behavior_quality_params_id:                    tinyint unsigned # behavior quality params ID number
    ---
    behavior_quality_max_lag = 0.2:                decimal(4,3)     # maximum absolute time lag for shifting each trial (s)
    behavior_quality_min_error_tolerance = 2:      decimal(6,4) 
    behavior_quality_error_tolerance_weight = 0.5: decimal(5,4)
    behavior_quality_max_error_duration = 0.03:    decimal(4,4)
    behavior_quality_max_std = 3.5:                decimal(5,4)
    """
    
    @classmethod
    def populate(self, 
        condition_rel: pacman_acquisition.Behavior=pacman_acquisition.Behavior.Condition(), 
        max_lag: float=0.2,
        min_error_tolerance: float=2,
        error_tolerance_weight: float=0.5,
        max_error_duration: float=0.03,
        max_std: float=3.5
    ) -> None:

        # check inputs
        assert isinstance(condition_rel, pacman_acquisition.Behavior.Condition), 'Unrecognized behavior condition table'

        # map inputs to dictionary
        params = dict(
            behavior_quality_max_lag=max_lag,
            behavior_quality_min_error_tolerance=min_error_tolerance,
            behavior_quality_error_tolerance_weight=error_tolerance_weight,
            behavior_quality_max_error_duration=max_error_duration,
            behavior_quality_max_std=max_std
        )

        # construct "key source" from join of behavior and task state tables
        key_source = condition_rel - (self & params)

        # insert task state for every behavior
        for key in key_source.fetch('KEY'):

            # update behavior quality params ID
            behavior_quality_params_key = dict(
                **key,
                behavior_quality_params_id=datajointutils.next_unique_int(self, 'behavior_quality_params_id', key),
                **params
            )

            # insert behavior quality params
            self.insert1(behavior_quality_params_key)


@schema
class EphysTrialStart(dj.Imported):
    definition = """
    # Synchronizes continuous acquisition ephys data with behavior trials
    -> processing.EphysSync
    -> pacman_acquisition.Behavior.Trial
    ---
    ephys_trial_start = null: int unsigned # sample index (ephys time base) corresponding to the trial start
    """

    # process each session with sync blocks
    key_source = processing.EphysSync & pacman_acquisition.Behavior.Trial

    def make(self, key):

        # ephys sample rate
        fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate') 

        # fetch all trial keys with simulation time
        trial_keys = (pacman_acquisition.Behavior.Trial & key).fetch('KEY','simulation_time',as_dict=True)

        # pop simulation time (Speedgoat clock) from trial key
        trial_time = [trial.pop('simulation_time',None) for trial in trial_keys]

        # sync block start index and encoded time stamp
        sync_block_start, sync_block_time = (processing.EphysSync.Block & key).fetch('sync_block_start', 'sync_block_time')

        # get trial start index in ephys time base
        ephys_trial_start_idx = datasync.get_ephys_trial_start(fs_ephys, trial_time, sync_block_start, sync_block_time)

        # legacy adjustment
        if key['session_date'] <= datetime.strptime('2018-10-11','%Y-%m-%d').date():
            ephys_trial_start_idx += round(0.1 * fs_ephys)

        # append ephys trial start to key
        trial_keys = [dict(key, **trial_key, ephys_trial_start=i0) for trial_key, i0 in zip(trial_keys, ephys_trial_start_idx)]

        self.insert(trial_keys)


@schema
class FilterParams(dj.Manual):
    definition = """
    # Set of filter parameters for smoothing forces and spike trains
    -> pacman_acquisition.Behavior.Condition
    filter_params_id: tinyint unsigned # filter params ID number
    ---
    -> processing.Filter
    """

    @classmethod
    def populate(self,
        condition_rel: pacman_acquisition.Behavior.Condition=pacman_acquisition.Behavior.Condition(), 
        filter_attr: dict={'sd':25e-3,'width':4}):

        _, filter_parts = datajointutils.join_parts(processing.Filter, filter_attr)
        filter_rel = next(x for x in filter_parts if x in datajointutils.get_children(processing.Filter))

        # check inputs
        assert isinstance(condition_rel, pacman_acquisition.Behavior.Condition), 'Unrecognized condition table'
        assert filter_rel in datajointutils.get_children(processing.Filter), 'Unrecognized filter table'

        # construct "key source" from join of condition and filter tables
        key_source = (condition_rel * filter_rel) - self

        cond_source = condition_rel & key_source.proj()
        filt_source = filter_rel & key_source.proj()

        # insert task state for every session
        for cond_key, filt_key in itertools.product(cond_source.fetch('KEY'), filt_source.fetch('KEY')):

            # get filter params ID
            if not self & cond_key:
                new_param_id = 0
            else:
                all_param_id = (self & cond_key).fetch('filter_params_id')
                new_param_id = next(i for i in range(2+max(all_param_id)) if i not in all_param_id)

            self.insert1(dict(**cond_key, filter_params_id=new_param_id, **filt_key))


# =======
# LEVEL 1
# =======

@schema
class BehaviorTrialAlignment(dj.Computed):
    definition = """
    # Trial alignment indices for behavior data
    -> pacman_acquisition.Behavior.Trial
    -> AlignmentParams
    ---
    valid_alignment = 0:       bool     # whether the trial can be aligned
    behavior_alignment = null: longblob # trial alignment indices for behavioral data
    """

    # restrict to successful trials
    key_source = (pacman_acquisition.Behavior.Trial & 'successful_trial') \
    * AlignmentParams

    def make(self, key):

        # trial table
        trial_rel = pacman_acquisition.Behavior.Trial & key

        # fetch all parameters from key source
        full_key = (self.key_source & key).fetch1()

        # set alignment index
        if pacman_acquisition.ConditionParams.Stim & trial_rel:

            # align to stimulationa
            stim = trial_rel.fetch1('stim')
            align_idx = next(i for i in range(len(stim)) if stim[i])

        else:
            # align to task state
            task_state = trial_rel.fetch1('task_state')
            align_idx = next(i for i in range(len(task_state)) if task_state[i] == full_key['task_state_id'])

        # behavioral sample rate
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')

        # fetch target force and time
        t, target_force = (pacman_acquisition.Behavior.Condition & trial_rel).fetch1('condition_time', 'condition_force')
        
        # behavior time indices and zero index
        t_idx_beh = (fs_beh * t).astype(int)
        zero_idx = np.argmax(t_idx_beh == 0)   

        # phase correct dynamic conditions
        if not pacman_acquisition.ConditionParams.Static & trial_rel:

            # generate lag range
            max_lag = float(full_key['alignment_max_lag'])
            max_lag_samp = int(round(fs_beh * max_lag))
            lags = range(-max_lag_samp, 1+max_lag_samp)

            # truncate time indices
            precision = int(np.ceil(np.log10(fs_beh)))
            trunc_idx = np.flatnonzero((t>=round(t[0]+max_lag, precision)) & (t<=round(t[-1]-max_lag, precision)))
            target_force_trunc = target_force[trunc_idx]
            align_idx_trunc = trunc_idx - zero_idx

            # process force signal
            force = trial_rel.process_force()[0]

            # compute normalized Mean squared error for each lag
            nmse = np.full(1+2*max_lag_samp, -np.inf)
            for idx, lag in enumerate(lags):
                if (align_idx + lag + align_idx_trunc[-1]) < len(force):
                    force_align = force[align_idx+lag+align_idx_trunc]
                    nmse[idx] = 1 - np.sqrt(np.mean((force_align-target_force_trunc)**2)/np.var(target_force_trunc))

            # shift alignment indices by optimal lag
            align_idx += lags[np.argmax(nmse)]

        # behavior alignment indices
        behavior_alignment = t_idx_beh + align_idx

        if behavior_alignment[-1] < len(trial_rel.fetch1('force_raw_online')):

            key.update(valid_alignment=1, behavior_alignment=behavior_alignment)

        self.insert1(key)


# =======
# LEVEL 2
# =======

@schema
class EphysTrialAlignment(dj.Computed):
    definition = """
    # Trial alignment indices for ephys data 
    -> BehaviorTrialAlignment
    -> EphysTrialStart
    ---
    ephys_alignment = null:    longblob # trial alignment indices for ephys data
    """

    key_source = BehaviorTrialAlignment * (EphysTrialStart & 'ephys_trial_start')

    def make(self, key):

        # fetch behavior alignment indices
        behavior_alignment = (BehaviorTrialAlignment & key).fetch1()

        if behavior_alignment['valid_alignment']:

            # fetch behavior and ephys sample rates
            fs_beh = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))
            fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

            # fetch condition time (behavior time base)
            t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

            # make condition time in ephys time base
            t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

            # convert time vectors to samples
            x_beh = np.round(fs_beh * t_beh).astype(int)
            x_ephys = np.round(fs_ephys * t_ephys).astype(int)

            # extract alignment index from behavior alignment indices and convert to ephys time base
            align_idx_beh = behavior_alignment['behavior_alignment'][x_beh==0]
            align_idx_ephys = int(align_idx_beh * round(fs_ephys/fs_beh))

            # fetch ephys trial start index and make ephys alignment indices
            ephys_trial_start = (EphysTrialStart & key).fetch1('ephys_trial_start')
            
            key.update(ephys_alignment=(x_ephys + align_idx_ephys + ephys_trial_start))

        self.insert1(key)


@schema 
class GoodTrial(dj.Computed):
    definition = """
    # Trials that meet behavior quality thresholds
    -> BehaviorTrialAlignment
    -> BehaviorQualityParams
    ---
    good_trial: bool
    """

    # process keys per condition
    key_source = (pacman_acquisition.Behavior.Condition & (BehaviorTrialAlignment & 'valid_alignment')) \
        * AlignmentParams \
        * BehaviorQualityParams

    def make(self, key):

        # get behavior quality params
        behavior_quality_params = (BehaviorQualityParams & key).fetch1()

        # convert params to float
        behavior_quality_params = {k:float(v) if isinstance(v,Decimal) else v for k,v in behavior_quality_params.items()}

        # trial table
        trial_rel = pacman_acquisition.Behavior.Trial & key & (BehaviorTrialAlignment & 'valid_alignment')

        # process force signal (default filter -- 25 ms Gaussian)
        trial_forces = trial_rel.process_force()

        # align force signals
        beh_align = (BehaviorTrialAlignment & trial_rel).fetch('behavior_alignment', order_by='trial')
        trial_forces = np.stack([f[idx] for f, idx in zip(trial_forces, beh_align)])

        # get target force profile
        target_force = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_force')

        # compute maximum shift in samples
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        max_lag_sec = behavior_quality_params['behavior_quality_max_lag']
        max_lag_samp = int(round(fs * max_lag_sec))

        # compute error tolerance for the condition (range of target force values within lag window)
        n_samples = len(target_force)
        err_tol = np.zeros(n_samples)

        for idx in range(n_samples):

            start_idx = max(0, idx-max_lag_samp)
            stop_idx = min(n_samples-1, idx+max_lag_samp)+1
            err_tol[idx] = target_force[start_idx:stop_idx].ptp()

        # bound error tolerance
        err_tol = np.maximum(
            behavior_quality_params['behavior_quality_min_error_tolerance'], 
            behavior_quality_params['behavior_quality_error_tolerance_weight'] * err_tol
        )

        # construct upper/lower bounds for trial forces
        mean_force = trial_forces.mean(axis=0)
        upper_bound = np.maximum(mean_force, target_force) + err_tol
        lower_bound = np.minimum(mean_force, target_force) - err_tol

        # identify good trials whose force values remain within bounds for at least 97% of the condition duration
        good_trials = np.mean((trial_forces < upper_bound) & (trial_forces > lower_bound), axis=1) \
            > (1 - behavior_quality_params['behavior_quality_max_error_duration'])

        # remove any remaining extreme outliers
        std_force = trial_forces.std(axis=0)
        good_trials = good_trials & np.all(
            (trial_forces < (mean_force + behavior_quality_params['behavior_quality_max_std'] * std_force))
            & (trial_forces > (mean_force - behavior_quality_params['behavior_quality_max_std'] * std_force)
        ), axis=1)

        # fetch alignment keys
        alignment_keys = (BehaviorTrialAlignment & key & 'valid_alignment').fetch('KEY', order_by='trial')

        # augment alignment keys with good trial
        alignment_keys = [
            dict(
                **alignment_key, 
                behavior_quality_params_id=key['behavior_quality_params_id'],
                good_trial=int(good_trial)
            ) 
        for alignment_key, good_trial in zip(alignment_keys, good_trials)]

        # insert good trials
        self.insert(alignment_keys)