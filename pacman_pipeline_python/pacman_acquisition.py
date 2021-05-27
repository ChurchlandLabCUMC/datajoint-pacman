import datajoint as dj
import os, re, inspect
import numpy as np
from churchland_pipeline_python import lab, acquisition, equipment, reference, processing
from churchland_pipeline_python.utilities import speedgoat, datajointutils
from decimal import Decimal
from functools import reduce
from typing import Tuple, List

DataJointTable = dj.user_tables.UserTable

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_acquisition')

# =======
# LEVEL 0
# =======

@schema 
class ArmPosture(dj.Lookup):
    definition = """
    # Arm posture
    -> lab.Monkey
    arm_posture_id:   tinyint unsigned # arm posture ID number
    ---
    elbow_flexion:    tinyint unsigned # elbow flexion angle (deg)
    shoulder_flexion: tinyint unsigned # shoulder flexion angle relative to coronal plane (deg)
    """
    
    contents = [
        ['Cousteau', 0, 90, 65],
        ['Cousteau', 1, 90, 40],
        ['Cousteau', 2, 90, 75]
    ]


@schema
class ConditionParams(dj.Lookup):
    """
    Task condition parameters. Each condition consists of a unique combination of force, 
    stimulation, and general target trajectory parameters. For conditions when stimulation
    was not delivered, stimulation parameters are left empty. Each condition also includes
    a set of parameters unique to the particular type of target trajectory.
    """

    definition = """
    condition_id: smallint unsigned # condition ID number
    """

    class Force(dj.Part):
        definition = """
        # Force parameters
        -> master
        force_id:       smallint unsigned # force ID number
        ---
        force_max:      tinyint unsigned  # maximum force (N)
        force_offset:   decimal(5,4)      # baseline force (N)
        force_inverted: bool              # whether pushing on the load cell moves PacMan up (False) or down (True) onscreen
        """
        
    class Stim(dj.Part):
        definition = """
        # CereStim parameters
        -> master
        stim_id:         smallint unsigned         # stim ID number
        ---
        -> equipment.ElectrodeArrayModel.Electrode # stim electrode
        stim_current:    smallint unsigned         # stim current (uA)
        stim_polarity:   tinyint unsigned          # cathodic (0) or anodic (1) first //TODO check this
        stim_pulses:     tinyint unsigned          # number of pulses in stim train
        stim_width1:     smallint unsigned         # first pulse duration (us)
        stim_width2:     smallint unsigned         # second pulse duration (us)
        stim_interphase: smallint unsigned         # interphase duration (us)
        stim_frequency:  smallint unsigned         # stim frequency (Hz)
        """

    class Target(dj.Part):
        definition = """
        # Target force profile parameters
        -> master
        target_id:       smallint unsigned # target ID number
        ---
        target_duration: decimal(5,4)      # target duration (s)
        target_offset:   decimal(5,4)      # target offset from baseline (proportion playable window)
        target_pad_pre:  decimal(5,4)      # duration of "padding" dots preceding target force profile (s)
        target_pad_post: decimal(5,4)      # duration of "padding" dots following target force profile (s)
        """
        
    class Static(dj.Part):
        definition = """
        # Static force profile parameters
        -> master.Target
        """

        def proj_label(self, keep_self: bool=True, n_sigfigs: int=4):
            """Project label."""

            rel = (self * ConditionParams.Target * ConditionParams.Force) \
                .proj(amp='CONVERT(ROUND(force_max*target_offset,{}), char)'.format(n_sigfigs)) \
                .proj(condition_label='CONCAT("Static (", amp, " N)")')

            if keep_self:
                rel = self * rel

            return rel

        def proj_rank(self, keep_self: bool=True):
            """Project ranking based on frequency and amplitude."""

            rel = (self * ConditionParams.Target * ConditionParams.Force) \
                .proj(amp='CONVERT(ROUND(force_max*target_offset, 4), char)') \
                .proj(condition_rank='CONCAT("00_", LPAD(amp, 8, 0))')

            if keep_self:
                rel = self * rel

            return rel


    class Ramp(dj.Part):
        definition = """
        # Linear ramp force profile parameters
        -> master.Target
        ---
        target_amplitude: decimal(5,4) # target amplitude (proportion playable window)
        """

        def proj_label(self, keep_self: bool=True, n_sigfigs: int=4):
            """Project label."""

            rel = (self * ConditionParams.Target * ConditionParams.Force) \
                .proj(amp='CONVERT(ROUND(force_max*target_amplitude/target_duration,{}), char)'.format(n_sigfigs)) \
                .proj(condition_label='CONCAT("Ramp (", amp, " N/s)")')

            if keep_self:
                rel = self * rel

            return rel

        def proj_rank(self, keep_self: bool=True):
            """Project ranking based on frequency and amplitude."""

            rel = (self * ConditionParams.Target * ConditionParams.Force) \
                .proj(amp='ROUND(force_max*target_amplitude/target_duration, 4)') \
                .proj(condition_rank='CONCAT("10_", LPAD(CONVERT(ABS(amp),char), 8, 0), "_", IF(amp>0, "0", "1"))')

            if keep_self:
                rel = self * rel

            return rel

        
    class Sine(dj.Part):
        definition = """
        # Sinusoidal (single-frequency) force profile parameters
        -> master.Target
        ---
        target_amplitude: decimal(5,4) # target amplitude (proportion playable window)
        target_frequency: decimal(5,4) # target frequency (Hz)
        """

        def proj_label(self, keep_self: bool=True, n_sigfigs: int=4):
            """Project label."""

            rel = (self * ConditionParams.Force) \
                .proj(
                    amp='CONVERT(ROUND(target_amplitude*force_max,{}), char)'.format(n_sigfigs), 
                    freq='CONVERT(ROUND(target_frequency,{}), char)'.format(n_sigfigs)
                ) \
                .proj(condition_label='CONCAT("Sine (", amp, " N, ", freq, " Hz)")')

            if keep_self:
                rel = self * rel

            return rel

        def proj_rank(self, keep_self: bool=True):
            """Project ranking based on frequency and amplitude."""

            rel = (self * ConditionParams.Target * ConditionParams.Force) \
                .proj(
                    amp='ROUND(target_amplitude*force_max, 4)', 
                    freq='CONVERT(ROUND(target_frequency, 4), char)'
                ) \
                .proj(condition_rank=(
                    'CONCAT("20_", LPAD(freq, 8, 0), "_", LPAD(CONVERT(ABS(amp),char), 8, 0), "_", IF(amp>0, "0", "1"))'
                ))

            if keep_self:
                rel = self * rel

            return rel

        
    class Chirp(dj.Part):
        definition = """
        # Chirp force profile parameters
        -> master.Target
        ---
        target_amplitude:       decimal(5,4) # target amplitude (proportion playable window)
        target_frequency_init:  decimal(5,4) # target initial frequency (Hz)
        target_frequency_final: decimal(5,4) # target final frequency (Hz)
        """

        def proj_label(self, keep_self: bool=True, n_sigfigs: int=4):
            """Project label."""

            rel = (self * ConditionParams.Force) \
                .proj(
                    amp='CONVERT(ROUND(force_max*target_amplitude,{}), char)'.format(n_sigfigs),
                    freq1='CONVERT(ROUND(target_frequency_init,{}), char)'.format(n_sigfigs),
                    freq2='CONVERT(ROUND(target_frequency_final,{}), char)'.format(n_sigfigs),
                ) \
                .proj(condition_label='CONCAT("Chirp (", amp, " N, ", freq1, "-", freq2, " Hz)")')

            if keep_self:
                rel = self * rel

            return rel

        def proj_rank(self, keep_self: bool=True):
            """Project ranking based on frequency and amplitude."""

            rel = (self * ConditionParams.Force) \
                .proj(
                    amp='ROUND(force_max*target_amplitude, 4)',
                    freq1='LPAD(CONVERT(ROUND(target_frequency_init, 4), char), 8, 0)',
                    freq2='LPAD(CONVERT(ROUND(target_frequency_final, 4), char), 8, 0)',
                ) \
                .proj(condition_rank=(
                    'CONCAT("30_", freq1, "_", freq2, "_", LPAD(CONVERT(ABS(amp),char), 8, 0), "_", IF(amp>0, "0", "1"))'
                ))

            if keep_self:
                rel = self * rel

            return rel


    def proj_label(self, n_sigfigs: int=4):
        """Project label in all child target tables and joins with master."""

        target_children = datajointutils.get_parts(ConditionParams.Target)

        target_labels = [dj.U('condition_id', 'condition_label') & (x & self).proj_label(n_sigfigs=n_sigfigs) for x in target_children]

        labeled_self = reduce(lambda x,y: x+y, target_labels)

        return labeled_self


    def proj_rank(self):
        """Project rank in all child target tables and joins with master."""

        target_children = datajointutils.get_parts(ConditionParams.Target)

        target_ranks = [dj.U('condition_id', 'condition_rank') & (x & self).proj_rank() for x in target_children]

        ranked_self = reduce(lambda x,y: x+y, target_ranks)

        return ranked_self


    def get_common_attributes(
        self, 
        table: DataJointTable, 
        include: List[str]=['label','rank'],
        n_sigfigs: int=4,
    ) -> List[dict]:
        """Fetches most common attributes in the input table.

        Args:
            table (DataJointTable): DataJoint table to use in the restriction
            include (List[str], optional): Attributes to project into the condition table. 
                Options: ['label','rank','time','force']. Defaults to ['label','rank'].
            n_sigfigs (int, optional): Number of significant figures include in label. Defaults to 4.

        Returns:
            condition_attributes (List[dict]): list of attributes
        """

        # count condition frequency in the table
        condition_counts = self.aggr(table, count='count(*)')

        # restrict by most counts
        max_count = dj.U().aggr(condition_counts, count='max(count)').fetch1('count')
        self = self & (condition_counts & 'count={}'.format(max_count)).proj()

        if include is not None:

            # project label
            self = self * ConditionParams().proj_label(n_sigfigs=n_sigfigs) if 'label' in include else self

            # project rank
            self = self * ConditionParams().proj_rank() if 'rank' in include else self

            # fetch attributes
            condition_attributes = self.fetch(as_dict=True, order_by=('condition_rank' if 'rank' in include else None))

            # aggregate target attributes
            target_attributes = []
            target_attributes.append('condition_time') if 'time' in include else None
            target_attributes.append('condition_force') if 'force' in include else None

            if any(target_attributes):

                # ensure matched sample rates across sessions
                behavior_recordings = acquisition.BehaviorRecording & table
                unique_sample_rates = dj.U('behavior_recording_sample_rate') & behavior_recordings
                assert len(unique_sample_rates) == 1, 'Mismatched sample rates!'

                fs = unique_sample_rates.fetch1('behavior_recording_sample_rate')

                # join condition table with secondary attributes
                for cond_attr in condition_attributes:

                    t, f = ConditionParams.target_force_profile(cond_attr['condition_id'], fs)

                    if 'time' in include:
                        cond_attr.update(condition_time=t)

                    if 'force' in include:
                        cond_attr.update(condition_force=f)

        else:
            condition_attributes = self.fetch(as_dict=True)

        return condition_attributes

        
    @classmethod
    def parse_params(self, params: dict, session_date: str=''):
        """
        Parses a dictionary constructed from a set of Speedgoat parameters (written
        on each trial) in order to extract the set of attributes associated with each
        part table of ConditionParams
        """

        # force attributes
        force_attr = dict(
            force_max = params['frcMax'], 
            force_offset = params['frcOff'],
            force_inverted = params['frcPol']==-1
        )

        cond_rel = self.Force

        # stimulation attributes
        if params.get('stim')==1:
                
            prog = re.compile('stim([A-Z]\w*)')
            stim_attr = {
                'stim_' + prog.search(k).group(1).lower(): v
                for k,v in zip(params.keys(), params.values()) 
                if prog.search(k) is not None and k != 'stimDelay'
                }

            # replace stim electrode with electrode array model electrode key
            try:
                ephys_stimulation_rel = acquisition.EphysStimulation & {'session_date': session_date}
                electrode_model_key = (equipment.ElectrodeArrayModel & ephys_stimulation_rel).fetch1('KEY')

            except:
                print('Missing EphysStimulation entry for session {}'.format(session_date))

            else:
                # get electrode array model electrode key (convert index from matlab convention)
                electrode_idx_key = {'electrode_idx': stim_attr['stim_electrode'] - 1}
                electrode_key = (equipment.ElectrodeArrayModel.Electrode & electrode_model_key & electrode_idx_key).fetch1('KEY')
                stim_attr.update(**electrode_key)

                # remove stim electrode attribute
                stim_attr.pop('stim_electrode')

            cond_rel = cond_rel * self.Stim
            
        else:
            stim_attr = dict()
            cond_rel = cond_rel - self.Stim

        # target attributes
        targ_attr = dict(
            target_duration = params['duration'],
            target_offset = params['offset'][0]
        )

        # target pad durations
        pad_dur = [v for k,v in params.items() if re.search('padDur',k) is not None]
        if len(pad_dur) == 1:
            targ_attr.update(target_pad_pre=pad_dur[0], target_pad_post=pad_dur[0])

        # target type attributes
        if params['type'] == 'STA':

            targ_type_rel = self.Static
            targ_type_attr = dict()

        elif params['type'] == 'RMP':

            targ_type_rel = self.Ramp
            targ_type_attr = dict(
                target_amplitude = params['amplitude'][0]
            )

        elif params['type'] == 'SIN':

            targ_type_rel = self.Sine
            targ_type_attr = dict(
                target_amplitude = params['amplitude'][0],
                target_frequency = params['frequency'][0]
            )

        elif params['type'] == 'CHP':

            targ_type_rel = self.Chirp
            targ_type_attr = dict(
                target_amplitude = params['amplitude'][0],
                target_frequency_init = params['frequency'][0],
                target_frequency_final = params['frequency'][1]
            )

        cond_rel = cond_rel * self.Target * targ_type_rel

        # aggregate all parameter attributes into a dictionary
        cond_attr = dict(
            Force = force_attr,
            Stim = stim_attr,
            Target = targ_attr,
            TargetType = targ_type_attr
        )

        return cond_attr, cond_rel, targ_type_rel
    
    @classmethod
    def target_force_profile(self, condition_id: int, fs: int):

        # ensure integer frequency
        assert fs == round(fs), 'Non-integer frequency'
        fs = int(fs)

        # join condition table with part tables
        joined_table, part_tables = datajointutils.join_parts(self, {'condition_id': condition_id}, depth=2, context=inspect.currentframe())

        # condition parameters
        cond_params = joined_table.fetch1()

        # convert sample rate to decimal type with precision inferred from condition parameters
        fs_dec = Decimal(fs).quantize(cond_params['target_duration'])

        # lengths of each target region
        target_lens = (
            int(round(cond_params['target_pad_pre']  * fs_dec)),
            int(round(cond_params['target_duration'] * fs_dec)) + 1,
            int(round(cond_params['target_pad_post'] * fs_dec))
        )

        # time samples
        xi = (
            np.arange(-target_lens[0], 0),
            np.arange(0, target_lens[1]),
            np.arange(target_lens[1], sum(target_lens[-2:]))
        )

        # target force functions
        if self.Static in part_tables:

            force_fcn = lambda t,c: c['target_offset'] * np.zeros(t.shape)

        elif self.Ramp in part_tables:

            force_fcn = lambda t,c: (c['target_amplitude']/c['target_duration']) * t

        elif self.Sine in part_tables:

            force_fcn = lambda t,c: c['target_amplitude']/2 * (1 - np.cos(2*np.pi*c['target_frequency']*t))

        elif self.Chirp in part_tables:

            force_fcn = lambda t,c: c['target_amplitude']/2 * \
                (1 - np.cos(2*np.pi*t * (c['target_frequency_init'] + (c['target_frequency_final']-c['target_frequency_init'])/(2*c['target_duration'])*t)))

        else:
            print('Unrecognized condition table')

        # convert condition parameters to float
        cond_params = {k:float(v) if isinstance(v,Decimal) else v for k,v in cond_params.items()}

        # construct target force profile
        force = np.hstack((
            force_fcn(xi[1][0]/fs,  cond_params) * np.ones(target_lens[0]),
            force_fcn(xi[1]/fs,     cond_params),
            force_fcn(xi[1][-1]/fs, cond_params) * np.ones(target_lens[2])
        ))

        # add force offset
        force += cond_params['target_offset']

        # scale force from screen units to Newtons
        force *= cond_params['force_max']

        # concatenate time samples and convert to seconds
        t = np.hstack(xi) / fs

        # round time to maximum temporal precision
        t = t.round(int(np.ceil(np.log10(fs))))

        return t, force


@schema
class TaskState(dj.Lookup):
    definition = """
    # Simulink Stateflow task state IDs and names
    task_state_id:   tinyint unsigned # task state ID number
    ---
    task_state_name: varchar(255)     # task state name
    """
    

# =======
# LEVEL 1
# =======
    
@schema
class Behavior(dj.Imported):
    definition = """
    # Behavioral data imported from Speedgoat
    -> acquisition.BehaviorRecording
    """

    key_source = acquisition.BehaviorRecording

    class Condition(dj.Part):
        definition = """
        # Condition data
        -> master
        -> ConditionParams
        ---
        condition_time:  longblob # condition time vector (s)
        condition_force: longblob # condition force profile (N)
        """

    class SaveTag(dj.Part):
        definition = """
        # Save tags and associated notes
        -> master
        save_tag: tinyint unsigned # save tag number
        """

    class Trial(dj.Part):
        definition = """
        # Trial data
        -> master.Condition
        trial:             smallint unsigned # session trial number
        ---
        -> master.SaveTag
        successful_trial:  bool             # whether the trial was successful
        simulation_time:   longblob         # task model simulation time
        task_state:        longblob         # task state IDs
        force_raw_online:  longblob         # amplified output of load cell
        force_filt_online: longblob         # online (boxcar) filtered and normalized force used to control Pac-Man
        reward:            longblob         # TTL signal indicating the delivery of juice reward
        photobox:          longblob         # photobox signal
        stim = null:       longblob         # TTL signal indicating the delivery of a stim pulse
        """

        def process_force(self, data_type='raw', apply_filter=True, keep_keys=False):

            # aggregate load cell parameters per session
            load_cell_params = (acquisition.Session.Hardware & {'hardware': '5lb Load Cell'}) * equipment.Hardware.Parameter & self

            force_capacity_per_session = dj.U(*acquisition.Session.primary_key) \
                .aggr((load_cell_params & {'equipment_parameter': 'force capacity'}), force_capacity='equipment_parameter_value')

            voltage_output_per_session = dj.U(*acquisition.Session.primary_key) \
                .aggr((load_cell_params & {'equipment_parameter': 'voltage output'}), voltage_output='equipment_parameter_value')

            load_cell_params_per_session = force_capacity_per_session * voltage_output_per_session

            # 25 ms Gaussian filter
            filter_rel = processing.Filter.Gaussian & {'sd':25e-3, 'width':4}

            # join trial force data with force and load cell parameters
            force_rel = self * ConditionParams.Force * load_cell_params_per_session

            # fetch force data
            data_type_attr = {'raw':'force_raw_online', 'filt':'force_filt_online'}
            data_attr = data_type_attr[data_type]
            force_data = force_rel \
                .proj(data_attr, 'force_max', 'force_offset', 'force_capacity', 'voltage_output') \
                .fetch(as_dict=True, order_by='trial')

            # sample rate
            fs = (acquisition.BehaviorRecording & self).fetch1('behavior_recording_sample_rate')

            # process trial data
            for f in force_data:

                f[data_attr] = f[data_attr].copy()

                # normalize force (V) by load cell capacity (V)
                f[data_attr] /= f['voltage_output']

                # convert force to proportion of maximum load cell output (N)
                f[data_attr] *= f['force_capacity']/f['force_max']

                # subtract baseline force (N)
                f[data_attr] -= float(f['force_offset'])

                # multiply force by maximum gain (N)
                f[data_attr] *= f['force_max']

                # filter
                if apply_filter:
                    f[data_attr] = filter_rel.filt(f[data_attr], fs)

            # pop force parameters
            for key in ['force_id', 'force_max', 'force_offset', 'force_capacity', 'voltage_output']:
                [f.pop(key) for f in force_data]

            # limit output to force signal
            if not keep_keys:
                force_data = np.array([f[data_attr] for f in force_data])

            return force_data            
        
    def make(self, key):

        self.insert1(key)

        if (acquisition.Session.Hardware & key & {'hardware': 'Speedgoat'}):

            # behavior sample rate
            fs = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))

            # summary file path
            summary_file_path = (acquisition.BehaviorRecording.File & key & {'behavior_file_extension': 'summary'})\
                .proj_file_path().fetch1('behavior_file_path')

            # ensure local path
            summary_file_path = reference.EngramTier.ensure_local(summary_file_path)

            # read summary file
            summary = speedgoat.read_task_states(summary_file_path)

            # update task states
            TaskState.insert(summary, skip_duplicates=True)

            # parameter and data file paths
            params_file_paths = (acquisition.BehaviorRecording.File & key & {'behavior_file_extension': 'params'})\
                .proj_file_path().fetch('behavior_file_path')

            data_file_paths = (acquisition.BehaviorRecording.File & key & {'behavior_file_extension': 'data'})\
                .proj_file_path().fetch('behavior_file_path')

            # ensure local paths
            params_file_paths = [reference.EngramTier.ensure_local(pth) for pth in params_file_paths]
            data_file_paths = [reference.EngramTier.ensure_local(pth) for pth in data_file_paths]

            # populate conditions from parameter files
            for params_path in params_file_paths:

                # trial number
                trial = re.search(r'beh_(\d*)', params_path).group(1)

                # ensure matching data file exists
                if params_path.replace('params','data') not in data_file_paths:

                    print('Missing data file for trial {}'.format(trial))

                else:
                    # read params file
                    params = speedgoat.read_trial_params(params_path)

                    if not params:
                        continue

                    # extract condition attributes from params file
                    cond_attr, cond_rel, targ_type_rel = ConditionParams.parse_params(params, key['session_date'])

                    # aggregate condition part table parameters into a single dictionary
                    all_cond_attr = {k: v for d in list(cond_attr.values()) for k, v in d.items()}
                    
                    # insert new condition if none exists
                    if not(cond_rel & all_cond_attr):

                        # insert condition table
                        new_cond_id = datajointutils.next_unique_int(ConditionParams, 'condition_id')
                        cond_key = {'condition_id': new_cond_id}

                        ConditionParams.insert1(cond_key)

                        # insert Force, Stim, and Target tables
                        for cond_part_name in ['Force', 'Stim', 'Target']:

                            # attributes for part table
                            cond_part_attr = cond_attr[cond_part_name]

                            if not(cond_part_attr):
                                continue

                            cond_part_rel = getattr(ConditionParams, cond_part_name)
                            cond_part_id = cond_part_name.lower() + '_id'

                            if not(cond_part_rel & cond_part_attr):

                                cond_part_attr[cond_part_id] = datajointutils.next_unique_int(cond_part_rel, cond_part_id)
                                
                            else:
                                cond_part_attr[cond_part_id] = (cond_part_rel & cond_part_attr).fetch(cond_part_id, limit=1)[0]

                            cond_part_rel.insert1(dict(**cond_key, **cond_part_attr))

                        # insert target type table
                        targ_type_rel.insert1(dict(**cond_key, **cond_attr['TargetType'], target_id=cond_attr['Target']['target_id']))
                    

            # populate trials from data files
            success_state = (TaskState() & 'task_state_name="Success"').fetch1('task_state_id')

            for data_path in data_file_paths:

                # trial number
                trial = int(re.search(r'beh_(\d*)',data_path).group(1))

                # find matching parameters file
                try:
                    params_path = next(filter(lambda f: data_path.replace('data','params')==f, params_file_paths))
                except StopIteration:
                    print('Missing parameters file for trial {}'.format(trial))
                else:
                    # convert params to condition keys
                    params = speedgoat.read_trial_params(params_path)

                    if not params:
                        continue

                    cond_attr, cond_rel, targ_type_rel = ConditionParams.parse_params(params, key['session_date'])

                    # read data
                    data = speedgoat.read_trial_data(data_path, success_state, fs)

                    if not data:
                        continue
                        
                    # aggregate condition part table parameters into a single dictionary
                    all_cond_attr = {k: v for d in list(cond_attr.values()) for k, v in d.items()}

                    # insert condition data
                    cond_id = (cond_rel & all_cond_attr).fetch1('condition_id')
                    cond_key = dict(**key, condition_id=cond_id)
                    if not(self.Condition & cond_key):
                        t, force = ConditionParams.target_force_profile(cond_id, fs)
                        cond_key.update(condition_time=t, condition_force=force)
                        self.Condition.insert1(cond_key, allow_direct_insert=True)

                    # insert save tag key
                    save_tag_key = dict(**key, save_tag=params['saveTag'])
                    if not (self.SaveTag & save_tag_key):
                        self.SaveTag.insert1(save_tag_key)

                    # insert trial data
                    trial_key = dict(**key, trial=trial, condition_id=cond_id, **data, save_tag=params['saveTag'])
                    self.Trial.insert1(trial_key)

        else: 
            print('Unrecognized task controller')
            return None