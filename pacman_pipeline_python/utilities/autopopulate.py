"""Utilities for automatically populating manual tables for the Pac-Man Task."""

import datajoint as dj
import os, re, inspect, time, shutil
import pandas as pd
import numpy as np
import neo
import progressbar
from src.churchland_pipeline_python import acquisition, action, equipment, lab, processing, reference
from src.churchland_pipeline_python.utilities import datajointutils
from src.pacman_pipeline_python import pacman_acquisition, pacman_processing, pacman_behavior, pacman_brain, pacman_muscle
from . import datasynthesis
from datetime import datetime
from typing import List, Tuple
import pdb

# =======
# SESSION
# =======

def session(
        users: List[str],
        monkey: str='Cousteau',
        task_version: str='1.0',
        dates: List[str]=None,
        hardware: Tuple[str]=('Speedgoat', 'Cerebus', '5lb Load Cell'),
        software: Tuple[str]=('Simulink', 'Psychtoolbox')
    ) -> None:

        # input keys
        user_keys =   (lab.User & [{'user_uni': uni} for uni in users]).fetch('KEY')
        monkey_key = (lab.Monkey & {'monkey': monkey}).fetch1('KEY')
        rig_key =    (lab.Rig & {'rig': 'Jumanji'}).fetch1('KEY')
        task_key =   (acquisition.Task & {'task': 'pacman', 'task_version': task_version}).fetch1('KEY')

        # get session dates and raw path
        session_dates, raw_path = datasynthesis.get_sessions(monkey)
        
        # restrict dates based on user input
        if dates:
            session_dates = [d for d in session_dates if d in dates]            

        # remove sessions with inputs
        session_dates = [date for date in session_dates if not acquisition.Session & {'session_date': date}]

        for date in session_dates:

            session_path = os.path.sep.join([raw_path, date, ''])
            session_files = os.listdir(session_path)

            # check if behavior directory exists
            try:
                if 'Speedgoat' in hardware:
                    behavior_dir = 'speedgoat'

                next(filter(lambda x: x==behavior_dir, session_files))

            except StopIteration:
                print('Missing behavior files for session {}'.format(date))
                has_behavior = False

            else:
                has_behavior = True

            # check if ephys directory exists
            try:
                if 'Cerebus' in hardware:
                    ephys_dir = 'blackrock'

                elif 'IMEC' in hardware:
                    ephys_dir = 'imec'

                next(filter(lambda x: x==ephys_dir, session_files))

            except StopIteration:
                print('Missing ephys files for session {}'.format(date))
                has_ephys = False
                
            else:
                has_ephys = True

            # insert session if ephys or behavioral data
            if has_behavior or has_ephys:
            
                # session key
                session_key = dict(session_date=date, **monkey_key)
                
                # insert session
                acquisition.Session.insert1(dict(**session_key, **rig_key, **task_key))

                # insert users
                for user_key in user_keys:
                    acquisition.Session.User.insert1(dict(**session_key, **user_key))

                # insert notes
                try:
                    notes_files = next(x for x in session_files if re.search('.*notes\.txt',x))
                except StopIteration:
                    print('Missing notes for session {}'.format(date))
                else:
                    with open(session_path + notes_files,'r') as f:
                        acquisition.Session.Notes.insert1(dict(**session_key, session_notes_id=0, session_notes=f.read()))

                # insert hardware
                for hardware_name in hardware:
                    acquisition.Session.Hardware.insert1(dict(
                        **session_key, 
                        **(equipment.Hardware & {'hardware': hardware_name}).fetch1('KEY')
                    ))

                # insert software
                for software_name in software:
                    acquisition.Session.Software.insert1(dict(
                        **session_key, 
                        **(equipment.Software & {'software': software_name}).fetch1('KEY')
                    ))


# ==================
# BEHAVIOR RECORDING
# ==================

def behaviorrecording(behavior_sample_rate: int=1e3, display_progress: bool=True):

    # function to get path to behavior files
    def getbehaviorpath(session_key):

        # path to raw data
        raw_path = datasynthesis.get_data_path(session_key['monkey'])

        # directory name
        if (acquisition.Session.Hardware & session_key & {'hardware': 'Speedgoat'}):

            behavior_path = raw_path + os.path.sep.join([str(session_key['session_date']), 'speedgoat', ''])

        return behavior_path

    # remove problematic sessions and those with behavior recording entries
    key_source = acquisition.Session - 'session_problem' - acquisition.BehaviorRecording

    # remove sessions without valid behavior paths
    key_source = key_source - [key for key in key_source.fetch('KEY') if not os.path.isdir(getbehaviorpath(key))]

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, session_key in enumerate(key_source.fetch('KEY')):

        # get behavior files
        behavior_path = getbehaviorpath(session_key)
        behavior_files = sorted(list(os.listdir(behavior_path)))

        # split files by extension
        behavior_file_parts = [x.split(sep='.') for x in behavior_files]
        # behavior file keys

        behavior_file_keys = [
            dict(**session_key, behavior_file_id=idx, behavior_file_path='', behavior_file_name=x[0], behavior_file_extension=x[1]) \
                for idx, x in enumerate(behavior_file_parts)
            ]

        
        # insert behavior recording
        acquisition.BehaviorRecording.insert1(dict(
            **session_key, 
            behavior_recording_sample_rate=behavior_sample_rate,
            behavior_recording_path=reference.EngramTier.ensure_remote(behavior_path)
        ))

        # insert behavior recording files
        acquisition.BehaviorRecording.File.insert(behavior_file_keys)

        if display_progress:
            bar.update(1+key_idx)


# ===============
# EPHYS RECORDING
# ===============

def ephysrecording(display_progress: bool=True):

    # function to get path to ephys files
    def getephyspath(session_key):

        # path to raw data
        raw_path = datasynthesis.get_data_path(session_key['monkey'])

        # directory name
        if (acquisition.Session.Hardware & session_key & {'hardware': 'Cerebus'}):

            ephys_path = os.path.sep.join([raw_path[:-1], str(session_key['session_date']), 'blackrock', ''])

        return ephys_path

    # remove problematic sessions and those with ephys recording entries
    key_source = acquisition.Session - 'session_problem' - acquisition.EphysRecording

    # remove sessions without valid behavior paths
    key_source = key_source - [key for key in key_source.fetch('KEY') if not os.path.isdir(getephyspath(key))]

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, session_key in enumerate(key_source.fetch('KEY')):

        # ephys file regular expression
        ephys_file_regexp = '_'.join([
            'pacman-task',
            session_key['monkey'][0].lower(),
            str(session_key['session_date']).replace('-','')[2:],
            '(emg|neu|neu_emg)',
            '\d{3}\.ns\d'
            ])

        if (acquisition.Session.Hardware & session_key & {'hardware': 'Cerebus'}):

            # get blackrock files
            ephys_path = getephyspath(session_key)
            blackrock_files = list(os.listdir(ephys_path))

            # recording key
            ephys_recording_key = dict(**session_key,
                ephys_recording_path=reference.EngramTier.ensure_remote(ephys_path), 
                ephys_recording_duration=0)

            # NSx files
            nsx_files = [f for f in blackrock_files if re.search(ephys_file_regexp,f)]

            # ephys file and channel keys
            ephys_file_keys = []
            ephys_channel_keys = []

            for i_file, file_name in enumerate(nsx_files):

                # behavior file keys
                file_parts = file_name.split(sep='.')
                ephys_file_keys.append(dict(
                    **session_key, 
                    ephys_file_id=i_file, 
                    ephys_file_path='',
                    ephys_file_name=file_parts[0], 
                    ephys_file_extension=file_parts[1]
                ))

                # read NSx file
                reader = neo.rawio.BlackrockRawIO(ephys_path + file_name)
                reader.parse_header()

                # pull sample rate and update recording duration
                fs = int(next(iter(reader.sig_sampling_rates.values())))
                ephys_recording_key.update(ephys_recording_sample_rate = fs)
                ephys_recording_key['ephys_recording_duration'] += reader.get_signal_size(0,0) / fs

                # channel header name and ID indices
                name_idx, id_idx = [
                    idx for idx, name in enumerate(reader.header['signal_channels'].dtype.names) \
                    if name in ['name','id']
                ]

                # insert channel header information
                chan_attr = []
                for j, chan in enumerate(reader.header['signal_channels']):

                    chan_name = chan[name_idx]

                    # read channel type
                    if re.search('^(?!ainp)', chan_name):
                        chan_type = 'brain'

                    elif re.search('ainp[1-8]$', chan_name):
                        chan_type = 'emg'

                    elif chan_name == 'ainp15':
                        chan_type = 'stim'

                    elif chan_name == 'ainp16':
                        chan_type = 'sync'

                    else:
                        print('Channel type {} unrecognized'.format(chan_name))
                        return

                    # write channel attributes
                    chan_attr.append(dict(
                        **session_key, 
                        ephys_file_id=i_file, 
                        ephys_channel_idx=j, 
                        ephys_channel_id=chan[id_idx], 
                        ephys_channel_type=chan_type
                    ))

                # write channel attributes
                ephys_channel_keys.extend(chan_attr)

            # insert ephys recording
            acquisition.EphysRecording.insert1(ephys_recording_key)

            # insert ephys recording files
            acquisition.EphysRecording.File.insert(ephys_file_keys)

            # insert ephys recording channels
            acquisition.EphysRecording.Channel.insert(ephys_channel_keys)

            if display_progress:
                bar.update(1+key_idx)


# =================
# EPHYS STIMULATION
# =================

def ephysstimulation(display_progress: bool=True):

    # remove problematic sessions and those with ephys recording entries
    key_source = acquisition.Session - 'session_problem' - acquisition.EphysStimulation

    # filter sessions by those with "Stimulating" in notes
    key_source = key_source & [key for key in key_source.fetch('KEY') \
        if any(['Stimulating' in note for note in (acquisition.Session.Notes & key).fetch('session_notes')])]

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    # probe depth pattern
    probe_depth_pattern = re.compile(r'probe.*?(\d+\.?\d*)\s?mm')

    for key_idx, session_key in enumerate(key_source.fetch('KEY')):

        # initialize ephys stimulation key
        ephys_stimulation_key = dict(
            **session_key,
            ephys_stimulation_probe_id=datajointutils.next_unique_int(acquisition.EphysStimulation, 'ephys_stimulation_probe_id', session_key)
        )

        # read stimulating record note
        session_note = next(note for note in (acquisition.Session.Notes & session_key).fetch('session_notes') \
            if 'Stimulating' in note)

        # read electrode array model
        electrode_array_model = None
        if '(N|n)europixels' in session_note:
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'Neuropixels', 'electrode_array_model_version': 'nhp demo'}).fetch1('KEY')

        elif re.search(r'(S|s)(-|\s)(P|p)robe', session_note):
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'S-Probe'}).fetch1('KEY')

        elif re.search(r'(V|v)(-|\s)(P|p)robe', session_note):
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'V-Probe'}).fetch1('KEY')

        if electrode_array_model:
            ephys_stimulation_key.update(**(equipment.ElectrodeArray & electrode_array_model & {'electrode_array_id': 0}).fetch1('KEY'))

        # read probe depth
        if probe_depth_pattern.search(session_note):
            ephys_stimulation_key.update(probe_depth=float(probe_depth_pattern.search(session_note).group(1)))

        # insert ephys stimulation key
        acquisition.EphysStimulation.insert1(ephys_stimulation_key)

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ===================
# BRAIN CHANNEL GROUP
# ===================

def brainchannelgroup(display_progress: bool=True):

    # remove files without brain channels or with pre-existing channel group entries
    key_source = (acquisition.EphysRecording.File \
        & (acquisition.EphysRecording.Channel & {'ephys_channel_type': 'brain'})) \
        - acquisition.BrainChannelGroup

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, ephys_file_key in enumerate(key_source.fetch('KEY')):

        # read brain recording attributes from notes file
        brain_attr, _ = datasynthesis.parse_notes(ephys_file_key, read_type=('brain'))

        if brain_attr:

            # check for config file
            config_rel = equipment.ElectrodeArrayConfig & brain_attr[0]
            if config_rel and config_rel.attributes_in_restriction():
                [attr.update(**config_rel.fetch1('KEY')) for attr in brain_attr]

            # insert brain channel group
            acquisition.BrainChannelGroup.insert(brain_attr)

            # get channel group primary keys
            brain_channel_group_keys = (acquisition.BrainChannelGroup \
                & ephys_file_key).fetch('KEY', order_by='brain_channel_group_id')

            # get ephys recording channel keys
            ephys_recording_chan_keys = (acquisition.EphysRecording.Channel \
                & ephys_file_key & {'ephys_channel_type': 'brain'}).fetch('KEY', order_by='ephys_channel_idx')

            num_brain_groups = len(brain_channel_group_keys)
            assert num_brain_groups == 1, '{} brain channel groups. Action unclear'.format(num_brain_groups)

            # replicate channel group key for each brain recording channel
            for idx, ephys_key in enumerate(ephys_recording_chan_keys):
                ephys_key.update(**brain_channel_group_keys[0], brain_channel_idx=idx)

            # insert brain channel group channels
            acquisition.BrainChannelGroup.Channel.insert(ephys_recording_chan_keys)

        else:
            print('Missing attributes for session {}'.format(ephys_file_key['session_date']))

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# =================
# EMG CHANNEL GROUP
# =================

def emgchannelgroup(display_progress: bool=True):

    # remove files without EMG channels or with pre-existing channel group entries
    key_source = (acquisition.EphysRecording.File \
        & (acquisition.EphysRecording.Channel & {'ephys_channel_type': 'emg'})) \
        - acquisition.EmgChannelGroup

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, ephys_file_key in enumerate(key_source.fetch('KEY')):

        # read emg recording attributes from notes file
        _, emg_attr = datasynthesis.parse_notes(ephys_file_key, read_type=('emg'))

        if emg_attr:

            # insert emg channel group
            acquisition.EmgChannelGroup.insert(emg_attr)

            # get channel group primary keys
            emg_channel_group_keys = (acquisition.EmgChannelGroup \
                & ephys_file_key).fetch('KEY', order_by='emg_channel_group_id')

            # get ephys recording channel keys
            ephys_recording_chan_keys = (acquisition.EphysRecording.Channel \
                & ephys_file_key & {'ephys_channel_type': 'emg'}).fetch('KEY', order_by='ephys_channel_idx')

            num_muscle_groups = len(emg_channel_group_keys)
            num_emg_channels = len(ephys_recording_chan_keys)
            assert num_muscle_groups in (1, num_emg_channels), \
                '{} emg channel groups and {} emg recording channels. Action unclear'.format(num_muscle_groups, num_emg_channels)

            if num_muscle_groups == 1:

                # replicate channel group key for each emg recording channel
                for idx, ephys_chan_key in enumerate(ephys_recording_chan_keys):
                    ephys_chan_key.update(**emg_channel_group_keys[0], emg_channel_idx=idx)

            else:
                # copy emg group attributes to each emg recording channel
                for emg_group_key, ephys_chan_key in zip(emg_channel_group_keys, ephys_recording_chan_keys):
                    ephys_chan_key.update(**emg_group_key, emg_channel_idx=0)

            # insert emg channel group channels
            acquisition.EmgChannelGroup.Channel.insert(ephys_recording_chan_keys)

        else:
            print('Missing attributes for session {}'.format(ephys_file_key['session_date']))

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ==========
# BRAIN SORT
# ==========

def brainsort(monkey: str='Cousteau', spike_sorter: Tuple[str]=('Kilosort','2.0'), display_progress: bool=True):

    key_source = acquisition.BrainChannelGroup.fetch('KEY')

    # path to processed data
    processed_path = datasynthesis.get_data_path(monkey, data_type='processed')

    # match spike sorter input to software key
    _, software_key = datajointutils.match_fuzzy_key(equipment.Software, spike_sorter)

    # path to spike sorter file
    if software_key['software'] == 'Kilosort':

        # remote path to kilosort files
        kilosort_path = [(key, processed_path + os.path.sep.join([str(key['session_date']), 'kilosort-manually-sorted', '']))
            for key in key_source]

        # remove non-existent paths
        kilosort_path = [(key, pth) for key, pth in kilosort_path if os.path.isdir(pth)]

    else:
        print('Spike sorter {} unrecognized'.format(software_key))
        return None

    # brain sort file regular expression
    brain_sort_file_regexp = '_'.join([
        'pacman-task',
        monkey[0].lower(),
        '\d{6}',
        '(neu|neu_emg)',
        '\d{3}'
        ])

    # full sort paths
    sort_path = []
    for key, pth in kilosort_path:
        sort_dir = [d for d in os.listdir(pth) if re.search(brain_sort_file_regexp, d)]
        sort_path.extend([(key, pth + os.path.sep.join([d, ''])) for d in sort_dir])

    # ensure remote path
    sort_path = [(pth[0], reference.EngramTier.ensure_remote(pth[1])) for pth in sort_path]

    # remove paths already in table
    sort_path = [pth for pth in sort_path if not (processing.BrainSort & {'brain_sort_path': pth[1]})]

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(sort_path))
        bar.update(0)

    for key_idx, (brain_sort_key, brain_sort_path) in enumerate(sort_path):

        # update key with spike sorter and sort path attributes
        brain_sort_key.update(**software_key, brain_sort_path=brain_sort_path)

        # increment sort ID number
        brain_sort_id = datajointutils.next_unique_int(processing.BrainSort, 'brain_sort_id', brain_sort_key)
        brain_sort_key.update(brain_sort_id=brain_sort_id)

        # insert brain sort
        processing.BrainSort.insert1(brain_sort_key)

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ========
# EMG SORT
# ========

def emgsort(monkey: str='Cousteau', spike_sorter: Tuple[str]=('Myosort','1.0'), display_progress: bool=True):

    key_source = acquisition.EmgChannelGroup.fetch('KEY')

    # path to processed data
    processed_path = datasynthesis.get_data_path(monkey, data_type='processed')

    # match spike sorter input to software key
    _, software_key = datajointutils.match_fuzzy_key(equipment.Software, spike_sorter)

    # path to spike sorter file
    if software_key['software'] == 'Myosort':

        # local path to kilosort files
        myosort_path = [(key, processed_path + os.path.sep.join([str(key['session_date']), 'myosort-out', '']))
            for key in key_source]

        # remove non-existent paths
        myosort_path = [(key, pth) for key, pth in myosort_path if os.path.isdir(pth)]

    else:
        print('Spike sorter {} unrecognized'.format(software_key))
        return None

    # emg sort file regular expression
    emg_sort_file_regexp = '_'.join([
        'pacman-task',
        monkey[0].lower(),
        '\d{6}',
        '(emg|neu_emg)',
        '\d{3}'
        ])

    # full sort paths
    sort_path = []
    for key, pth in myosort_path:
        sort_dir = [d for d in os.listdir(pth) if re.search(emg_sort_file_regexp, d)]

        # downsample to matlab export if available
        if any(['matlab_export' in d for d in sort_dir]):
            sort_dir = [d for d in sort_dir if 'matlab_export' in d]

        sort_path.extend([(key, pth + os.path.sep.join([d, ''])) for d in sort_dir])

    # ensure remote path
    sort_path = [(pth[0], reference.EngramTier.ensure_remote(pth[1])) for pth in sort_path]

    # remove paths already in table
    sort_path = [pth for pth in sort_path if not (processing.EmgSort & {'emg_sort_path': pth[1]})]

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(sort_path))
        bar.update(0)

    for key_idx, (emg_sort_key, emg_sort_path) in enumerate(sort_path):

        # update key with spike sorter and sort path attributes
        emg_sort_key.update(**software_key, emg_sort_path=emg_sort_path)

        # increment sort ID number
        emg_sort_id = datajointutils.next_unique_int(processing.EmgSort, 'emg_sort_id', emg_sort_key)
        emg_sort_key.update(emg_sort_id=emg_sort_id)

        # insert emg sort
        processing.EmgSort.insert1(emg_sort_key)

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ========
# PIPELINE
# ========

def pipeline(monkey: str='Cousteau', display_progress: bool=True):

    # session descendants populate functions
    populate_functions = {
        acquisition.BehaviorRecording:            (behaviorrecording,                                 {'display_progress': display_progress}),
        acquisition.EphysRecording:               (ephysrecording,                                    {'display_progress': display_progress}),
        acquisition.EphysStimulation:             (ephysstimulation,                                  {'display_progress': display_progress}),
        acquisition.BrainChannelGroup:            (brainchannelgroup,                                 {'display_progress': display_progress}),
        acquisition.EmgChannelGroup:              (emgchannelgroup,                                   {'display_progress': display_progress}),
        processing.BrainSort:                     (brainsort,                                         {'display_progress': display_progress}),
        processing.EmgSort:                       (emgsort,                                           {'display_progress': display_progress}),
        processing.Neuron:                        (processing.Neuron.populate,                        {'display_progress': display_progress}),
        processing.MotorUnit:                     (processing.MotorUnit.populate,                     {'display_progress': display_progress}),
        pacman_acquisition.Behavior:              (pacman_acquisition.Behavior.populate,              {'display_progress': display_progress}),
        processing.EphysSync:                     (processing.EphysSync.populate,                     {'display_progress': display_progress}),
        pacman_processing.AlignmentParams:        (pacman_processing.AlignmentParams.populate,        {}),
        pacman_processing.BehaviorBlock:          (pacman_processing.BehaviorBlock.insert_from_file,  {'monkey': monkey}),
        pacman_processing.BehaviorQualityParams:  (pacman_processing.BehaviorQualityParams.populate,  {}),
        pacman_processing.EphysTrialStart:        (pacman_processing.EphysTrialStart.populate,        {'display_progress': display_progress}),
        pacman_processing.FilterParams:           (pacman_processing.FilterParams.populate,           {}),
        pacman_processing.BehaviorTrialAlignment: (pacman_processing.BehaviorTrialAlignment.populate, {'display_progress': display_progress}),
        pacman_processing.EphysTrialAlignment:    (pacman_processing.EphysTrialAlignment.populate,    {'display_progress': display_progress}),
        pacman_processing.GoodTrial:              (pacman_processing.GoodTrial.populate,              {'display_progress': display_progress}),
        pacman_behavior.Force:                    (pacman_behavior.Force.populate,                    {'display_progress': display_progress}),
        pacman_behavior.ForceMean:                (pacman_behavior.ForceMean.populate,                {'display_progress': display_progress}),
        pacman_brain.NeuronSpikeRaster:           (pacman_brain.NeuronSpikeRaster.populate,           {'display_progress': display_progress}),
        pacman_brain.NeuronRate:                  (pacman_brain.NeuronRate.populate,                  {'display_progress': display_progress}),
        pacman_brain.NeuronPsth:                  (pacman_brain.NeuronPsth.populate,                  {'display_progress': display_progress}),
        pacman_muscle.MotorUnitSpikeRaster:       (pacman_muscle.MotorUnitSpikeRaster.populate,       {'display_progress': display_progress}),
        pacman_muscle.MotorUnitRate:              (pacman_muscle.MotorUnitRate.populate,              {'display_progress': display_progress}),
        pacman_muscle.MotorUnitPsth:              (pacman_muscle.MotorUnitPsth.populate,              {'display_progress': display_progress}),
        pacman_muscle.Emg:                        (pacman_muscle.Emg.populate,                        {'display_progress': display_progress}),
        pacman_muscle.EmgEnvelope:                (pacman_muscle.EmgEnvelope.populate,                {'display_progress': display_progress}),
        pacman_muscle.EmgEnvelopeMean:            (pacman_muscle.EmgEnvelopeMean.populate,            {'display_progress': display_progress}),
    }

    for table in populate_functions.keys():

        # read table name and tier
        attributes = inspect.getmembers(table, lambda a:not(inspect.isroutine(a)))
        table_name = next(a[1] for a in attributes if a[0].startswith('full_table_name'))

        print('\n\nPopulating ' + table_name + '\n')
        time.sleep(0.2)

        func,  kwargs = populate_functions[table]
        func(**kwargs)