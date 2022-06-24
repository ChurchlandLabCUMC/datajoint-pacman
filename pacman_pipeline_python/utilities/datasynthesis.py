"""Utilities for pulling data into the pipeline."""

import datajoint as dj
import os, re
from src.churchland_pipeline_python import acquisition, equipment, reference
from datetime import datetime
from typing import List, Tuple


def get_data_path(monkey: str, data_type: str='raw') -> str:
    """Get path to raw or processed data."""

    assert data_type in ('raw','processed'), 'Data type {} unrecognized'.format(data_type)

    # get local path to locker
    local_path = (reference.EngramTier & {'engram_tier': 'locker'}).get_local_path()

    # get local path to raw data (rig > task > monkey)
    data_path_parts = ['Jumanji', 'pacman-task', monkey.lower(), data_type, '']
    
    return local_path + os.path.sep.join(data_path_parts)


def get_sessions(monkey: str, data_type: str='raw') -> List[str]:
    """Get list of raw or processed session dates."""

    # get raw directories
    data_path = get_data_path(monkey, data_type=data_type)
    data_dir = sorted(list(os.listdir(data_path)))

    # restrict to valid dates
    def isdate(x):
        try:
            datetime.strptime(x,'%Y-%m-%d')
        except ValueError:
            return None
        else:
            return True

    session_dates = [d for d in data_dir if isdate(d)]

    return session_dates, data_path


def parse_notes(key, read_type: Tuple[str]=('brain', 'emg')):
    """Parses notes file to extract brain and emg channel group metadata."""

    brain_attr = None
    emg_attr = None

    note = (acquisition.Session.Notes & key).fetch1('session_notes')

    if ('brain' in read_type) and (acquisition.EphysRecording.Channel & key & {'ephys_channel_type': 'brain'}):

        # initialize
        brain_attr = [dict(
            **key, 
            brain_region_abbr='M1',
            brain_channel_group_id=0,
            chamber_id=0,
            burr_hole_id=0,
            brain_hemisphere='right'
        )]

        # electrode array model
        electrode_array_model = None
        if 'Neuropixels' in note:
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'Neuropixels', 'electrode_array_model_version': 'nhp demo'}).fetch1('KEY')

        elif re.search('S(-|\s)(P|p)robe', note):
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'S-Probe'}).fetch1('KEY')

        elif re.search('V(-|\s)(P|p)robe', note):
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'V-Probe'}).fetch1('KEY')

        if electrode_array_model:
            electrode_array = (equipment.ElectrodeArray & electrode_array_model & {'electrode_array_id': 0}).fetch1('KEY')
            brain_attr[0].update(**electrode_array)

        # probe depth
        probe_depth = re.search(r'(depth|lowered|inserted).*?(\d+\.?\d*)\s?mm', note)
        if probe_depth:
            brain_attr[0].update(probe_depth=float(probe_depth.group(2)))

    if ('emg' in read_type) and (acquisition.EphysRecording.Channel & key & {'ephys_channel_type': 'emg'}):

        # initialize
        emg_attr = dict(**key)

        # electrode array model
        if any(x in note for x in ['QF','quadrifilar','quad']):
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'Hook-Wire', 'electrode_array_model_version': 'quad'}).fetch1('KEY')

        elif 'clipped' in note:
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'Hook-Wire', 'electrode_array_model_version': 'clipped'}).fetch1('KEY')

        else:
            electrode_array_model = (equipment.ElectrodeArrayModel \
                & {'electrode_array_model': 'Hook-Wire', 'electrode_array_model_version': 'paired'}).fetch1('KEY')

        electrode_array = (equipment.ElectrodeArray & electrode_array_model & {'electrode_array_id': 0}).fetch1('KEY')
        emg_attr.update(**electrode_array)
        
        # read muscles and their positions
        muscle_abbr = []
        muscle_order = []
        for muscle_attr in reference.Muscle().fetch():
            full_muscle_name = ' '.join(list(muscle_attr)[:0:-1])
            match = re.search(full_muscle_name, note.lower())
            if match:
                muscle_abbr.append(muscle_attr[0])
                muscle_order.append(match.start())

        if muscle_abbr:
            
            # sort list
            muscle_abbr = [x for _, x in sorted(zip(muscle_order, muscle_abbr))]

            emg_attr = [dict(**emg_attr, emg_channel_group_id=idx, muscle_abbr=abbr) \
                for idx,abbr in enumerate(muscle_abbr)]

        else:
            emg_attr = [emg_attr]

    return brain_attr, emg_attr






