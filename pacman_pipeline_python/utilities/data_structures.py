import itertools, re
import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr
from churchland_pipeline_python import acquisition
from sklearn import decomposition
from typing import List
from .. import pacman_acquisition

class NeuroDataArrayConstructor:

    def __init__(
        self,
        data: List[list],
        unit_ids: List[list]=None,
        condition_ids: list=None,
        condition_times: list=None,
        data_name: str='data',
        sample_rate: int=1,
        ):
        self.data = data
        self.unit_ids = unit_ids
        self.condition_ids = condition_ids
        self.condition_times = condition_times
        self.data_name = data_name
        self.sample_rate = sample_rate
        self._read_data()
        self._read_unit_ids()
        self._read_condition_ids()
        self._read_condition_times()
        self._make_data_set()
        self._cleanup()
        self.reset_data_set()

    def _read_data(self):
        assert isinstance(self.data, list) and all(isinstance(X,list) for X in self.data), \
            'Expected data as a list of lists'
        self._data_stats = {
            'n_sessions': len(self.data),
            'n_conditions': len(self.data[0]),
            'n_units_per_session': [X[0].shape[0] for X in self.data],
            'n_samples_per_condition': [X.shape[1] for X in self.data[0]],
            'multi_trial': any([Y.ndim == 3 for X in self.data for Y in X])
        }
        assert all(len(X)==self._data_stats['n_conditions'] for X in self.data), \
            'Mismatched number of conditions across sessions'
        for session_data in self.data:
            assert np.array_equal([X.shape[1] for X in session_data], self._data_stats['n_samples_per_condition']), \
                'Mismatched condition durations across session'
        for X, unit_count in zip(self.data, self._data_stats['n_units_per_session']):
            assert all(xi.shape[0]==unit_count for xi in X), \
                'Mismatched number of units across conditions'

    def _read_unit_ids(self):
        if self.unit_ids is None:
            self._set_default_unit_ids()
        else:
            self._validate_user_unit_ids()

    def _read_condition_ids(self):
        if self.condition_ids is None:
            self._set_default_condition_ids()
        else:
            self._validate_user_condition_ids()

    def _read_condition_times(self):
        if self.condition_times is None:
            self._set_default_condition_times()
        else:
            self._validate_user_condition_times()

    def _make_data_set(self):
        session_arrays = []
        for session_data, unit_ids in zip(self.data, self.unit_ids):

            condition_arrays = []
            for condition_data, condition_id, condition_time in zip(session_data, self.condition_ids, self.condition_times):
                unit_coords = self._make_session_unit_coords(unit_ids)
                time_coords = self._make_condition_time_coords(condition_id, condition_time)
                coords = [unit_coords] + [time_coords]
                if self._data_stats['multi_trial']:
                    condition_data = np.atleast_3d(condition_data)
                    coords.append(('trial', np.arange(condition_data.shape[2])))

                condition_arrays.append(xr.DataArray(condition_data, coords=coords))
            
            session_arrays.append(xr.concat(condition_arrays, dim=time_coords[0]))
        
        self._raw_data_set = xr.Dataset(
            {self.data_name: xr.concat(session_arrays, dim=unit_coords[0])}, 
            attrs={'sample_rate': self.sample_rate}
        )

    def _make_session_unit_coords(self, unit_ids):
        unit_indexes, unit_index_names = self._ids_to_indexes(unit_ids)
        unit_indexes = self._correct_coord_datatypes(unit_ids[0], unit_indexes, unit_index_names)
        if len(unit_index_names) == 1:
            unit_coords = (unit_index_names[0], unit_indexes[0])
        else:
            unit_multi_index = pd.MultiIndex.from_arrays(unit_indexes, names=unit_index_names)
            unit_coords = ('_'.join(unit_index_names), unit_multi_index)
        return unit_coords

    def _make_condition_time_coords(self, condition_id, condition_time):
        condition_indexes, condition_index_names = self._ids_to_indexes(condition_id)
        condition_indexes = self._correct_coord_datatypes(condition_id, condition_indexes, condition_index_names)
        condition_indexes = [np.tile(cidx, len(condition_time)) for cidx in condition_indexes]
        condition_indexes += [condition_time]
        condition_index_names += ['time']
        condition_time_multi_index = pd.MultiIndex.from_arrays(condition_indexes, names=condition_index_names)
        return ('_'.join(condition_index_names), condition_time_multi_index)

    def _ids_to_indexes(self, coord_ids):
        index_names = np.vstack(coord_ids)[:,0]
        index_values = np.vstack(coord_ids)[:,1]
        unique_index_names = list(np.unique(index_names))
        grouped_index_values = [index_values[index_names==name] for name in unique_index_names]
        return grouped_index_values, unique_index_names

    def _correct_coord_datatypes(self, coord_id, index_values, index_names):
        coord_dtypes = {x[0]: type(x[1]) for x in coord_id}
        for idx, name in enumerate(index_names):
            index_values[idx] = index_values[idx].astype(coord_dtypes[name])
        return index_values

    def _set_default_unit_ids(self):
        if self._data_stats['n_sessions'] == 1:
            self.unit_ids = [[('unit', unit)] for unit in range(self._data_stats['n_units_per_session'][0])]
        else:
            self.unit_ids = []
            for session in range(self._data_stats['n_sessions']):
                self.unit_ids.append([
                    [('session',session), ('unit',unit)] for unit in range(self._data_stats['n_units_per_session'][session])
                ])

    def _validate_user_unit_ids(self):
        assert isinstance(self.unit_ids, list) and all(isinstance(X,list) for X in self.unit_ids), \
            'Expected unit IDs as a list of lists'
        assert len(self.unit_ids) == self._data_stats['n_sessions'], \
            'Mismatched number of sessions and sets of units'
        assert np.array_equal([len(X) for X in self.unit_ids], self._data_stats['n_units_per_session']), \
            'Mismatched units within sessions'
        reformatted_unit_ids = []
        for session_unit_ids in self.unit_ids:
            if all(isinstance(X,list) for X in session_unit_ids):
                for unit_ids in session_unit_ids:
                    assert(all(isinstance(uid,tuple)) for uid in unit_ids), \
                        'Expected a list of list of tuples'                        
                reformatted_unit_ids.append(session_unit_ids)
            if all(isinstance(X,tuple) for X in session_unit_ids):
                reformatted_unit_ids.append([[uid] for uid in session_unit_ids])
            if all(isinstance(X,int) or isinstance(X,str) for X in session_unit_ids):
                reformatted_unit_ids.append([[('unit',uid)] for uid in session_unit_ids])
        self.unit_ids = reformatted_unit_ids
    
    def _set_default_condition_ids(self):
        self.condition_ids = [[('condition',cid)] for cid in range(self._data_stats['n_conditions'])]

    def _validate_user_condition_ids(self):
        assert isinstance(self.condition_ids, list), 'Expected a list of condition IDs'
        if all(isinstance(cid,int) or isinstance(cid,str) for cid in self.condition_ids):
            self.condition_ids = [[('condition',cid)] for cid in self.condition_ids]
        else:
            for condition_id in self.condition_ids:
                assert all(isinstance(cid,tuple) for cid in condition_id), \
                    'Expected a list of tuples'

    def _set_default_condition_times(self):
        self.condition_times = [np.arange(T)/self.sample_rate for T in self._data_stats['n_samples_per_condition']]

    def _validate_user_condition_times(self):
        assert np.array_equal([len(t) for t in self.condition_times], self._data_stats['n_samples_per_condition']), \
            'Mismatched condition time vectors and data dimensions'

    def _cleanup(self):
        delattr(self, 'data')
        delattr(self, 'unit_ids')
        delattr(self, 'condition_ids')
        delattr(self, 'condition_times')
        delattr(self, 'data_name')
        delattr(self, 'sample_rate')
        delattr(self, '_data_stats')

    def reset_data_set(self):
        self.data_set = self._raw_data_set.copy(deep=True)

    @classmethod
    def from_datajoint_table(
        cls,
        table, 
        data_name: str,
        unit_names: List[str]=None,
        new_sample_rate: int=None,
        ):
        # unit ID names
        unit_id_names = unit_names if unit_names is not None else []

        session_keys = (dj.U('session_date') & table).fetch('KEY')
        if len(session_keys) > 1 or unit_names is None:
            unit_id_names.insert(0, 'session_date')

        # condition IDs
        condition_keys = (pacman_acquisition.ConditionParams & table).fetch('KEY')
        condition_ids = [key['condition_id'] for key in condition_keys]

        # fetch condition times
        try:
            sample_rates = list(np.unique(table.fetch('sample_rate')))
            fetch_times = True
        except Exception:
            fetch_times = False
            new_condition_times = None
            new_sample_rate = 1
        else:
            assert len(sample_rates)==1 or new_sample_rate is not None, \
                'Multiple sample rates in table and new rate unspecified'
            if new_sample_rate is not None:
                sample_rates.append(new_sample_rate)
            else:
                new_sample_rate = sample_rates[0]

            condition_times = {}
            for cid, fs in itertools.product(condition_ids, sample_rates):
                t, _ = pacman_acquisition.ConditionParams.target_force_profile(cid, fs)
                condition_times.update({(cid, fs): t})

            new_condition_times = [condition_times[(cid, new_sample_rate)] for cid in condition_ids]

        # check multi trial
        is_multi_trial = 'trial' in table.primary_key

        session_data = []
        unit_ids = []
        for session_key in session_keys:

            unit_keys = (dj.U(*unit_id_names) & table & session_key).fetch(as_dict=True, order_by=unit_id_names)
            unit_ids.append([[(name, key[name]) for name in unit_id_names] for key in unit_keys])

            condition_data = []
            for condition_key in condition_keys:
                condition_table = table & session_key & condition_key
                if is_multi_trial:
                    if fetch_times:
                        trial, sample_rates, X = condition_table.fetch('trial', 'sample_rate', data_name, order_by=unit_id_names)
                        for idx, fs in enumerate(sample_rates):
                            if fs != new_sample_rate:
                                t_old = condition_times[(condition_key['condition_id'], fs)]
                                t_new = condition_times[(condition_key['condition_id'], new_sample_rate)]
                                X[idx] = np.interp(t_old, t_new, X[idx])
                    else:
                        trial, X = condition_table.fetch('trial', data_name, order_by=unit_id_names)
                    X = np.vstack(X)
                    X = [X[trial==tr,:] for tr in np.unique(trial)]
                    condition_data.append(np.stack(X, axis=2))
                else:
                    if fetch_times:
                        sample_rates, X = condition_table.fetch('sample_rate', data_name, order_by=unit_id_names)
                        for idx, fs in enumerate(sample_rates):
                            if fs != new_sample_rate:
                                t_old = condition_times[(condition_key['condition_id'], fs)]
                                t_new = condition_times[(condition_key['condition_id'], new_sample_rate)]
                                X[idx] = np.interp(t_old, t_new, X[idx])
                    else:
                        X = condition_table.fetch(data_name, order_by=unit_id_names)
                    condition_data.append(np.vstack(X))

            session_data.append(condition_data)

        return cls(session_data, unit_ids, condition_ids, new_condition_times, data_name, new_sample_rate)

class NeuroDataArray(NeuroDataArrayConstructor):

    def __init__(
        self,
        data: List[list],
        unit_ids: List[list]=None,
        condition_ids: list=None,
        condition_times: list=None,
        data_name: str='data',
        sample_rate: int=1,
        ):
        super().__init__(
            data, 
            unit_ids, 
            condition_ids, 
            condition_times, 
            data_name, 
            sample_rate,
        )

    def mean_center(self, only_vars: list=None, reference_var: str=None):
        """Centers each data array by its cross-condition mean.
        If only_vars is provided, centering will only be applied to the corresponding arrays.
        If reference_var is provided, centering uses the cross-condition mean of the corresponding array."""
        data_vars = list(self.data_set.data_vars)
        if only_vars is not None:
            unrecognized_variables = list(set(only_vars) - set(data_vars))
            assert not any(unrecognized_variables), \
                'Attributes ' + ' '.join(['{}']*len(unrecognized_variables)) + ' not found in data set'
            data_vars = list(set(data_vars) & set(only_vars))

        time_dim_name = next(x for x in list(self.data_set.dims.keys()) if re.match(r'.*time',x) is not None)
        data_means = {var: self.data_set[var].mean(dim=time_dim_name) for var in data_vars}
        if reference_var is not None:
            assert reference_var in data_vars, \
                'Reference variable {} not found in data set'.format(reference_var)
            data_means = {var: data_means[reference_var] for var in data_vars}

        for var in data_vars:
            self.data_set[var] -= data_means[var]

    def normalize(self, only_vars: list=None, reference_var: str=None, soft_factor: int=0):
        """Normalizes each data array by its cross-condition range.
        If only_vars is provided, normalization will only be applied to the corresponding arrays.
        If reference_var is provided, normalization uses the cross-condition range of the corresponding array.
        If soft_factor is provided, soft normalizes as x -> x/(range(x) + soft_factor)."""
        data_vars = list(self.data_set.data_vars)
        if only_vars is not None:
            unrecognized_variables = list(set(only_vars) - set(data_vars))
            assert not any(unrecognized_variables), \
                'Attributes ' + ' '.join(['{}']*len(unrecognized_variables)) + ' not found in data set'
            data_vars = list(set(data_vars) & set(only_vars))

        time_dim_name = next(x for x in list(self.data_set.dims.keys()) if re.match(r'.*time',x) is not None)
        data_ranges = {var: self.data_set[var].max(dim=time_dim_name) - self.data_set[var].min(dim=time_dim_name) for var in data_vars}
        if reference_var is not None:
            assert reference_var in data_vars, \
                'Reference variable {} not found in data set'.format(reference_var)
            data_ranges = {var: data_ranges[reference_var] for var in data_vars}

        for var in data_vars:
            self.data_set[var] /= (soft_factor + data_ranges[var])

    def project_pc_space(self, var_name: str, condition_ids: list=None, soft_factor: int=0, max_components: int=None):
        """Projects a data array into its principal component space and saves as a new data array.
        If condition_ids is provided, uses those IDs to get the principal components.
        Limits PCA to the top max_components, if provided.
        Note: resets the data set, then mean centers and (soft) normalizes the source data array with soft_factor."""
        condition_time_dim_name = next(index for index in list(self.data_set.indexes) if re.match(r'.*time',index))
        condition_dim_name = re.match(r'(.*)_time', condition_time_dim_name).group(1)
        if condition_ids is None:
            condition_ids = np.unique(self.data_set[condition_dim_name].values)

        self.reset_data_set()
        self.mean_center(only_vars=[var_name])
        self.normalize(only_vars=[var_name], soft_factor=soft_factor)

        subspace_data = self.data_set[var_name].sel(condition_time=(condition_ids, np.s_[::])).values
        pca = decomposition.PCA(n_components=max_components).fit(subspace_data.T)
        data_projection = pca.components_ @ self.data_set[var_name].values

        self.data_set[var_name + '_projection'] = xr.DataArray(
            data_projection,
            coords = [
                ('principal_component', 1+np.arange(pca.components_.shape[0])),
                (condition_time_dim_name, self.data_set.indexes[condition_time_dim_name])
            ],
            attrs = {'subspace_condition_ids': condition_ids}
        )
        
    def reorder_conditions(self, sorted_condition_ids: list):
        """Down selects and reorders data set by provided condition IDs."""
        condition_time_dim_name = next(index for index in list(self.data_set.indexes) if re.match(r'.*time',index))
        condition_dim_name = re.match(r'(.*)_time', condition_time_dim_name).group(1)

        ct_indexes = np.vstack([np.array(ct) for ct in self.data_set.indexes[condition_time_dim_name]])
        sorted_ct_indexes = np.vstack([ct_indexes[ct_indexes[:,0]==cid,:] for cid in sorted_condition_ids])
        sorted_ct_indexes = [ct for ct in sorted_ct_indexes.T]
        sorted_ct_indexes[0] = sorted_ct_indexes[0].astype(int)
        new_ct_indexes = pd.MultiIndex.from_arrays(sorted_ct_indexes, names=[condition_dim_name, 'time'])
        self.data_set = self.data_set.reindex(condition_time=new_ct_indexes)

    def trial_average(self, n_folds: int=1):
        assert 'trial' in self.data_set.coords, 'Only applicable for multi-trial data'
        fold_ids = np.tile(np.arange(n_folds), np.ceil(self.data_set.dims['trial']/n_folds).astype(int))[:self.data_set.dims['trial']]
        self.data_set = self.data_set.assign_coords(fold=('trial', fold_ids)).groupby('fold').mean(dim='trial', skipna=True)