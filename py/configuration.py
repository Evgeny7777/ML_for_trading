from functools import wraps
import json
import pprint
import pickle
import time


from py.config import Config as conf
from py.config import PipelineStages
from py.exceptions import NoTrainingData, MissingConfigValues
from py.logger import l
from py.my_pd_utils import concat_dfs, exclude_fields, make_list_if_single
from py.mongo import Evaluation, Experiment, ModelCollection, Configuration, Point, connect_to_mongo


def consistency_check(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        result = method(self, *method_args, **method_kwargs)
        for key in conf.REQUIRED_CONFIG_VALUES:
            if not getattr(self, key, None):
                l.error(f'value {key} is not passed to ExperimentConfiguration')
                raise MissingConfigValues(f'key {key} is missing')
        return result
    return _impl

def set_default_values(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        result = method(self, *method_args, **method_kwargs)
        for key, value in conf.DEFAULT_CONFIG_VALUES.items():
            if not getattr(self, key, None):
                setattr(self, key, value)
                l.debug(f'Using default value {value} of {key}')
        return result
    return _impl

class ExperimentConfiguration:
    def __init__(self):
        self._update_revision()
        self.loaded_keys = set()
    
    @set_default_values
    def _load_from_dict(self, values):
        self._update_revision()
        for key, val in values.items():
            setattr(self, key, val)
        self.loaded_keys = self.loaded_keys.union(set(values.keys()))
        self.adjusted_parameters = []
        return self

    @set_default_values
    def _load_from_space_dict(self, values):
        self._update_revision()
        for key, val in values.items():
            setattr(self, key, make_list_if_single(val)[0])
        self.loaded_keys = self.loaded_keys.union(set(values.keys()))
        self.adjusted_parameters = []

    @consistency_check
    @set_default_values
    def load_from_json_file(self, fpath):
        self._update_revision()
        l.debug(f'reading config from {fpath}')
        j = json.load(open(fpath, 'r'))
        for key, val in j.items():
            setattr(self, key, val)
        self.loaded_keys.union(set(j.keys()))
        self.adjusted_parameters = []
    
    def load_from_experiment_by_id(self, experiment_object_id):
        ex = Experiment.objects(pk=experiment_object_id).first()
        if not ex:
            raise ValueError(f'Experiment {experiment_object_id} with status DONE is not found in DB')
        self.load_from_experiment_object(ex)

    @consistency_check
    @set_default_values
    def load_from_experiment_object(self, experiment_object):
        self._update_revision()
        ex = experiment_object
        self._load_from_space_dict(ex.space)
        self._load_from_dict(ex.data)
        self._load_from_dict({'MODEL_SPACE': experiment_object.model_space})
        l.debug(f'Configuration for experiment {experiment_object.id} is successfully loaded from DB')

    @consistency_check
    @set_default_values
    def load_from_experiment_step(self, experiment_object_id, step, fine_tuned=False):
        # load initial config first
        self.load_from_experiment_by_id(experiment_object_id)
        # take coordinates of the particular step
        if fine_tuned:
            p = Point.objects(experiment=experiment_object_id, step=step, fine_tuned=True).first()
        else:
            p = Point.objects(experiment=experiment_object_id, step=step, fine_tuned__in=[None, False]).first()
        self.adjust_parameters(**p.coordinates)
    
    
    @consistency_check
    @set_default_values
    def load_by_point_object(self, point_object):
        # load initial config first
        self.load_from_experiment_object(point_object.experiment)
        self.adjust_parameters(**point_object.coordinates)
        return self

    def load_by_point_id(self, point_object_id):
        # load initial config first
        p = Point.objects_safe(pk=point_object_id).first()
        self.load_by_point_object(p)


    @property
    def MAIN_LABEL(self):
        return f'time2{self.DEAL_TYPE}'
    
    @property
    def _as_dict(self):
        d = {}
        for k in self.loaded_keys:
            d[k] = getattr(self, k)
        return d

    @property
    def _as_json(self):
        return json.dumps(self._as_dict)

    def adjust_parameters(self, **kwargs):
        self._update_revision()
        for k, v in kwargs.items():
            if hasattr(self, k): 
                v = int(v) if (v%1==0) else v
                l.debug(f"{k} is updated {getattr(self, k)} --> {v}")
                
                setattr(self, k, v)
                self.adjusted_parameters.append(k)
            else:
                l.debug(f"{k} is not present among attributes, ignoring it")
    
    def save_to_file(self, file_path):
        l.debug(f'saving configuration to {file_path} ')
        pickle.dump(self, open(file_path, "wb"))
    
    @property
    def describe(self):
      return pprint.pformat(self.__dict__)
    
    @property
    def starting_stage(self):
        current_stage = 99
        for p in self.adjusted_parameters:
            if p.startswith('PREPROCESS'):
                current_stage = min(current_stage, PipelineStages.PREPROCESSING)
            if p.startswith('LABEL'):
                current_stage = min(current_stage, PipelineStages.LABEL)
            if p.startswith('MODEL'):
                current_stage = min(current_stage, PipelineStages.MODEL)
            if p.startswith('EVAL'):
                current_stage = min(current_stage, PipelineStages.EVAL)
        return current_stage

    def _update_revision(self):
        self.last_revision = int(time.time())
