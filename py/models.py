from abc import ABC, abstractclassmethod
from datetime import datetime, timezone
from functools import wraps
import pickle
import os 
import time
from typing import List

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit

from py.classes import Evaluation_results
from py.dataset import Dataset, get_datasets_from_exp_config, filter_datasets_with_tags
from py.config import Config as conf
from py.configuration import ExperimentConfiguration
from py.exceptions import NoTrainingData, MissingConfigValues, ClassNotFound, WrongParameter, RecordNotFound, BadLogic, MalformedModel
from py.logger import l
from py.mongo import Evaluation, Experiment, ModelCollection, Configuration, Point, connect_to_mongo
from py.my_pd_utils import concat_dfs, exclude_fields, make_list_if_single
from py.s3 import S3_client


class AbstractModel(ABC):
    @abstractclassmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def train(self, datasets: List[Dataset], tags: List[str], **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, X: np.ndarray):
        raise NotImplementedError

    @abstractclassmethod
    def predict_latest(self, dataset:Dataset):
        raise NotImplementedError

    def evaluate(self, datasets: List[Dataset], tags):
        """
            Stage #4: Evaluate results
        """
        l.debug('='*50)
        l.debug(f'Start evaluation with tags={tags}')
        relevant_datasets = filter_datasets_with_tags(
            make_list_if_single(datasets), tags)


        local_conf = self.exp_config
        not_feature_fields = conf.BASE_FIELDS+[f'time2{local_conf.DEAL_TYPE}', f'{local_conf.DEAL_TYPE}_profit']
        full_result = Evaluation_results()
        all_results = []
        for dataset in relevant_datasets:
            Y_pred = self.predict(dataset.X)
            df_deals  = dataset._data[not_feature_fields].assign(deal_pred = Y_pred.astype(bool))
            df_deals = df_deals[df_deals.deal_pred]
            expected_profits, deal_durations = [], []
            
            if local_conf.DEAL_TYPE == 'sell':
                profit_foo = dataset.expected_sell_profit
                duration_function = dataset.expected_sell_duration
            elif local_conf.DEAL_TYPE == 'buy':
                profit_foo = dataset.expected_buy_profit
                duration_function = dataset.expected_buy_duration
            else:
                ValueError(f'DEAL_TYPE shall be buy or sell. got {local_conf.DEAL_TYPE}')
            
            df_deals.index -= pd.Timedelta(minutes=dataset.shift)
            for dt in df_deals.index:
                dt_from = dt
                dt_to = dt + pd.Timedelta(minutes=local_conf.PREPROCESS_PERIOD_MINUTES*local_conf.LABEL_MAX_LENGTH)

                if local_conf.LABEL_EVAL_PARAM_SHARING: 
                    param_prefix = 'LABEL_'
                else:
                    param_prefix = 'EVAL_'

                exact_expected_profit = profit_foo(
                    dataset.source.df.loc[dt_from:dt_to].price, 
                    getattr(local_conf, param_prefix + 'TRAILING_STOP'),
                    getattr(local_conf, param_prefix + 'STOP_LOSS'),
                    getattr(local_conf, param_prefix + 'TAKE_PROFIT')
                )
                expected_profits.append(exact_expected_profit)

                deal_duration = duration_function(
                    dataset.source.df.loc[dt_from:dt_to].price, 
                    getattr(local_conf, param_prefix + 'TRAILING_STOP'),
                    getattr(local_conf, param_prefix + 'STOP_LOSS'),
                    getattr(local_conf, param_prefix + 'TAKE_PROFIT')
                )
                deal_durations.append(deal_duration)
            
            df_deals = df_deals.assign(profit=expected_profits, duration=deal_durations)
            
            if df_deals.shape[0] == 0: # if there are no deals
                res = Evaluation_results(None, code=dataset.source.name, days=np.unique(dataset._data.index.date).shape[0])
                df_real_deals = df_deals
            else:
                df_deals['deal_end'] = df_deals.index + pd.to_timedelta(df_deals.duration, unit='s')
                df_deals.sort_index(inplace=True)

                # remove overlapping deals
                start = True
                real_deals = []
                for idx, row in df_deals.iterrows():
                    if start:
                        real_deals.append(True)
                        start=False
                        deal_end = row.deal_end
                    elif deal_end > idx:
                        # cannot open another deal
                        real_deals.append(False)
                    else:
                        deal_end = row.deal_end
                        real_deals.append(True)
                df_real_deals = df_deals[real_deals]
                df_real_deals = df_real_deals.assign(real_profit=(df_real_deals.profit-local_conf.EVAL_COMMISSION_PER_DEAL))
                res = Evaluation_results(df_real_deals.real_profit, code=dataset.source.name, days=np.unique(dataset._data.index.date).shape[0])

            all_results.append(res)
            full_result += res

            l.debug('-'*10)
            l.debug(f'Evaluating on {dataset.source.code}, shift {dataset.shift}')
            l.debug(f'{res.count} deals')
            l.debug(f'avg deal profit: {res.mean}')
            
        l.debug('-'*50)
        l.debug(f'Overall profit: {full_result.total}')
        l.debug(f'Overall average deal profit: {full_result.mean:.1f}')
        l.debug(f'Overall std: {full_result.std:.1f}')
        l.debug(f'Overall deals: {full_result.count}')
        return full_result, all_results 

    @staticmethod
    def _generate_model_id() -> str:
        return datetime.strftime(datetime.now(timezone.utc), '%y%m%d.%H%M%S.%f')

    @abstractclassmethod
    def _load_self_from_mongo_object(self, obj:ModelCollection):
        raise NotImplementedError

    @abstractclassmethod
    def attach_data(self, dataset_factory):
        raise NotImplementedError
    
    @abstractclassmethod
    def update_attached_data(self):
        raise NotImplementedError
    
    @abstractclassmethod
    def save_to_cloud(self, *args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def drop_from_cloud(obj:ModelCollection):
        _id = obj.id
        obj.delete()
        l.debug(f'Model with id {_id} is deleted successfully from mongo')
    
    def load_from_cloud(self, *args, **kwargs):
        raise NotImplementedError


class SKModel(AbstractModel):

    def __init__(self, experiment_configuration:ExperimentConfiguration=None):
        if not experiment_configuration is None:
            self.exp_config = experiment_configuration
        self.s3 = S3_client()
        self.obj:ModelCollection = None
        self.was_trained = False
        self.read_only = False
        
    def init_from_point(self, point_object=None, point_object_id=None):
        if not self.obj is None:
            raise BadLogic('Cannot call init_from_point on a used model instance')
        if not point_object is None:
            p = point_object
        elif not point_object_id is None:
            p = Point.objects_safe(pk=point_object_id).first()
        else:
            raise ValueError('At least one of point_object or point_object_id must be provided')
        code = p.experiment.code or list(p.experiment.data['DATASOURCES'].keys())[0][:2]
        
        self.obj = ModelCollection()
        self.obj.model_id = f'{code}.{self._generate_model_id()}'
        self.obj.status =  'READY'
        self.obj.model_type = ModelTypeMatcher.get_type_name(self)
        self.obj.point = p
        self.obj.experiment = p.experiment.id
        self.obj.step = p.step
        self.exp_config = ExperimentConfiguration().load_by_point_object(p)
        self.obj.config = self.exp_config._as_dict
        self.obj.created_at = datetime.now(timezone.utc)
        self.obj.code = code
        l.debug(f'model is loaded from the point {p.id}')

    def train_preloaded(self, datasets=None):
        if datasets is None: 
            datasets_local = get_datasets_from_exp_config(self.exp_config)
        else:
            datasets_local = datasets
        self.train(datasets_local, 'ALL', model_params=self.obj.point.clf_params)
        self.obj.status='TRAINED'
        l.debug(f'Model is trained with preloaded setup')
        self.was_trained = True

        file_key = conf.AWS_MODELS_FOLDER + self.obj.model_id  + '.pkl'
        model_file_str = f's3:{file_key}'
        
    def attach_data(self, dataset_factory):
        self._dataset:Dataset = dataset_factory(self.exp_config)
    
    def update_attached_data(self, update_source):
        self._dataset.preprocess(cache=False)
        self._dataset.update(update_source)

    @property
    def latest_time(self):
        if not hasattr(self, '_dataset') or self._dataset is None:
            raise BadLogic('to call latest_time() method attach dataset first')
        return self._dataset.latest_time

    def train(self, datasets, tags, model_params=None, model_param_distributions=None, random_search_iterations=8):
        """
            model_params - dict with params for a classifier
            model_param_distributions -  dict with serach ranges for params of model

            Provide one ond only one of them
        """
        relevant_datasets = filter_datasets_with_tags(datasets, tags)
        # ==== make id section
        self._id  = f'{relevant_datasets[0].source.code[:2]}.{self._generate_model_id()}'
        X = np.concatenate([d.X for d in relevant_datasets])
        Y = np.concatenate([d.Y for d in relevant_datasets])
        if X.shape[0] == 0: raise NoTrainingData

        if not model_param_distributions is None and not model_params is None:
            raise ValueError('Only one of model_params and model_param_distributions must be provided')

        if model_params:
            l.debug('Init Random forest classifier')
            clf = RandomForestClassifier(**model_params)
        elif model_param_distributions:
            l.debug(f'Init RandomSearch of Random forest config. Space is {model_param_distributions}')
            # sn1 = tracemalloc.take_snapshot()
            clf = RandomizedSearchCV(
                                RandomForestClassifier(), 
                                param_distributions=model_param_distributions, 
                                cv=TimeSeriesSplit(n_splits=2), 
                                iid=False, 
                                n_jobs=conf.SK_JOBS, 
                                pre_dispatch=1,
                                n_iter=random_search_iterations,
                                random_state=143,
                                verbose=1)
            # connect_to_mongo() # reconnect after a long operation
        else:
            raise ValueError('At least one of model_params and model_param_distributions must be provided')

        l.debug('Start fit of classifier')
        start = time.time()
        clf.fit(X, Y)
        l.debug(f'Training a classifier took {time.time() - start}.') 
        if model_param_distributions: l.debug(f'Best params are {clf.best_params_}')

        self.clf = clf 

    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_latest(self, dataset:Dataset=None):
        if not hasattr(self, '_dataset') and dataset is None:
            raise WrongParameter('Dataset has to be attached of provided')
        if dataset is None:
            result = self.predict(self._dataset.X_latest_row)
        else:
            result = self.predict(dataset.X_latest_row)
        return result

    #-------saving and restoring
    def save_to_cloud(self):
        """
            Called when model was created locally and it's cloud representation to be created
        """
        if self.read_only: raise WrongParameter('Cannot save model in read only mode')
        if self.was_trained:
            file_key = conf.AWS_MODELS_FOLDER + self.obj.model_id  + '.pkl'
            self.s3.upload_binary(pickle.dumps(self.clf), file_key)
            self.obj.model_file = f's3:{file_key}'

        self.obj.save_safe()

        l.debug(f'Model with id={self.obj.model_id } is saved')
        return self.obj.model_id 

    #-------saving and restoring
    def drop_me_from_cloud(self):
        self.drop_from_cloud(self.obj, self.s3)
    
    @staticmethod
    def drop_from_cloud(obj:ModelCollection, s3_client:S3_client=None):
        # may take s3 client instance as param not to make multiple ones
        s3 = s3_client
        if s3 is None: s3 = S3_client()
        if obj.model_file.split(':')[0] == 's3':
            key = obj.model_file.split(':')[1]
            s3.remove_file(key)
        
        return super().drop_from_cloud(obj)
    
    def _load_self_from_mongo_object(self, obj:ModelCollection, read_only:bool=True):
        if not self.obj is None:
            raise BadLogic('Cannot load model instance twice')
        
        self.read_only = read_only
        
        self.obj = obj
        self.exp_config = ExperimentConfiguration()._load_from_dict(obj.config)
        key = obj.model_file.split(':')[1]
        self.clf = pickle.loads(self.s3.read_binary(key))
        return self


class AllAgreeEnsembleModel(AbstractModel):
    def __init__(self):
        self.obj = None
        self.models = None

    def predict(self, X):
        if self.models is None:
            raise BadLogic('Model has to be loaded before being able to predict')

        result = None

        for model in self.models:
            Y = model.predict(X)
            if result is None: 
                result = np.full_like(Y, True)
            result = np.logical_and(result, Y)
        
        return result
    
    def _load_self_from_mongo_object(self, obj:ModelCollection):
        if not self.obj is None:
            raise BadLogic('Cannot load model instance twice')
        self.obj = obj
        model_ensemble = getattr(obj, 'model_ensemble', None)
        if model_ensemble is None or len(obj.model_ensemble) == 0:
            raise MalformedModel('Model has to have "model_ensemble" field')
        if len(obj.model_ensemble) == 1:
            l.warning(f'Strange ensemble, that has only one model. Obj.id={obj.id}')
        
        self.models = []
        i = 0
        for model_obj in model_ensemble:
            self.models.append(ModelInterface.load_from_mongo_object(model_obj))
            i += 1
        
        l.debug(f'Successfully loaded ensemble model, that has {i} members. obj.id={obj.id}')
    
    def init_with_models(self, model_objects:List[ModelCollection], description=None):
        # check that models are compatible between each other
        # code is the same
        code = model_objects[0].code
        deal_type = model_objects[0].config['DEAL_TYPE']
        for model_obj in model_objects:
            if code != model_obj.code:
                raise WrongParameter('All the models in ensemble shall have the same code')
            if deal_type != model_objects[0].config['DEAL_TYPE']:
                raise WrongParameter('All the models in ensemble shall have the same deal type')
        
        i = 0
        self.models = []
        for model_obj in model_objects:
            self.models.append(ModelInterface.load_from_mongo_object(model_obj))
            i += 1

        self.obj = ModelCollection()
        self.obj.model_id = f'{code}.{self._generate_model_id()}'
        self.obj.status =  'READY'
        self.obj.model_type = ModelTypeMatcher.get_type_name(self)
        self.obj.created_at = datetime.now(timezone.utc)
        self.obj.code = code
        self.obj.description = description
        self.obj.model_ensemble = model_objects
        l.debug(f'Ensemble model is created with id {self.obj.model_id}. It has {i} models in ensemble')
        return self

    def predict_latest(self):
        if self.models is None:
            raise BadLogic('Model has to be loaded before being able to predict')

        result = None

        for model in self.models:
            Y = model.predict_latest()
            if result is None: 
                result = np.full_like(Y, True)
            result = np.logical_and(result, Y)
        
        return result

    def attach_data(self, dataset_factory):
        l.debug('Attaching dataset to and ensemble model')
        for model in self.models:
            model.attach_data(dataset_factory)
    
    def update_attached_data(self, update_source):
        l.debug('Updating dataset attached to ensemble model')
        for model in self.models:
            model.update_attached_data(update_source)

    @property
    def latest_time(self):
        if self.models is None: raise BadLogic('Load ensemble of models first')
        return self.models[0].latest_time

    @property
    def exp_config(self):
        if self.models is None: raise BadLogic('Load ensemble of models first')
        return self.models[0].exp_config

     #-------saving and restoring
        
    def save_to_cloud(self):
        """
            Called when model was created locally and it's cloud representation to be created

        """

        self.obj.save_safe()

        l.debug(f'Model with id={self.obj.model_id } is saved')
        return self.obj.model_id 


class ModelTypeMatcher:
    type_to_class_mapping = {
        'scikit':SKModel,
        'all_agree_ensemble': AllAgreeEnsembleModel
    }

    @classmethod
    def get_class(cls, type_name):
        return cls.type_to_class_mapping[type_name]

    @classmethod
    def get_type_name(cls, model_instance):
        found_type_name = None
        for  type_name, ModelClass in cls.type_to_class_mapping.items():
            if isinstance(model_instance, ModelClass):
                if found_type_name is None:
                    found_type_name = type_name
                else:
                    raise WrongParameter('More than one class corresponds to {model_instance}. Do you pass parent?')

        if found_type_name is None: raise ClassNotFound(f'Have no matching types for {model_instance}')
        return found_type_name


def model_id_or_obj_id_provided(method):
    @wraps(method)
    def _impl(cls, *method_args, **method_kwargs):
        if not ('model_id' in method_kwargs or 'object_id' in method_kwargs):
            raise WrongParameter('Provide at least one of model_id or object_id')
        return method(cls, *method_args, **method_kwargs)
    return _impl

class ModelInterface:
    
    @classmethod
    @model_id_or_obj_id_provided
    def load_from_cloud(cls, model_id:str=None, object_id=None) -> AbstractModel:
        obj = None
        if not model_id is None:
            l.debug(f'Loading model with model_id={model_id}')
            obj = ModelCollection.objects_safe(model_id=model_id).first()
        elif not object_id is None:
            l.debug(f'Loading model with object_id={object_id}')
            obj = ModelCollection.objects_safe(pk=object_id).first()
        if not obj: raise RecordNotFound('Cannot find model')

        model_instance = cls.load_from_mongo_object(obj)
        l.debug(f'Model is loaded successfully')
        return model_instance
    
    @classmethod
    def load_from_mongo_object(cls, obj:ModelCollection) -> AbstractModel:
        model_instance = cls.create_new_instance(obj.model_type)
        model_instance._load_self_from_mongo_object(obj)
        return model_instance

    @classmethod
    def create_new_instance(cls, model_type) -> AbstractModel:
        ModelClass = ModelTypeMatcher.get_class(model_type)
        l.debug(f'Creating instance of {ModelClass}')
        return ModelClass()

    @classmethod
    def drop_from_cloud(cls, model_id:str=None, object_id=None) -> bool:
        obj = None
        if not model_id is None:
            l.debug(f'Loading model with model_id={model_id}')
            obj = ModelCollection.objects_safe(model_id=model_id).first()
        elif not object_id is None:
            l.debug(f'Loading model with object_id={object_id}')
            obj = ModelCollection.objects_safe(pk=object_id).first()
        if not obj: raise RecordNotFound('Cannot find model')

        ModelTypeMatcher.get_class(obj.model_type).drop_from_cloud(obj)
        return True