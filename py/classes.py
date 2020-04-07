from datetime import datetime, timezone
import json
from functools import wraps
import os
import pprint
import pickle
import time

from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
import talib as ta
from enum import Enum
import pymysql
from sqlalchemy import create_engine

from py.config import Config as conf
from py.config import PipelineStages
from py.dataset import get_datasets_from_exp_config
from py.exceptions import NoTrainingData, MissingConfigValues
from py.logger import l
from py.mongo import Evaluation, Experiment, ModelCollection, Configuration, Point, connect_to_mongo
from py.my_pd_utils import concat_dfs, exclude_fields, make_list_if_single
from py.s3 import S3_client


class Evaluation_results:
    def __init__(self, ds_profit=None, code=None, days=0):
        self.ds = ds_profit
        self.code = code
        self.days = days
    
    @property
    def count(self): return 0 if self.ds is None else self.ds.shape[0]
    
    @property
    def mean(self): return 0 if self.ds is None else self.ds.mean()

    @property
    def std(self): return 0 if self.ds is None else self.ds.std()
    
    @property
    def min(self): return 0 if self.ds is None else self.ds.min()
    
    @property
    def max(self): return 0 if self.ds is None else self.ds.max()

    @property
    def total(self): return 0 if self.ds is None else self.ds.sum()
    
    @property
    def optimization_target(self): return 0 if self.ds is None else -self.ds.sum()

    @property
    def median(self): return 0 if self.ds is None else self.ds.median()
    
    @property
    def std(self): return 0 if self.ds is None else self.ds.std()
    
    @property
    def diff(self): return 0 if self.ds is None else self.mean - self.std

    @property
    def deals_per_day(self): return self.count/self.days if self.days > 0 else 0
    
    def _make_mongo_object(self, add_all_deals):
        ev = Evaluation(
            profit = float(self.total),
            mean = float(self.mean),
            median = float(self.median),
            best_deal = float(self.max),
            worst_deal = float(self.min),
            std = float(self.std),
            deals_count = int(self.count),
            days = int(self.days),
            deals_per_day = float(self.deals_per_day)
        )

        if add_all_deals:
                ev.all_deals = [] if self.ds is None else [{'date':str(i[0]), 'profit':i[1]} for i in self.ds.iteritems()]
        
        if self.code: ev.code = self.code
        return ev

    @property
    def mongo_object(self):
        return self._make_mongo_object(add_all_deals=False)

    @property
    def mongo_object_with_deals(self):
        return self._make_mongo_object(add_all_deals=True)

    def __add__(self, other):
        if self.ds is not None or other.ds is not None: # don't concat 2 Nones
            self.ds = pd.concat([self.ds, other.ds])
            self.days = self.days + other.days
        return self
    
    def __radd__(self, other):
        if self.ds is None or other.ds is None: # don't concat 2 Nones
            return Evaluation_results(None, f'{self.ds}+{other.ds}', days = self.days + other.days)    
        return Evaluation_results(pd.concat([self.ds, other.ds]), f'{self.ds}+{other.ds}', days = self.days + other.days)


class ExperimentPipeline:
    
    def __init__(self, experiment_configuration):
        self.config = experiment_configuration
        self._is_first_execution = True
        
    def run(self, ModelClass=None):
        if self._is_first_execution:
            self.datasets = get_datasets_from_exp_config(self.config)
            self._is_first_execution = False
        
        if self.config.starting_stage <= PipelineStages.PREPROCESSING: 
            for dataset in self.datasets:
                dataset.preprocess()

        if self.config.starting_stage <= PipelineStages.MODEL: 
            for dataset in self.datasets:
                dataset.update(update_source=False)
            self.model = ModelClass(self.config)
            
            train_params = dict(
                datasets=self.datasets, 
                tags='TRAIN', 
                model_param_distributions=self.config.MODEL_SPACE
            )
            self.model.train(**train_params)
        
        if self.config.starting_stage <= PipelineStages.EVAL: 
            result, result_list = self.model.evaluate(self.datasets, tags='VAL')
            self.last_eval_result = result
            self.last_eval_result_list = result_list
            return result.optimization_target
