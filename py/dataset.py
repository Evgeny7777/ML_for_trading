from datetime import datetime, timezone
import json
from functools import wraps
import os
import pprint
import pickle
import time
from typing import List

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

from py.configuration import ExperimentConfiguration
from py.config import Config as conf
from py.exceptions import NoTrainingData, MissingConfigValues
from py.logger import l
from py.mongo import Evaluation, Experiment, ModelCollection, Configuration, Point, connect_to_mongo
from py.my_pd_utils import concat_dfs, exclude_fields, make_list_if_single
from py.s3 import S3_client


class DataSource:
    name = 'no_name'
    def __init__(self):
        pass
    
    def load(self):
        raise NotImplementedError
    
    def update(self):
        pass

    @property
    def df(self):
        raise NotImplementedError


class FileDataSource(DataSource):
    def __init__(self, filepath, code, **kvargs):
        self._filepath = filepath
        self.code = code
    
    def load(self):
        l.debug(f'reading {self._filepath}')
        date_format = '%Y-%m-%d %H:%M:%S'
        dft = pd.read_csv(self._filepath, index_col='full_date', parse_dates=True)
        dft.index = pd.to_datetime(dft.index, format=date_format)
        self._df = dft
        return self
    
    @property
    def df(self):
        return self._df 


class S3DataSource(DataSource):
    def __init__(self, code, **kvargs):
        self.code = code
        self.s3 = S3_client()
    
    def load(self):
        l.debug(f'Loading S3 datasource for code {self.code}')
        filepath = self.s3.download_if_not_cached_and_get_path(
            key=f"data/{self.code}.csv.gz")
        date_format = '%Y-%m-%d %H:%M:%S'
        dft = pd.read_csv(filepath, index_col='full_date', parse_dates=True)
        dft.index = pd.to_datetime(dft.index, format=date_format)
        self._df = dft
        return self
    
    @property
    def df(self):
        return self._df 


class DBDataSource(DataSource):
    def __init__(self, code, last_n_hours=48):
        self.last_n_hours = last_n_hours
        self.engine = create_engine(conf.MYSQL_CONN_STRING, pool_recycle=1)
        self.code = code

        self._load_sql_query = (
            f'SELECT `tm`, `price`, `vol`, `id` FROM ticks WHERE `code`="{code}"'
            f'AND `tm` > (CURDATE() + INTERVAL -({last_n_hours}) HOUR) ORDER BY tm DESC'
        )

        self._update_sql_query = (
            'SELECT `tm`, `price`, `vol`, `id` FROM ticks WHERE `code`="{code}" AND '
            '`id` > "{last_id}" ORDER BY tm DESC'
        )

    def load(self, last_n_days=2):
        dft = pd.read_sql_query(
            sql=self._load_sql_query, 
            con=self.engine,
            index_col='tm')
        self._last_id = dft['id'].max()
        self._df = dft.drop(columns='id').applymap(int)
        return self

    def update(self):
        dft = pd.read_sql_query(
            sql=self._update_sql_query.format(code=self.code, last_id=self._last_id), 
            con=self.engine,
            index_col='tm')
        self._last_id = dft['id'].max()

        self._df = pd.concat([dft.drop(columns='id'), self._df]).applymap(int)

    @property
    def df(self):
        return self._df.sort_index()


class Dataset:
    SUPPORTED_SOURCE_TYPES = ('f', 's3', 'db')

    def __init__(
        self, 
        experiment_configuration:ExperimentConfiguration, 
        tags=[], 
        last_n_steps=-1,
        add_label=False,
        turn_on_rebalancing=False,
        data_source:DataSource=None,
        source_string:str=None,
        shift:int=0 #Don't use while inference
        ):
        """
            last_n_steps - int. If -1 then no restriction
        """
        if data_source is None:
            self.source = self._get_datasource(
                source_string,
                default_source_type = getattr(experiment_configuration, 'DEFAULT_SOURCE_TYPE', None)
                )
            self.source.load()
        else:
            self.source = data_source
            self.source.update()

        self.config = experiment_configuration
        self.last_config_revision = self.config.last_revision

        self.tags = make_list_if_single(tags)
        self.last_n_steps = last_n_steps
        self.add_label = add_label
        self.turn_on_rebalancing = turn_on_rebalancing
        self.shift = shift

        self._initial_candles_data = None

        self.preprocess()
        self.update()
    
    def preprocess(self, cache=True):
        if self._initial_candles_data is None or not cache: 
            self._initial_candles_data = self._preprocess().sort_index()

    def update(self, update_source=True):
        if update_source: 
            self.source.update()
       
        if self._initial_candles_data is None:
            raise ValueError('Before calling update() you would need to call preprocess()')

        self._data = self._initial_candles_data

        if self.add_label:
            self._data = self._assign_labels().sort_index()
            if self.turn_on_rebalancing:
                self._data = self._rebalance().sort_index()
        
        self.last_config_revision = self.config.last_revision
        return self

    @property
    def data(self):
        if self.last_config_revision == self.config.last_revision: return self._data.sort_index()

        if self.config.starting_stage <= PipelineStages.LABEL:
            self._data = self._initial_candles_data
            if self.add_label:
                self._data = self._assign_labels().sort_index()
                if self.turn_on_rebalancing:
                    self._data = self._rebalance().sort_index()

        self.last_config_revision = self.config.last_revision
        return self._data.sort_index()
    
    @property
    def X(self):
        return self.data.pipe(exclude_fields, conf.BASE_FIELDS+conf.ALL_LABEL_FIELDS).values
    
    @property
    def Y(self):
        if not self.add_label:
            raise ValueError('Labels are not calculated. See "add_label" param')
        return self.data[self.config.MAIN_LABEL].values

    @property
    def X_latest_row(self):
        return self.data.pipe(exclude_fields, conf.BASE_FIELDS+conf.ALL_LABEL_FIELDS).iloc[-1].values.reshape(1, -1)

    @property
    def latest_time(self):
        return self.data.index[-1]

    #----------------------
    def _preprocess(self):
        local_conf = self.config
        l.debug(f'Start preprocessing of {self.source.code}')
        # l.debug(f'Used memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
        dft = self.source.df.copy()
        # l.debug(f'Used memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
        
        # this shift is done to be able to do prediction every minute. Needed for inference phase
        timeshift = -1*dft.index.max().minute % local_conf.PREPROCESS_PERIOD_MINUTES

        # this shift is done to catch all the configurations. Don't use while inference
        dft.index = dft.index + pd.Timedelta(minutes=self.shift) + pd.Timedelta(minutes=timeshift)
        # l.debug(f'Used memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

        df_temp = dft.pipe(self.to_candles, f'{local_conf.PREPROCESS_PERIOD_MINUTES}T')
        # l.debug(f'Used memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
        
        if self.last_n_steps > 0: df_temp = df_temp[-self.last_n_steps:]
        # l.debug(f'Used memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

        df_temp = (df_temp
                .assign(weekday = df_temp.index.weekday, 
                        month=df_temp.index.month, 
                        hour=df_temp.index.hour)
                .pipe(self.add_ta_features)
                .pipe(self.add_days_to_expiration, self.source.code)
        )
        df_temp.index = df_temp.index - pd.Timedelta(minutes=timeshift)
        return df_temp
    
    def _assign_labels(self):
        """
            Stage #2: assign labels
        """
        conf = self.config
        return self._data.pipe(
            self.add_labels, 
            conf.LABEL_TRAILING_STOP, 
            conf.LABEL_MAX_LENGTH, 
            conf.LABEL_MIN_PROFIT,
            deal_type = conf.DEAL_TYPE)

    def _rebalance(self):
        """
            Stage #2.1: rebalance
        """
        conf = self.config
        
        old_size = self._data.shape[0]
        df_pos = self._data[self._data[conf.MAIN_LABEL].astype(bool)]
        df_neg = self._data[~self._data[conf.MAIN_LABEL].astype(bool)]
        if df_neg.shape[0] == 0:
            l.debug(f'{self.source.code}: {old_size}, No negative samples. No rebalancing will be done')
            return self._data
        if df_pos.shape[0] == 0:
            l.debug(f'{self.source.code}: {old_size}, No negative samples. No rebalancing will be done')
            return self._data
        if df_neg.shape[0] <= conf.LABEL_DEAL_NO_DEAL * df_pos.shape[0]:
            l.debug(f'{self.source.code}: {old_size}, n={df_neg.shape[0]}, p={df_pos.shape[0]}. No rebalancing is necessary')
            return self._data

        number_of_negative_samples = conf.LABEL_DEAL_NO_DEAL * df_pos.shape[0]
        df_neg = df_neg.sample(n=number_of_negative_samples)
        df = pd.concat([df_pos, df_neg])
        l.debug(f'{self.source.code}: {old_size}-->{df.shape[0]}, n={df_neg.shape[0]}, p={df_pos.shape[0]}')
        return df
    #----------------------
    
    @classmethod
    def _get_datasource(cls, source_string, default_source_type=None):
        
        if len(source_string.split(':')) == 2:
            source_type = source_string.split(':')[0].lower()
            code = source_string.split(':')[1]
        elif len(source_string.split(':')) == 1:
            if not default_source_type:
                msg = f'source type is not provided. source_string = {source_string}'
                l.error(msg)
                raise ValueError(msg)
            source_type = default_source_type.lower()
            code = source_string
        else:
            msg = f'got wrong source_string={source_string}'
            l.error(msg)
            raise ValueError(msg)
        
        if source_type not in cls.SUPPORTED_SOURCE_TYPES:
            msg = f'got wrong source type. source_string={source_string}, source_type={source_type}'
            l.error(msg)
            raise ValueError(msg)

        if source_type == 'f':
            return FileDataSource(conf.DATA_FILE_TEMPLATE.format(f'L0.{code}.csv.gz'), code=code)
        if source_type == 'db':
            return DBDataSource(code=code)
        if source_type == 's3':
            return S3DataSource(code)
    #----------------------    
    @staticmethod
    def to_candles(df, period):
        l.debug(f'period={period}')
        ds_vol = df.vol.resample(period).sum().dropna().astype(int)
        ds_deals = df.vol.resample(period).count().dropna().astype(int)
        ds_price = (df.vol * df.price).resample(period).sum().dropna() / ds_vol
        ds_price = ds_price.fillna(method='backfill').astype(int)
        
        df_candles = (df.price
                .resample(period)
                .ohlc()
                .dropna()
                .astype(int)
                .assign(vol = ds_vol)
                .assign(deals = ds_deals)
                .assign(avg_deal = ds_vol/ds_deals)
                .assign(price = ds_price)
            )
        df_candles = (df_candles
                        .assign(candle_body = df_candles.apply(lambda r: r.close - r.open, axis=1))
                        .assign(upper_shadow = df_candles.apply(lambda r: r.high - max(r.close, r.open), axis=1))
                        .assign(lower_shadow = df_candles.apply(lambda r: min(r.close, r.open) - r.low, axis=1))
                    )

        for candle_size in [3, 7]:
            df_candles[f'candle_body_{candle_size}'] = df_candles.close - df_candles.open.shift(candle_size)
            
        return df_candles

    @staticmethod
    def add_ta_features(df):
        pr = df.price
        HL = (df.high, df.low)
        HLC = (df.high, df.low, df.close)
        OHLC = (df.open, df.high, df.low, df.close)
        HLCV = (df.high, df.low, df.close, df.vol)
        PV = (pr, df.vol)
        initial_col_count = df.columns.shape[0]
        df = ( df
                .assign(SMA4 = (ta.SMA(pr, timeperiod=4) - pr))
                .assign(SMA12 = (ta.SMA(pr, timeperiod=12) - pr))
                .assign(SMA24 = (ta.SMA(pr, timeperiod=24) - pr))
                .assign(BBAND20_HI=(ta.BBANDS(pr,timeperiod = 20)[0] - pr))
                .assign(BBAND20_LO=(pr - ta.BBANDS(pr, timeperiod = 20)[2]))
                .assign(SAREXT=(ta.SAREXT(*HL) - pr))
                .assign(ADX=ta.ADX(*HLC))
                .assign(ADOSC=ta.ADOSC(*HLCV))
                .assign(CDL3STARSINSOUTH=(ta.CDL3STARSINSOUTH(*OHLC)))
                .assign(HT_TRENDLINE = ta.HT_TRENDLINE(pr)-pr)
                .assign(KAMA = ta.KAMA(pr)-pr)
                .assign(MIDPOINT = ta.MIDPOINT(pr)-pr)
                .assign(T3 = ta.T3(pr)-pr)
                .assign(TEMA = ta.TEMA(pr)-pr)
                .assign(TRIMA = ta.TRIMA(pr)-pr)
                .assign(ADXR = ta.ADXR(*HLC))      
                .assign(APO = ta.APO(pr))      
                .assign(AROON1 = ta.AROON(*HL)[0])      
                .assign(AROON2 = ta.AROON(*HL)[1])            
                .assign(AROONOSC = ta.AROONOSC(*HL))      
                .assign(BOP = ta.BOP(*OHLC))
                .assign(CMO = ta.CMO(pr))
                .assign(DX = ta.DX(*HLC))
                .assign(MACD1 = ta.MACD(pr)[0])
                .assign(MACD2 = ta.MACD(pr)[1])
                .assign(MACDEXT1 = ta.MACDEXT(pr)[0])
                .assign(MACDEXT2 = ta.MACDEXT(pr)[1])   
                .assign(MFI = ta.MFI(*HLCV))
                .assign(MINUS_DI = ta.MINUS_DI(*HLC))
                .assign(MINUS_DM = ta.MINUS_DM(*HL))
                .assign(CMOMCI = ta.MOM(pr))
                .assign(PLUS_DI = ta.PLUS_DI(*HLC))
                .assign(PLUS_DM = ta.PLUS_DM(*HL)) 
                .assign(PPO = ta.PPO(pr))
                .assign(ROC = ta.ROC(pr))
                .assign(ROCP = ta.ROCP(pr))
                .assign(ROCR = ta.ROCR(pr))
                .assign(RSI = ta.RSI(pr))
                .assign(STOCH1 = ta.STOCH(*HLC)[0])
                .assign(STOCH2 = ta.STOCH(*HLC)[1])
                .assign(STOCHF1 = ta.STOCHF(*HLC)[0])
                .assign(STOCHF2 = ta.STOCHF(*HLC)[1])
                .assign(STOCHRSI1 = ta.STOCHRSI(pr)[0])
                .assign(STOCHRSI2 = ta.STOCHRSI(pr)[1])
                .assign(ULTOSC = ta.ULTOSC(*HLC))
                .assign(WILLR=ta.WILLR(*HLC))
                .assign(AD = ta.AD(*HLCV))
                .assign(ADOSC = ta.ADOSC(*HLCV))
                .assign(OBV = ta.OBV(*PV))
                .assign(HT_DCPERIOD = ta.HT_DCPERIOD(pr))
                .assign(HT_DCPHASE = ta.HT_DCPHASE(pr))
                .assign(HT_PHASOR1 = ta.HT_PHASOR(pr)[0])
                .assign(HT_PHASOR2 = ta.HT_PHASOR(pr)[1])
                .assign(HT_SINE1 = ta.HT_SINE(pr)[0])
                .assign(HT_SINE2 = ta.HT_SINE(pr)[1])
                .assign(HT_TRENDMODE = ta.HT_TRENDMODE(pr))
                .assign(ATR14 = ta.ATR(*HLC, timeperiod=14))
                .assign(NATR14 = ta.NATR(*HLC, timeperiod=14))
                .assign(TRANGE = ta.TRANGE(*HLC))  
                .assign(ATR7 = ta.ATR(*HLC, timeperiod=7))
                .assign(NATR7 = ta.NATR(*HLC, timeperiod=7))
                .assign(ATR21 = ta.ATR(*HLC, timeperiod=21))
                .assign(NATR21 = ta.NATR(*HLC, timeperiod=21))
                # .astype(int, errors='ignore')          
            )
        # adding difference
        for col in df.columns[initial_col_count:]:
            df[col+'_change'] = df[col] - df[col].shift(1)
        candle_methods = [method_name for method_name in dir(ta) if method_name.startswith('CDL')]
        for method in candle_methods:
            df[method] = getattr(ta, method)(*OHLC)

        # n_before = df.shape[0]
        df = df.dropna()
        # l.debug(f'TA indicators do require {n_before - df.shape[0]} steps of history')
        return df

    @staticmethod
    def add_days_to_expiration(df, code):
        if df.shape[0] == 0: return df
        if len(code)!=4 or code[-2] not in ['H', 'M', 'U', 'Z']:
            raise ValueError(f"add_days_to_expiration(): got wrong value of code {code}")
        if code[-2] == 'H': last_dt = datetime(df.index.max().year, 3, 20)
        if code[-2] == 'M': last_dt = datetime(df.index.max().year, 6, 20)
        if code[-2] == 'U': last_dt = datetime(df.index.max().year, 9, 20)
        if code[-2] == 'Z': last_dt = datetime(df.index.max().year, 12, 20)

        df['days_to_expiration']  = last_dt - df.index
        df.days_to_expiration = df.days_to_expiration.dt.days

        return df
    #----------------------
    @staticmethod    
    def expected_buy_profit(ds, trailing_stop=0, stop_loss=0, take_profit=0):
        _open = _max = ds[0]

        for _curr in ds:
            if trailing_stop:
                _max = max(_max, _curr)
                drop = _max - _curr 
                if drop > trailing_stop: break
            if stop_loss and (_open - _curr) > stop_loss: break
            if take_profit and (_curr - _open) > take_profit: break

        profit = _curr - _open
        return profit
    
    @staticmethod
    def expected_buy_duration(ds, trailing_stop=0, stop_loss=0, take_profit=0):
        _open = _max = ds[0]
        _start = ds.index[0]
        for idx, _curr in ds.iteritems():
            if trailing_stop:
                _max = max(_max, _curr)
                drop = _max - _curr 
                if drop > trailing_stop: break
            if stop_loss and (_open - _curr) > stop_loss: break
            if take_profit and (_curr - _open) > take_profit: break
        return (idx-_start).seconds

    @staticmethod
    def expected_sell_duration(ds, trailing_stop=0, stop_loss=0, take_profit=0):
        _open = _min = ds[0]
        _start = ds.index[0]
        for idx, _curr in ds.iteritems():
            _min = max(_min, _curr)
            drop = _curr - _min
            if drop > trailing_stop: break
            if stop_loss and (_curr - _open) > stop_loss: break
            if take_profit and (_open  -_curr) > take_profit: break

        return (idx-_start).seconds

    @staticmethod
    def expected_sell_profit(ds, trailing_stop=0, stop_loss=0, take_profit=0):
        _open = _min = ds[0]

        for _curr in ds:
            _min = max(_min, _curr)
            drop = _curr - _min
            if drop > trailing_stop: break
            if stop_loss and (_curr - _open) > stop_loss: break
            if take_profit and (_open  -_curr) > take_profit: break

        profit = _open - _curr
        return profit

    @classmethod
    def add_labels(
            cls, 
            df, 
            trailing_stop, 
            max_length, 
            min_profit, 
            deal_type, 
            stop_loss=0, 
            take_profit=0):
        dft = df.copy()

        if deal_type == 'buy':
            profit_function = cls.expected_buy_profit
        if deal_type == 'sell':
            profit_function = cls.expected_sell_profit

        dft[f'{deal_type}_profit'] = (
            dft
            .price
            .rolling(int(max_length))
            .agg(profit_function, trailing_stop=trailing_stop, 
                    stop_loss=stop_loss, take_profit=take_profit)
            .shift(periods=-int(max_length)+1)
            .fillna(0)
            .astype(int)
            )
        
        dft[f'time2{deal_type}'] = (dft[f'{deal_type}_profit'] >= min_profit)

        return dft

def get_datasets_from_exp_config(ec: ExperimentConfiguration) -> List[Dataset]:
    if not hasattr(ec, 'DATASOURCES'):
        raise ValueError('Experiment configuration does not have DATASOURCES')

    datasource_shift = getattr(ec, 'DATASOURCE_SHIFT', 0)
    if datasource_shift > 0:
        # make preload data sources
        datasources_dict = {
            source_string:Dataset._get_datasource(
                source_string=source_string,
                default_source_type=getattr(ec, 'DEFAULT_SOURCE_TYPE', None)
                ).load()
            for  source_string in ec.DATASOURCES
        }
        
        l.debug(f'Got DATASOURCE_SHIFT equal to {datasource_shift}')
        datasets = []
        for shift in range(0, ec.PREPROCESS_PERIOD_MINUTES, datasource_shift):
            l.debug(f'Processing with shift={shift}')
            datasets = datasets + [Dataset(
                            data_source=datasources_dict[source_string],
                            experiment_configuration=ec, 
                            add_label=True, 
                            turn_on_rebalancing='TRAIN' in tags, # do rebalancing for train only
                            tags=tags,
                            shift=shift) 
                        for source_string, tags in ec.DATASOURCES.items()]
    else:
        datasets = [Dataset(
                            source_string=source_string, 
                            experiment_configuration=ec, 
                            add_label=True, 
                            turn_on_rebalancing='TRAIN' in tags, # do rebalancing for train only
                            tags=tags) 
                        for source_string, tags in ec.DATASOURCES.items()]
    l.debug(f'{len(ec.DATASOURCES)} sources --> {len(datasets)} datasets')
    return datasets

def filter_datasets_with_tags(datasets: List[Dataset], tags: List[str]) -> List[Dataset]:
    if tags=='ALL': return list(datasets)
    
    res = []
    for dataset in datasets:
        for tag in make_list_if_single(tags):
            if tag in dataset.tags: 
                res.append(dataset)
                break
    return res
