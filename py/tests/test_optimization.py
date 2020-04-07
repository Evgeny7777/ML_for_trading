import json
from datetime import datetime, timezone
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + '/../..')
import sys; sys.path.append(os.getcwd())
import time

from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, dump

import py.callbacks as callbacks
import py.classes as classes
from py.models import SKModel
from py.logger import l
import py.mongo as mongo
from py.my_utils import make_search_space
from py.start_optimization import pick_one_experiment_and_run

import linecache
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()

def create_test_optimization_config():
    obj = mongo.OptimizationConfiguration(
        name = "optimization config for a short experiment",
        description = "Opt config used for debugging",
        values = dict(
            n_calls=2, 
            n_random_starts=1,
            acq_func='LCB', 
            kappa=20
        )
    )
    obj.save()
    return obj.id 

def create_test_configuration():
    
    conf = {
        "ALL_LABEL_FIELDS": ["buy_profit", "sell_profit", "time2sell", "time2buy"],
        "BASE_FIELDS": ["open", "high", "low", "close", "vol", "price"],
        "DEAL_TYPE": "buy",
        "DATASOURCE_SHIFT": 1,
        "EVAL_COMMISSION_PER_DEAL": 10,
        "EVAL_MAX_LENGTH": 20,
        "EVAL_MIN_PROFIT": 500,
        "EVAL_STOP_LOSS": 400,
        "EVAL_TAKE_PROFIT": 1000,
        "EVAL_TRAILING_STOP": 0,
        "LABEL_DEAL_NO_DEAL": 4,
        "LABEL_MAX_LENGTH": 20,
        "LABEL_MIN_PROFIT": 500,
        "LABEL_STOP_LOSS": 400,
        "LABEL_TAKE_PROFIT": 1000,
        "LABEL_TRAILING_STOP": 0,
        "MODEL_MAX_DEPTH": 18,
        "MODEL_N_ESTIMATORS": 150,
        "PREPROCESS_PERIOD_MINUTES": 30
        }

    conf_values_obj = mongo.ConfigurationValues(**conf)
    conf_obj = mongo.Configuration(
        name = "simple test config",
        description = "Configuration used for debugging",
        created_at = datetime.now(timezone.utc),
        values = conf_values_obj
    )
    conf_obj.save()
    return conf_obj.id

def create_test_experiment(opt_config_id, status='OPEN'):
    space_dict = {
        "DATASOURCE_SHIFT": 1,
        "DEAL_TYPE": "buy",
        "EVAL_COMMISSION_PER_DEAL": 10,
        "EVAL_MAX_LENGTH": 20,
        "EVAL_MIN_PROFIT": 500,
        "EVAL_STOP_LOSS": 400,
        "EVAL_TAKE_PROFIT": 1000,
        "EVAL_TRAILING_STOP": 0,
        "LABEL_DEAL_NO_DEAL": (1, 9),
        "LABEL_MAX_LENGTH": (3, 9),
        "LABEL_MIN_PROFIT": 500,
        "LABEL_TAKE_PROFIT": 1000,
        "LABEL_TRAILING_STOP": 0,
        "MODEL_MAX_DEPTH": 18,
        "MODEL_N_ESTIMATORS": (10, 200),
        "PREPROCESS_PERIOD_MINUTES": 5,
        'LABEL_EVAL_PARAM_SHARING': False
        }
    model_space_dict = {"max_depth": [3, 7], "n_estimators": [10, 30, 70, 100, 150, 200]}


    experiment = mongo.Experiment(
        name = 'debug_task',
        status = status,
        description = 'this is a test task',
        space = space_dict,
        model_space = model_space_dict,
        created_at = datetime.now(timezone.utc),
        optimization_config = mongo.OptimizationConfiguration.objects(id=opt_config_id).first(),
        code='TS',
        data = {
            "DEFAULT_SOURCE_TYPE": "s3",
            "DATASOURCES": {
            "TSH0":["TRAIN"],
            "TSH1":["VAL"],
            "TSH2":["TEST"]
            }
        }
    )

    experiment.save()
    return experiment.id

def train_test_model(experiment_object_id):
    model = SKModel()
    model.train_from_experiment_step(experiment_object_id, step_no=1)
    model_object_id = model.save_to_cloud()
    return model_object_id, model._id

def load_test_model_by_object_id(model_object_id):
    model = SKModel()
    model.load_from_cloud(model_object_id=model_object_id)
    l.debug('Loaded a model via model object id')
    return model

def load_test_model_by_id(model_id):
    model = SKModel()
    model.load_from_cloud(model_id=model_id)
    l.debug('Loaded a model via model_id')
    return model

def delete_test_model(model_object_id):
    model = SKModel()
    model.drop_from_cloud(model_object_id)

def main():
    db_client = mongo.connect_to_mongo(db_name='tests', drop_db=True, localhost=False)
    try:
        opt_conf_id = create_test_optimization_config()
        # conf_id = create_test_configuration()
        exp_id  = create_test_experiment(opt_conf_id)
        pick_one_experiment_and_run()
        model_object_id, model_id = train_test_model(exp_id)
        model = load_test_model_by_object_id(model_object_id)
        model = load_test_model_by_id(model_id)
        # assert False
        delete_test_model(model_object_id)
    finally:
        db_client.close()
    

if __name__ == '__main__':
    main()
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)