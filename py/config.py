import os
class Config():
    RESULTS_FOLDER = './results/'
    LOG_FOLDER = './logs/'
    DATA_FOLDER = './data/'
    MODEL_FOLDER = './models/'
    DATA_FILE_TEMPLATE = './data/{}'
    CONFIGURATIONS_FOLDER = '../configurations/'
    BEST_CONFIG_FILE_NAME = 'best_config.pkl'
    BEST_MODEL_FILE_NAME = 'model.joblib'
    
    MONGO_DB_NAME = 'evo_v2'
    _MONGO_CONN_STRING_TEMPLATE = 'mongodb://XXX:XXX@mongodb-2482-0.cloudclusters.net:10009/{db_name}?authSource=admin'
    _LOCAL_MONGO_CONN_STRING_TEMPLATE = 'mongodb://localhost:27017/{db_name}?authSource=admin'
    
    MYSQL_CONN_STRING = 'mysql+pymysql://XXX:XXX@trading.cmevsoc4w2mt.us-east-2.rds.amazonaws.com/trading'

    AWS_ACCESS_KEY_ID = 'XXX'
    AWS_SECRET_ACCESS_KEY ='XXX'
    AWS_BUCKET_NAME = 'estermanight'
    AWS_LOCAL_CACHE_FOLDER = './s3/'
    AWS_MODELS_FOLDER = 'models/'

    ALL_LABEL_FIELDS = ["buy_profit", "sell_profit", "time2sell", "time2buy"]
    BASE_FIELDS = ["open", "high", "low", "close", "vol", "price"]
    REQUIRED_CONFIG_VALUES = [
        'PREPROCESS_PERIOD_MINUTES', 
        'DEAL_TYPE',
        'EVAL_COMMISSION_PER_DEAL',
        'LABEL_MAX_LENGTH',
        'LABEL_MIN_PROFIT',
        # 'MODEL_MAX_DEPTH',
        # 'MODEL_N_ESTIMATORS',
        'DATASOURCES',
        'LABEL_EVAL_PARAM_SHARING'
        ]

    DEFAULT_CONFIG_VALUES = { 
        'DATASOURCE_SHIFT': 0, 
        'LABEL_DEAL_NO_DEAL': 1,
        "LABEL_TAKE_PROFIT" : 0,
        "LABEL_TRAILING_STOP" : 0,
        "LABEL_STOP_LOSS" : 10000,
        'LABEL_EVAL_PARAM_SHARING': True # ToDo: remove after a while 
    }

    DEFAULT_TUNING_SPACE = dict(
            top_mean=10,
            top_diff=10,
            remove_prev_results=True, 
            minimal_deals_per_day=0.2, 
            search_iterations=30,
            model_space={
                "max_depth": {'type':'int', 'bounds':(1,20)}, 
                "n_estimators": {'type':'int', 'bounds':(5,200)}
            }
        )
    SK_JOBS = int(os.getenv('SK_JOBS', -1))
    EXECUTOR = os.getenv('EXECUTOR', os.uname().nodename)


class PipelineStages():
    PREPROCESSING = 1
    LABEL = 2
    MODEL = 3
    EVAL = 4