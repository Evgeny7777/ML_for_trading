import time
import os; os.chdir('..')
import pickle
import sys; sys.path.append('.')
from py.logger import l

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, dump

from py.config import Config as conf
from py.callbacks import BestResultConfigSaver, BestResultModelSaver, MongoSaver, CheckpointSaver
from py.classes import ExperimentConfiguration, ExperimentPipeline
from py.tradeutils import add_datetimeindex_and_save


DEFAULT_CONFIGURATION_FILE = 'no_tp.sell.json'
EXPERIMENT_DESCRIPTION = 'sell, new gen, not tuned space, no take profit, SL in space'
DEBUG_MODE = False

experiment_id = str(int(time.time()))   
l.info(f"Kick off of new experiment, id is {experiment_id}")
result_folder = conf.RESULTS_FOLDER + experiment_id + '/'

l.debug(f'Creating folder {result_folder} to store results')
os.makedirs(result_folder)

space  = [
    Integer(3, 9, name='LABEL_MAX_LENGTH'),
    Integer(1, 9, name='LABEL_DEAL_NO_DEAL'),
    Integer(100, 500, name='LABEL_TRAILING_STOP'),
    Integer(30, 70, name='LABEL_MIN_PROFIT'),
    Integer(10, 200, name='MODEL_N_ESTIMATORS'),
    Integer(2, 30, name='MODEL_MAX_DEPTH'),
    Integer(100, 500  , name='EVAL_TRAILING_STOP'),
    Integer(200, 1000, name='LABEL_STOP_LOSS'),
    Integer(200, 1000  , name='EVAL_STOP_LOSS')
]
l.debug(f'Search space is {space}')

ec = ExperimentConfiguration(
  experiment_id, 
  space=space)
ec.load_setup(conf.CONFIGURATIONS_FOLDER+DEFAULT_CONFIGURATION_FILE)

l.debug(f' default configuration is {ec.describe}')

ep = ExperimentPipeline(experiment_configuration=ec)

@use_named_args(space)
def execute_experiment(**params):
    ec.load_setup(conf.CONFIGURATIONS_FOLDER+DEFAULT_CONFIGURATION_FILE)
    ec.adjust_parameters(**params)
    ep.config = ec
    return ep.run()

best_result_saver = BestResultConfigSaver(ec, result_folder + conf.BEST_CONFIG_FILE_NAME)
best_model_saver = BestResultModelSaver(ep, result_folder + conf.BEST_MODEL_FILE_NAME)
mongo_saver = MongoSaver(ec, ep, EXPERIMENT_DESCRIPTION)
checkpoint_saver = CheckpointSaver(f'{result_folder}result.gz', compress=9)
try:
    res_gp = None
    res_gp = gp_minimize(
      execute_experiment, 
      space, 
      n_calls=1 if DEBUG_MODE else 800, 
      n_random_starts=1 if DEBUG_MODE else 400,
    #     random_state=0, 
      acq_func='LCB', 
      kappa=20,
      callback=[mongo_saver],
      verbose=True)
except KeyboardInterrupt:
    l.info('Manually interrupted')

l.info(f"Experiment {experiment_id} is over")