# import json
from datetime import datetime, timezone
import os; os.chdir('..')
import time
import sys; sys.path.append('.')
from pathlib import Path

from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize, dump

import py.models as models
import py.callbacks as callbacks
from py.config import Config as conf
from py.configuration import ExperimentConfiguration
import py.classes as classes
from py.logger import l
from py.models import SKModel
import py.mongo as mongo
from py.exceptions import NoTrainingData, MissingConfigValues, MalformedExperiment
from py.my_utils import make_search_space


def pick_one_experiment_and_run():
    # first check restart statuses
    last_step = None
    experiment = mongo.Experiment.objects(status='RESTART').first()
    if experiment:
        l.debug(f'At least one experiment in status RESTART is found. id:{experiment.id}')
        pt = mongo.Point.objects(experiment=experiment).order_by("-step").limit(-1).first()
        last_step = pt.step + 1
        l.debug(f'starting with step {last_step}')
    else:    
        experiment = mongo.Experiment.objects(status='OPEN').first()
        if not experiment:
            l.debug("There are no experiments in status OPEN or RESTART")
            return
        else:
            l.debug(f'At least one experiment in status OPEN is found. id:{experiment.id}')

    experiment.status = 'IN_PROCESS'
    experiment.executor = conf.EXECUTOR
    experiment.exp_id = str(int(time.time()))  
    experiment.started_at = datetime.now(timezone.utc)
    experiment.save()
    
    # Read search space
    search_space = make_search_space(experiment.space)
    # Init new experiment configuration instance
    ec = ExperimentConfiguration()
    initial_space = experiment.space
    ec.load_from_experiment_object(experiment)

    # Init new experiment pipeline instance
    ep = classes.ExperimentPipeline(experiment_configuration=ec)

    @use_named_args(search_space)
    def execute_experiment(**params):
        ec._load_from_dict(initial_space)
        ec.adjust_parameters(**params)
        ep.config = ec
        return ep.run(ModelClass=SKModel)
    
    mongo_saver = callbacks.MongoSaver(
        ec, 
        ep, 
        experiment, 
        step=last_step or 1,
        search_space=search_space)
    try:
        res_gp = gp_minimize(
        execute_experiment, 
        search_space, 
        callback=[mongo_saver],
        verbose=True,
        **experiment.optimization_config.values)
    except NoTrainingData:
        experiment.status = 'ERROR'
        experiment.status_message = ('Got NoTrainingData exception')
    except MissingConfigValues:
        experiment.status = 'ERROR'
        experiment.status_message = (f'Got MissingConfigValues: {e.args[0]}')
    except MalformedExperiment as e:
        l.error(e.args[0])
        experiment.status = 'ERROR'
        experiment.status_message = (f'Got MalformedExperiment: {e.args[0]}')
    else:    
        experiment.status = 'DONE'
    finally:
        experiment.finished_at = datetime.now(timezone.utc)
        experiment.save()

def main():
    try:
        db_client = mongo.connect_to_mongo()
        while not Path('stop').is_file():
            pick_one_experiment_and_run()
            time.sleep(10)
        os.remove('stop')
        l.info("Found stop file and has stopped processing queue")
    finally:
        db_client.close()


if __name__ == '__main__':
    main()