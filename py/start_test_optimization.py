# import json
from datetime import datetime, timezone
import os; os.chdir('..')
import time
import sys; sys.path.append('.')
from pathlib import Path

from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize, dump

import py.callbacks as callbacks
import py.classes as classes
from py.logger import l
import py.mongo as mongo
from py.my_utils import make_search_space

def pick_one_experiment_and_run():
    experiment = mongo.Experiment.objects(pk='5cdd609fd71f100c646bde55').first() #GD
    # experiment = mongo.Experiment.objects(pk='5cdba572d71f102dbd7b03c4').first() #RI
    if not experiment:
        l.debug("There are no experiments found")
        return
    experiment.status = 'IN_PROCESS'
    experiment.executor = os.uname().nodename
    experiment.exp_id = str(int(time.time()))  
    experiment.started_at = datetime.now(timezone.utc)
    experiment.save()
    
    # Read search space
    
    search_space = make_search_space(experiment.space)
    # Init new experiment configuration instance
    ec = classes.ExperimentConfiguration()
    initial_space = experiment.space
    ec.load_from_experiment_object(experiment)

    # Init new experiment pipeline instance
    ep = classes.ExperimentPipeline(experiment_configuration=ec)

    @use_named_args(search_space)
    def execute_experiment(**params):
        ec._load_from_space_dict(initial_space)
        ec.adjust_parameters(**params)
        ep.config = ec
        return ep.run()
    
    mongo_saver = callbacks.MongoSaver(ec, ep, experiment, search_space=search_space)

    res_gp = gp_minimize(
      execute_experiment, 
      search_space, 
      callback=[mongo_saver],
      verbose=True,
      **experiment.optimization_config.values)

    experiment.status = 'DONE'
    experiment.finished_at = datetime.now(timezone.utc)
    experiment.save()

def main():
    try:
        db_client = mongo.connect_to_mongo(db_name='evo')
        while not Path('stop').is_file():
            pick_one_experiment_and_run()
            time.sleep(10)
        os.remove('stop')
        l.info("Found stop file and has stopped processing queue")
    finally:
        db_client.close()

if __name__ == '__main__':
    main()