from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os.path

from joblib import dump, load
from skopt import dump
import numpy as np

from py.logger import l
from py.mongo import Point, Experiment, Evaluation
from pymongo.errors import DocumentTooLarge


class BestResultCallback(ABC):
    
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, result):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        self.result = result
        curr_value = result.func_vals[-1] 
        best_value = np.min(result.func_vals)
        if curr_value == best_value:
            l.debug(f'Best result {best_value} is achieved')
            self.do_if_best_result(best_value)
    
    @abstractmethod
    def do_if_best_result(self, best_value):
        pass
            
class BestResultConfigSaver(BestResultCallback):
    """
    Parameters
    ----------
    * `experiment_configuration`: instance of ExperimentConfiguration
    """
    def __init__(self, experiment_configuration, file_path):
        self.experiment_configuration = experiment_configuration
        self.file_path = file_path

    def do_if_best_result(self):
        self.experiment_configuration.save_to_file(self.file_path)

class BestResultModelSaver(BestResultCallback):
    """
    Parameters
    ----------
    * `experiment_configuration`: instance of ExperimentConfiguration
    """
    def __init__(self, experiment_pipepline, file_path):
        self.experiment_pipepline = experiment_pipepline
        self.file_path = file_path

    def do_if_best_result(self):
        self.experiment_pipepline.dump_model(self.file_path)

class EarlyStopper(object):
    """Decide to continue or not given the results so far.
    The optimization procedure will be stopped if the callback returns True.
    """
    def __call__(self, result):
        return self._criterion(result)

    def _criterion(self, result):
        file_name = 'stop'
        if os.path.isfile(file_name):
            os.remove(file_name) # will remove a file.
            l.debug('Stop file is found. Stopping optimization task')
            return True
        return False

class MongoSaver(object):
    """Decide to continue or not given the results so far.
    The optimization procedure will be stopped if the callback returns True.
    """
    def __init__(self, experiment_configuration, experiment_pipeline, mongo_experiment, step=1, search_space=[]):
        self.experiment_configuration = experiment_configuration
        self.ep = experiment_pipeline
        self.mongo_experiment = mongo_experiment
        self.step = step
        self.space = search_space
            
    def __call__(self, result):
        curr_value = int(result.func_vals[-1])
        best_value = int(np.min(result.func_vals))
        
        result_test, result_list_test = self.ep.model.evaluate(
            self.ep.datasets, tags='TEST')
        result_val, result_list_val = self.ep.last_eval_result, self.ep.last_eval_result_list

        if curr_value == best_value:
            self.mongo_experiment.best_evaluation_on_test = result_test.mongo_object
            self.mongo_experiment.best_evaluation_on_val = result_val.mongo_object
            self.mongo_experiment.best_point = dict(
                zip([d.name for d in self.space], [float(x) for x in result.x]))

        point = Point(
                step = self.step, 
                evaluation_on_test = result_test.mongo_object_with_deals,
                evaluation_on_val = result_val.mongo_object,
                detailed_evaluation_on_val = [r.mongo_object for r in result_list_val],
                detailed_evaluation_on_test = [r.mongo_object for r in result_list_test],
                coordinates = dict(zip([d.name for d in self.space], [float(x) for x in result.x_iters[-1]])) ,
                experiment = self.mongo_experiment,
                test_days=result_test.days,
                test_mean=result_test.mean,
                test_std=result_test.std,
                test_deals_per_day=result_test.deals_per_day,
                test_diff=result_test.diff,
                test_min=result_test.min,
                test_max=result_test.max,
                test_total=result_test.total,
                val_days=result_val.days,
                val_mean=result_val.mean,
                val_std=result_val.std,
                val_deals_per_day=result_val.deals_per_day,
                val_diff=result_val.diff,
                val_min=result_val.min,
                val_max=result_val.max,
                val_total=result_val.total,
                clf_params=getattr(self.ep.model.clf, 'best_params_', None),
                fine_tuned=False
                )
        point.save()

        self.mongo_experiment.points.append(point)
        self.mongo_experiment.updated_at = datetime.now(timezone.utc)
        try: 
            self.mongo_experiment.save()
        except DocumentTooLarge: # if doc is too large then let's decrease it's size
            l.warning('Got document too large. Trying to decrease size of the doc')
            for p in self.mongo_experiment.points:
                p.detailed_evaluation_on_val = None
            self.mongo_experiment.save()
            
        self.step += 1

class CheckpointSaver(object):
    """
    Save current state after each iteration with `skopt.dump`.
    Example usage:
        import skopt

        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])

    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint will be saved to;
    * `dump_options`: options to pass on to `skopt.dump`, like `compress=9`
    """
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        dump(res, self.checkpoint_path, **self.dump_options)