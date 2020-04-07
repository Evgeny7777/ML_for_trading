import json
from datetime import datetime, timezone
import os; os.chdir('../..')
import time
import sys; sys.path.append('.')

from scipy.stats import randint as sp_randint

import py.classes as classes
from py.config import Config as conf
from py.logger import l
import py.mongo as mongo
from py.exceptions import NoTrainingData, MissingConfigValues, MalformedExperiment



def fine_tune_experiment(exp_obj_id, 
    model_space,
    top_mean=0, 
    top_diff=0, 
    remove_prev_results=False, 
    minimal_deals_per_day=0.2, 
    search_iterations=80
    ):
    
    l.debug(f'Fine tuning experiment. Space is {model_space}')
    model_param_distributions = {}
    for k,v in model_space.items():
        if v['type'] == 'int':
            model_param_distributions[k] = sp_randint(v['bounds'][0], v['bounds'][1])
    
    if top_mean==0 and top_diff==0:
        l.warning('No finetuning is done. Provide top_profit or top_diff')
        return
    
    if remove_prev_results: 
        for p in mongo.Point.objects_safe(experiment=exp_obj_id, fine_tuned=True): p.delete()
        already_fine_tuned = []
    else:
        already_fine_tuned = [p.step for p in mongo.Point.objects_safe(experiment=exp_obj_id, fine_tuned=True).only('step')]

    top_mean_points = mongo.Point.objects_safe(
            experiment=exp_obj_id, 
            fine_tuned__in=[None, False], 
            test_deals_per_day__gt=minimal_deals_per_day,
            step__nin=already_fine_tuned
        ).order_by('-test_mean').limit(top_mean)

    top_diff_points = mongo.Point.objects_safe(
            experiment=exp_obj_id, 
            fine_tuned__in=[None, False], 
            test_deals_per_day__gt=minimal_deals_per_day,
            step__nin=already_fine_tuned
        ).order_by('-test_diff').limit(top_diff)

    ec = classes.ExperimentConfiguration()
    ec.load_from_experiment_step(exp_obj_id, 1, fine_tuned=False) # load initial config that we will update later on
    datasets = None
    processed_points = []
    for points_set in [top_mean_points, top_diff_points]:
        for p in points_set:
            if p.id in processed_points: continue
            processed_points.append(p.id)

            ec.adjust_parameters(**p.coordinates)
            
            # assume that there are same datasets used within entire experiment
            if datasets == None: 
                datasets = classes.SKModel.get_datasets_from_exp_config(ec)
            else: 
                for d in datasets: d.update(update_source=False)

            model = classes.SKModel(ec)
            model.train(
                datasets=datasets,
                tags='TRAIN', 
                model_param_distributions=model_param_distributions,
                random_search_iterations=search_iterations)

            result_test, result_list_test = model.evaluate(datasets, tags='TEST')
            result_val, result_list_val = model.evaluate(datasets, tags='VAL')

            point = classes.Point(
                        step = p.step, 
                        evaluation_on_test = result_test.mongo_object_with_deals,
                        evaluation_on_val = result_val.mongo_object,
                        detailed_evaluation_on_val = [r.mongo_object for r in result_list_val],
                        detailed_evaluation_on_test = [r.mongo_object for r in result_list_test],
                        coordinates = p.coordinates,
                        experiment = exp_obj_id,
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
                        clf_params=getattr(model.clf, 'best_params_', None),
                        fine_tuned=True
                        )
            point.save_safe()

def pick_one_experiment_and_fine_tune():
    experiment = mongo.Experiment.objects_safe(status='2TUNE').first()
    if experiment is None: 
        l.debug('No experiments in status 2TUNE are found')
        return
    else:
        l.debug(f'Processing experiment {experiment.id}, name {experiment.name} ')
    experiment.status = 'FINETUNING_IN_PROCESS'
    if not experiment.model_tuning_params:
        experiment.model_tuning_params = conf.DEFAULT_TUNING_SPACE
    experiment.save_safe()

    try:
        fine_tune_experiment(experiment.id, **experiment.model_tuning_params)
    except NoTrainingData:
        experiment.status = 'TUNING_ERROR'
        experiment.status_message = ('Got NoTrainingData exception')
    except MissingConfigValues:
        experiment.status = 'TUNING_ERROR'
        experiment.status_message = (f'Got MissingConfigValues: {e.args[0]}')
    except MalformedExperiment as e:
        l.error(e.args[0])
        experiment.status = 'TUNING_ERROR'
        experiment.status_message = (f'Got MalformedExperiment: {e.args[0]}')
    else:    
        experiment.status = 'DONE_AND_TUNED'
        l.debug('Experiment is successfully tuned')
    finally:
        experiment.save_safe()

def main():
    db_client = mongo.connect_to_mongo()
    try:
        while True:
            pick_one_experiment_and_fine_tune()
            time.sleep(10)
    finally:
        db_client.close()
    

if __name__ == '__main__':
    main()