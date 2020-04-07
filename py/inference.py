from datetime import datetime, timezone
import json

from py.logger import l
from py.configuration import ExperimentConfiguration
from py.mongo import Evaluation, Experiment, ModelCollection, Configuration, Point, Predictions, connect_to_mongo
from py.dataset import Dataset, DBDataSource
from py.models import ModelInterface

class InferencePack:
    def __init__(self, code):
        self.loaded_models = []
        self.code = code
        l.debug(f'{code}: Preparing inference pack')
        if ModelCollection.objects(status='DEPLOYED', code=self.code[:2]).count() > 0:
            # load data source
            l.debug(f'{code}: load data source')
            self.source = DBDataSource(code).load()
            # load models
            self.loaded_models = []
            for obj in ModelCollection.objects(status='DEPLOYED', code=self.code[:2]):
                model_instance = ModelInterface.load_from_mongo_object(obj)
                model_instance.attach_data(self.dataset_factory)
                self.loaded_models.append(model_instance)
            l.debug(f'{code}: found {len(self.loaded_models)} models in status DEPLOYED')
        else:
            l.debug(f'{code}: no models found in status DEPLOYED')

    def do_predictions(self):
        l.debug(f'{self.code}: Doing predictions')
        self.source.update()
        for model in self.loaded_models:
            model.update_attached_data(update_source=False)
            model.trade_signal = model.predict_latest()[0]
            if model.trade_signal:
                l.debug(f'{self.code}: got deal recomendation for the model {model._id}/{model.config.DEAL_TYPE}')
    
    @property
    def json_response(self):
        advices = []
        for model in self.loaded_models:
            if model.trade_signal:
                one_model_response = dict(
                    recommendation=1,
                    deal_type = model.exp_config.DEAL_TYPE[0].upper(),
                    model_id = model.obj.model_id,
                    stop_loss = model.exp_config.EVAL_STOP_LOSS,
                    trailing_stop = model.exp_config.LABEL_TRAILING_STOP,
                    take_profit = model.exp_config.EVAL_TAKE_PROFIT,
                    maximum_lifetime = model.exp_config.LABEL_MAX_LENGTH * model.exp_config.PREPROCESS_PERIOD_MINUTES,
                    latest_data_from = str(model.latest_time)
                    )
                Predictions(model_id=model.obj.model_id, deal_type=model.exp_config.DEAL_TYPE, created_at=datetime.now(timezone.utc)).save_safe()
            else:
                one_model_response = dict(model_id = model.obj.model_id, recommendation=0, latest_data_from = str(model.latest_time))
            advices.append(one_model_response)
        reponse_obj = dict(
            advices=advices,
            code=200
        )
        response_json = json.dumps(reponse_obj)
        return response_json

    def dataset_factory(self, exp_config:ExperimentConfiguration):
        return Dataset(
                experiment_configuration = exp_config, 
                add_label=False, 
                turn_on_rebalancing=False,
                tags='INFERENCE',
                data_source=self.source
            )
