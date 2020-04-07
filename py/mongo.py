import json
from pprint import pprint, pformat
import time 

from mongoengine import *
from mongoengine.connection import disconnect
import pymongo

from py.config import Config as conf
from py.logger import l


def connect_to_mongo(db_name=None, drop_db=False, localhost=False):
    disconnect()
    db_name_local = db_name if db_name else conf.MONGO_DB_NAME
    if localhost:
        db_client = connect(host=conf._LOCAL_MONGO_CONN_STRING_TEMPLATE.format(db_name=db_name_local))
    else:
        db_client = connect(host=conf._MONGO_CONN_STRING_TEMPLATE.format(db_name=db_name_local))
    if drop_db: db_client.drop_database(db_name_local)
    return db_client

class SafeDocumentMixin:
    
    def save_safe(self, *args, **kwargs):
        for attempt in range(5):
            try:
                return self.save(*args, **kwargs)
            except pymongo.errors.AutoReconnect as e:
                wait_t = 0.5 * pow(2, attempt) # exponential back off
                l.warning("PyMongo auto-reconnecting... %s. Waiting %.1f seconds.", str(e), wait_t)
                time.sleep(wait_t)
    
    @classmethod
    def objects_safe(cls, *args, **kwargs):
        for attempt in range(5):
            try:
                return cls.objects(*args, **kwargs)
            except pymongo.errors.AutoReconnect as e:
                wait_t = 0.5 * pow(2, attempt) # exponential back off
                l.warning("PyMongo auto-reconnecting... %s. Waiting %.1f seconds.", str(e), wait_t)
                time.sleep(wait_t)


class Evaluation(EmbeddedDocument, SafeDocumentMixin):
    code = StringField()
    profit = FloatField()
    deals_count = IntField()
    mean = FloatField()
    median = FloatField()
    std = FloatField()
    days = IntField()
    deals_per_day = FloatField()
    best_deal = FloatField()
    worst_deal = FloatField()
    all_deals = ListField(DictField())
    all_deals_test = ListField(DictField())


class ConfigurationValues(DynamicEmbeddedDocument, SafeDocumentMixin):
    _id = StringField()
    
    def to_dict(self):
        return json.loads(self.to_json())


class OptimizationConfiguration(Document, SafeDocumentMixin):
    name = StringField()
    description = StringField()
    created_at = DateTimeField()
    values = DictField()


class Configuration(Document, SafeDocumentMixin):
    name = StringField()
    description = StringField()
    created_at = DateTimeField()
    values = EmbeddedDocumentField(ConfigurationValues)

    def __repr__(self):
        val = self.values.to_json()
        val = json.loads(val)
        return pformat(val)


class Experiment(Document, SafeDocumentMixin):
    best_evaluation_on_test = EmbeddedDocumentField(Evaluation)
    best_evaluation_on_val = EmbeddedDocumentField(Evaluation)    
    best_point = DictField()
    code = StringField(max_length=4)
    configuration = ReferenceField(Configuration)
    optimization_config = ReferenceField(OptimizationConfiguration)
    created_at = DateTimeField()
    started_at = DateTimeField()
    updated_at = DateTimeField()
    finished_at = DateTimeField()
    description = StringField()
    exp_id = StringField()
    initial_config = DictField()
    
    points = ListField(GenericLazyReferenceField())
    models = ListField(GenericReferenceField())
    name = StringField()
    status = StringField()
    description = StringField()
    exp_id = StringField()

    data = DictField()
    space = DictField()
    model_space = DictField()
    model_tuning_params = DictField()
    executor = StringField()
    status_message = StringField()
    

    def __repr__(self):
        val = self.to_json()
        val = json.loads(val)
        fields_to_ignore = ['best_evaluation_on_test', 'best_evaluation_on_val', 'points']
        for f in fields_to_ignore:
            if f in val: val.pop(f)
        return pformat(val)

    def k__repr__(self):
        s = ''
        for field in ['id', 'name', 'status', 'description', 'created_at']:
            s = s + f'{field}: {self[field]}\n'
        s = s + f'space: {pformat(self.space)}\n'
        # if hasattr(self, 'configuration'):
        try:
            s = s + f'config name: {self.configuration.name}\n'
        except:
            pass
        return s


class Point(Document, SafeDocumentMixin):
    step = IntField()
    result = FloatField()
    coordinates = DictField()
    evaluation_on_test = EmbeddedDocumentField(Evaluation)
    evaluation_on_val = EmbeddedDocumentField(Evaluation)
    detailed_evaluation_on_val = ListField(EmbeddedDocumentField(Evaluation))
    detailed_evaluation_on_test = ListField(EmbeddedDocumentField(Evaluation))
    all_deals_test = ListField(DictField())
    test_days = IntField()
    val_days = IntField()
    test_mean = FloatField()
    val_mean = FloatField()
    test_std = FloatField()
    val_std = FloatField()
    test_deals_per_day = FloatField()
    val_deals_per_day = FloatField()
    test_diff = FloatField()
    val_diff = FloatField()
    test_total = FloatField()
    val_total = FloatField()
    test_min = FloatField()
    val_min = FloatField()
    test_max = FloatField()
    val_max = FloatField()
    clf_params = DictField()
    experiment = ReferenceField(Experiment, reverse_delete_rule=CASCADE)
    fine_tuned = BooleanField(default=False)
    
    def to_dict(self):
        return json.loads(self.to_json())


class ModelCollection(Document, SafeDocumentMixin):
    status = StringField()
    description = StringField()
    experiment = ReferenceField(Experiment)
    point = ReferenceField(Point)
    model_queue = GenericReferenceField()
    model_ensemble = ListField(GenericReferenceField())
    step = IntField()
    config = DictField()
    trained_model = StringField()
    model_file = StringField()
    model_id = StringField()
    model_type = StringField()
    created_at = DateTimeField()
    code = StringField()
    future_code = StringField()


class ModelQueue(Document, SafeDocumentMixin):
    point = ReferenceField(Point)
    model = ReferenceField(ModelCollection)
    status = StringField(default='PLANNED')

class Predictions(Document, SafeDocumentMixin):
    model_id = StringField()
    created_at = DateTimeField()
    deal_type = StringField()