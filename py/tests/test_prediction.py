import os; os.chdir('..')
import time
import sys; sys.path.append('.')
from mongoengine.connection import disconnect
from importlib import reload
import py.classes as classes;reload(classes)
import py.mongo as mongo;reload(mongo)
db = mongo.connect_to_mongo(db_name='evo_v2')
import py.models as models;reload(models)
import py.dataset as dataset;reload(dataset)
import py.inference as inference;reload(inference)

import pprint
import json
import logging