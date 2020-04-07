import datetime
import sys; sys.path.append('.')
import os; os.chdir('..')
sys.path.append('.')

import json
import html
from bson import json_util
from datetime import date
from pprint import pformat

from flask import Flask, g, request
from flask import Markup
from flask import url_for

import flask_admin as admin
from flask_admin.model import typefmt

from flask_mongoengine import MongoEngine
from flask_admin.form import rules
from flask_admin.contrib.mongoengine import ModelView

import py.mongo as mongo
# import  mongo
os.chdir('./flask_ui')


app = Flask(__name__)
app.config['MONGODB_DB'] = 'evo_v2'
app.secret_key = 'super secret key'

def pre_format(view, value):
    return Markup('<div class="pre">{}</div>'.format(html.escape(value)))

def json_format(view, value):  
    # useful for mongoengine dict and list fields
    return pre_format(view,  json.dumps(value, indent=2, default=json_util.default))

def date_format(view, value):
    return value.strftime('%d.%m.%Y %H:%M:%S')

def dict_formatter(view, context, model, name):
    return json.dumps(getattr(model, name), indent=4, sort_keys=True)

MY_DEFAULT_FORMATTERS = dict(typefmt.BASE_FORMATTERS)
MY_DEFAULT_FORMATTERS.update({
    date: date_format,
    dict: json_format
})

def nice_json(old_json):
    parsed = json.loads(old_json)
    return json.dumps(parsed, indent=4, sort_keys=True)

# Customized admin views
class PointView(ModelView):
    column_type_formatters = MY_DEFAULT_FORMATTERS
    column_list = ['step', 'experiment',  'test_mean', 'coordinates']
    column_filters = ['experiment', 'step']
    column_formatters = dict(
        test_mean=lambda v, c, m, p: round(m.test_mean, 2),
        # exp_conf=lambda v, c, m, p: Markup(pformat(m.experiment.configuration.values.to_dict()))
        )
    column_sortable_list = ['test_mean', 'step']
    can_view_details = True
    named_filter_urls = True
    can_delete = False
    can_create = False
    can_edit = False


class ExperimentView(ModelView):
    column_type_formatters = MY_DEFAULT_FORMATTERS
    column_list = ('code', 'id', 'name', 'status', 'points', 'space', 'finished_at', 'executor')#, 'configuration')
    column_filters = ['name', 'status', 'code']
    column_editable_list = ['status', 'name']

    # {url_for("experiment.details_view", id=m.id)}
    column_formatters = dict(
        # name=lambda v, c, m, p: Markup(f'<a href={m.status} title="{m.description}">{m.name}</a>'),
        points=lambda v, c, m, p: Markup(f'<a href={url_for("point.index_view", flt1_experiment_objectid_equals=m.id)}>{len(m.points)}</a>'),
        # configuration=lambda v, c, m, p: Markup(f'<a href={url_for("configuration.details_view", id=m.configuration.id)} title={m.configuration.values.to_json()}>{m.configuration.name}</a>'),
        space=dict_formatter,
        # config_values=lambda v, c, m, p: nice_json(m.configuration.values.to_json())
        )
    can_view_details = True
    named_filter_urls = True
    column_display_pk = True


    # column_searchable_list = ('name')

    # form_ajax_refs = {
    #     'tags': {
    #         'fields': ('name',)
    #     }
    # }

class ConfigView(ModelView):
    column_list = ('id', 'name', 'values')
    column_editable_list = ['name']
    column_filters = ['name']

    named_filter_urls = True
    column_display_pk = True
    can_view_details = True
    column_formatters = dict(
        # name=lambda v, c, m, p: Markup(f'<a href=# title="{m.description}">{m.name}</a>'),
        values=lambda v, c, m, p: nice_json(m.values.to_json())
    )
    
class ModelsView(ModelView):
    column_list = ('id', 'model_id', 'code', 'future_code', 'description', 'experiment', 'step_no', 'status')
    column_filters = ['model_id', 'code', 'experiment', 'status']
    column_editable_list = ['description', 'status', 'code', 'future_code']
    column_formatters = dict(
        experiment=lambda v, c, m, p: Markup(f'<a href={url_for("experiment.details_view", id=(0 if not getattr(m, "experiment", None) else m.experiment.id) )}>{"no" if not getattr(m, "experiment", None) else m.experiment.name}</a>')
    )
    can_view_details = True

    # configuration=lambda v, c, m, p: Markup(f'<a href={url_for("configuration.details_view", id=m.configuration.id)} title={m.configuration.values.to_json()}>{m.configuration.name}</a>'),

@app.route('/')
def index():
    return '<a href="/admin/">Click me to get to Admin!</a>'

@app.before_request
def connect_to_db():
    print('Open connection')
    g.db_client = mongo.connect_to_mongo(db_name='evo_v2')

@app.teardown_appcontext
def teardown_db(error):
    if hasattr(g, 'db_client'):
        g.db_client.close()
    print("closing db connection")

if __name__ == "__main__":
    # Create admin
    admin = admin.Admin(app, 'Evotrade dashboard')

    # Add views
    admin.add_view(ExperimentView(mongo.Experiment))
    admin.add_view(PointView(mongo.Point))
    admin.add_view(ConfigView(mongo.Configuration))
    admin.add_view(ModelsView(mongo.ModelCollection))


    # Start app


    app.run(debug=True, port=5006)