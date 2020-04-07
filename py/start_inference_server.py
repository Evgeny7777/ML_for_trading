import json
import os; os.chdir('..')
import time
import sys; sys.path.append('.')

from flask import Flask, request, g

from py.inference import InferencePack
from py.logger import l
import py.mongo as mongo
os.chdir('./py')

sec_codes = ['GZ', 'RI', 'SR', 'GD']
time_code = 'U9'

db_client = mongo.connect_to_mongo()
inf_packs = {}
for sec_code in sec_codes:
    inf_packs[sec_code+time_code] = InferencePack(sec_code+time_code)
db_client.close()

app = Flask(__name__)

@app.before_request
def before_request():
    print('Open connection')
    g.db_client = mongo.connect_to_mongo()

@app.teardown_appcontext
def teardown_db(error):
    if hasattr(g, 'db_client'):
        g.db_client.close()
    print("closing db connection")

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict/<code>', methods=['GET', 'POST'])
def predict(code):
    if request.method == 'GET':
        l.debug(f'got prediction request on {code}')
        if code not in inf_packs:
            return json.dumps(dict(
                code=200,
                message=f"No models for the code {code}",
                advices=[]
            ))
        
        inf_pack:InferencePack = inf_packs[code]
        inf_pack.do_predictions()
        return inf_pack.json_response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)#, ssl_context='adhoc')