import os; os.chdir('../..')
import pickle
import sys; sys.path.append('.')

from py.classes import Dataset, ExperimentConfiguration, SKModel
from py.my_pd_utils import make_list_if_single
from py.s3 import S3_client

if __name__ == '__main__':
    print('hello')
    ec = ExperimentConfiguration('no_id')
    GDZ8 = Dataset(
        source_string='s3:GDZ8', 
        experiment_configuration=ec, 
        add_label=True, 
        turn_on_rebalancing=False)
    print('end')
    # model = SKModel()

    # model.train_from_experiment_step(exp_id='1553462696', step_no=915)
    # ec = ExperimentConfiguration('no_id')
    
    # dRIM9 = Dataset('db:RIM9', ec)
    # dRIM9.update()
    # print(dRIM9.data.index)
    
    # dataset = Dataset(
        # source_string='db:RIM9',
    #     source_string='f:RIH9',
    #     experiment_configuration=ec,
    #     tags='INFERENCE'
    # )

    # model = SKModel(ec)
    # model.train([dataset], 'INFERENCE')
    # # full_result, all_results = model.evaluate(dataset, 'INFERENCE')
    # model.save()
    # model.load('190403.224701.184004')

    
    # s3c = S3_client()
    # s3c.upload_binary(open('1.txt', 'rb'), f'temp/1.txt')
    # s3c.upload_file('main.py', 'temp/main.py')
    # data = s3c.download_binary('temp/main.py')

    # key = 'temp/test'
    # d1 = {'a':3}
    # print(d1)
    # s3c.upload_binary(pickle.dumps(d1), key)
    # data = s3c.download_binary(key)
    # d2 = pickle.loads(data)
    # print(d2)

    





