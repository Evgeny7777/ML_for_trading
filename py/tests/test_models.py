import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + '/../..')
import sys; sys.path.append(os.getcwd())

from py.models import SKModel, ModelInterface
import py.mongo as mongo


def main():
    db_client = mongo.connect_to_mongo(db_name='evo_v2', drop_db=True, localhost=False)
    try:
        ModelInterface.drop_from_cloud(object_id='5d2183b7eb5d66251fb1896f')
    finally:
        db_client.close()
    

if __name__ == '__main__':
    main()