# -*- coding: utf-8 -*-
import os
from dotenv import find_dotenv, load_dotenv
from requests import session
import logging



def extract_data(url, file_path):
    with session() as c:
        c.post('https://www.kaggle.com/', data = payload)
        with open(file_path, 'wb') as handle:
            response = c.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)
                
def main(project_dir):
    #get logger
    logger = logging.getLogger(__name__)
    logger.info('getting raw data')
    
    #urls
    train_url = 'https://www.kaggle.com/c/titanic/download/train.csv'
    test_url = 'https://www.kaggle.com/c/titanic/download/test.csv'
    
    #file_paths
    raw_data_path = os.path.join(project_dir,'data','raw')
    train_data_path = os.path.join(raw_data_path, 'train1.csv')
    test_data_path = os.path.join(raw_data_path, 'test1.csv')
    
    #extract data
    extract_data(train_url,train_data_path)
    extract_data(test_url, test_data_path)
    logger.info('downloaded raw training and test data')
    
if __name__ == '__main__':
    #getting root directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    #print(project_dir)

    #setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # find .env automatically by walking up directories until it's found
    dotenv_path = find_dotenv()
    #print(dotenv_path)
    load_dotenv(dotenv_path)

    # payload to login to kaggle
    payload = {
        'action': 'login',
        'username': os.environ.get('KAGGLE_USERNAME'),
        'password': os.environ.get('KAGGLE_PASSWORD')
    }
    print(payload)
    
    #call main
    main(project_dir)