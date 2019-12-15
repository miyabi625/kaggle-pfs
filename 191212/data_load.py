import pandas as pd
import numpy as np
import logger
import logging
from sklearn.preprocessing import LabelEncoder

class DataLoad:
    ####################################################
    # 定数宣言
    ####################################################
    #FILE_TRAIN_CSV = './input/sales_train.csv'
    FILE_TRAIN_CSV = './input/sales_train_v2.csv'
    #FILE_TEST_CSV = './input/test.csv'
    FILE_TEST_CSV = './input/test2.csv'

    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self):
        #constructorでは特に処理を行わない
        pass
    
    #トレーニングデータの読み込み
    def read_train_csv(self):
        self.log.info('read_train_csv start')
        
#date	date_block_num	shop_id	item_id	item_price	item_cnt_day
#14.01.2013	0	2	11330	149	1

        ## Load training data
        tmp_df = pd.read_csv(self.FILE_TRAIN_CSV, header=0,
            dtype = {
                'date':'str',
                'date_block_num':'int',
                'shop_id':'int',
                'item_id':'int',
                'item_price':'float',
                'item_cnt_day':'float'})
        
        tmp_df = tmp_df.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg({
            #'item_price':np.mean,
            'item_cnt_day':np.sum})

        self.df = tmp_df

        self.log.info('read_train_csv end')

    #トレーニングデータの読み込み
    def read_test_csv(self):
        self.log.info('read_test_csv start')
        
#ID	shop_id	item_id
#20400	2	5037

        ## Load test data
        tmp_df = pd.read_csv(self.FILE_TEST_CSV, header=0,
            dtype = {
                'ID':'str',
                'shop_id':'int',
                'item_id':'int'})

        self.df_test = tmp_df

        self.log.info('read_test_csv end')

    # トレーニングデータの取得
    def getTrainValues(self):
        return self.df
        
    # テストデータの取得
    def getTestValues(self):
        return self.df_test
