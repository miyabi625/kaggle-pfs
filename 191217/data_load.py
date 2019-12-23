import pandas as pd
import numpy as np
import logger
import logging
from sklearn.preprocessing import LabelEncoder

class DataLoad:
    ####################################################
    # 定数宣言
    ####################################################
    FILE_TRAIN_CSV = '../input/sales_train.csv'
    #FILE_TRAIN_CSV = './input/sales_train_v2.csv'
    FILE_TEST_CSV = '../input/test.csv'
    #FILE_TEST_CSV = './input/test2.csv'

    LOOP_COUNT = 3

    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self):
        #CSVデータの読み込み
        self.log.info('DataLoad constructor start')
        
        #ID	shop_id	item_id
        #20400	2	5037

        ## Load test data
        test_df = pd.read_csv(self.FILE_TEST_CSV, header=0,
            dtype = {
                'ID':'str',
                'shop_id':'int',
                'item_id':'int'})

        #date	date_block_num	shop_id	item_id	item_price	item_cnt_day
        #14.01.2013	0	2	11330	149	1

        ## Load training data
        sales_train_df = pd.read_csv(self.FILE_TRAIN_CSV, header=0,
            dtype = {
                'date':'str',
                'date_block_num':'int',
                'shop_id':'int',
                'item_id':'int',
                'item_price':'float',
                'item_cnt_day':'float'})

        # testデータをtrainデータに合わせる
        mrg_df = pd.DataFrame()
        for i in range(35):
            tmp = test_df[['shop_id','item_id']]
            tmp['date_block_num'] = i
            mrg_df = pd.concat([mrg_df,tmp],axis=0)
        
        # 月別に集計する
        sales_train_df = sales_train_df.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg({
            #'item_price':np.mean,
            'item_cnt_day':np.sum})
        
        # Windowサイズ
        windows_size = 3 # testデータの11月も含めない期間

        # testデータに月別の売り上げ数をマージする
        for i in range(windows_size):
            t = 33 - i
            mrg_df = pd.merge(mrg_df
                ,sales_train_df[sales_train_df.date_block_num == t].rename(columns={'item_cnt_day': 'cnt'+str(i+1)}).drop('date_block_num', axis=1)
                ,on=['shop_id', 'item_id'], how='left')

        # N/Aを0に置換する
        mrg_df = mrg_df.fillna(0)
        print(mrg_df)
        
        self.df = mrg_df
        self.test_df = test_df

        self.log.info('DataLoad constructor end')

    # トレーニングデータの取得
    def getTrainValues(self):
        return self.df
        
    # テストデータの取得
    def getTestValues(self):
        return self.df_test
