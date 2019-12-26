import pandas as pd
import numpy as np
import logger
import logging
from sklearn.preprocessing import LabelEncoder

class DataLoad:
    ####################################################
    # 定数宣言
    ####################################################
    #FILE_TRAIN_CSV = '../input/sales_train.csv'
    FILE_TRAIN_CSV = '../input/sales_train_v2.csv'
    #FILE_TEST_CSV = '../input/test.csv'
    FILE_TEST_CSV = '../input/test2.csv'

    LOOP_COUNT = 3

    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self,windows_size):
        #CSVデータの読み込み
        self.log.info('DataLoad constructor start')
        
        #ID	shop_id	item_id
        #20400	2	5037

        ## Load test data
        test_df_csv = pd.read_csv(self.FILE_TEST_CSV, header=0,
            dtype = {
                'ID':'str',
                'shop_id':'int',
                'item_id':'int'})

        #date	date_block_num	shop_id	item_id	item_price	item_cnt_day
        #14.01.2013	0	2	11330	149	1

        ## Load training data
        train_df_csv = pd.read_csv(self.FILE_TRAIN_CSV, header=0,
            dtype = {
                'date':'str',
                'date_block_num':'int',
                'shop_id':'int',
                'item_id':'int',
                'item_price':'float',
                'item_cnt_day':'float'})

        # testデータをtrainデータに合わせる
        train_df = pd.DataFrame()
        for i in range(35):
            tmp = test_df_csv[['shop_id','item_id']]
            tmp['date_block_num'] = i
            train_df = pd.concat([train_df,tmp],axis=0)
        
        # 月別に集計する
        mon_train_df = train_df_csv.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg({
            #'item_price':np.mean,
            'item_cnt_day':np.sum}).rename(columns={'item_cnt_day':'item_cnt_month'})
        
        # 0～20の範囲でクリップする
        mon_train_df['item_cnt_month'] = mon_train_df['item_cnt_month'].clip(0,20)

        # testデータに月別集計の結果をマージする
        train_df = pd.merge(train_df,mon_train_df,on=['date_block_num','shop_id', 'item_id'], how='left').fillna(0)
        
        # shop_id*item_id*date_block_numでソート
        train_df = train_df.sort_values(
            ['shop_id', 'item_id','date_block_num'],
            ascending=[True, True,True]
        ).reset_index(drop=True)

        # lag用データを保持する
        lag_df = train_df

        # testデータに月別の売り上げ数をマージする
        for i in range(1,windows_size):
            train_df = pd.concat([train_df, lag_df.shift(i).rename(columns={'item_cnt_month': 'lag'+str(i)})['lag'+str(i)]], axis=1)

        # N/Aを0に置換する
        train_df = train_df.fillna(0)
        
        self.df = train_df
        self.test_df = test_df_csv

        self.log.info('DataLoad constructor end')

    # トレーニングデータの取得
    def getTrainValues(self):
        return self.df
        
    # テストデータの取得
    def getTestValues(self):
        return self.test_df
