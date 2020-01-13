import pandas as pd
import numpy as np
import logger
import logging
from sklearn.preprocessing import LabelEncoder

class DataLoad:
    ####################################################
    # 定数宣言
    ####################################################
    FILE_TRAIN_CSV = '../input/sales_train_v2.csv'
    FILE_TEST_CSV = '../input/test.csv'
    #FILE_TRAIN_CSV = '../input/sales_train_sample.csv'
    #FILE_TEST_CSV = '../input/test_sample.csv'
    FILE_ITEM_CATEGORIES_CSV = '../input/item_categories.csv'
    FILE_ITEMS_CSV = '../input/items.csv'
    FILE_SHOPS_CSV = '../input/shops.csv'

    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self,windows_size):
        #CSVデータの読み込み
        self.log.info('DataLoad constructor start')

        ## Load test data
        test_df_csv = pd.read_csv(self.FILE_TEST_CSV, header=0,
            dtype = {
                'ID':'str',
                'shop_id':'int',
                'item_id':'int'})

        ## Load training data
        train_df_csv = pd.read_csv(self.FILE_TRAIN_CSV, header=0,
            dtype = {
                'date':'str',
                'date_block_num':'int',
                'shop_id':'int',
                'item_id':'int',
                'item_price':'float',
                'item_cnt_day':'float'})

        ## Load FILE_ITEM_CATEGORIES_CSV data
        item_categories_df_csv = pd.read_csv(self.FILE_ITEM_CATEGORIES_CSV, header=0,
            dtype = {
                'item_category_name':'str',
                'item_category_id':'int'})

        ## Load FILE_ITEMS_CSV data
        items_df_csv = pd.read_csv(self.FILE_ITEMS_CSV, header=0,
            dtype = {
                'item_name':'str',
                'item_id':'int',
                'item_category_id':'int'})

        ## Load FILE_SHOPS_CSV data
        shops_df_csv = pd.read_csv(self.FILE_SHOPS_CSV, header=0,
            dtype = {
                'shop_name':'str',
                'shop_id':'int'})

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

        # shopsを結合する
        train_df = pd.merge(train_df, shops_df_csv, on='shop_id', how='left')

        # itemsを結合する
        train_df = pd.merge(train_df, items_df_csv, on='item_id', how='left')

        # item_categoriesを結合する
        train_df = pd.merge(train_df, item_categories_df_csv, on='item_category_id', how='left')

        # LabelEncoderの実施
        le = LabelEncoder()
        train_df['shop_name'] = pd.DataFrame({'shop_name':le.fit_transform(train_df['shop_name'])})
        train_df['item_name'] = pd.DataFrame({'item_name':le.fit_transform(train_df['item_name'])})
        train_df['item_category_name'] = pd.DataFrame({'item_category_name':le.fit_transform(train_df['item_category_name'])})

        self.df = train_df
        self.test_df = test_df_csv

        self.log.info('DataLoad constructor end')

    # トレーニングデータの取得
    def getTrainValues(self):
        return self.df
        
    # テストデータの取得
    def getTestValues(self):
        return self.test_df
