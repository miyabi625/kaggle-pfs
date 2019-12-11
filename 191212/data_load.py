import pandas as pd
import numpy as np
import logger
import logging
from sklearn.preprocessing import LabelEncoder

class DataLoad:
    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self, file_path):
        self.log.info('init start')

        # Load training data
        tmp_df = pd.read_csv(file_path, header=0)
        tmp_df["date_year"] = tmp_df['date'].str[6:10]
        tmp_df["date_month"] = tmp_df['date'].str[3:5]
        tmp_df = tmp_df.drop("date", axis=1)
        tmp_df = tmp_df.drop("item_id", axis=1)

        #お試しで年月で集計してみる
        tmp_df = tmp_df.groupby(["date_year","date_month"]).agg({"item_price": "mean",
                    "item_cnt_day":"sum"})

        self.df = tmp_df

        self.log.info('init end')

    # 該当項目の取得
    def getValues(self):
        return self.df.values
