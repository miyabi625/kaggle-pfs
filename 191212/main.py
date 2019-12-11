####################################################
# インポート
####################################################
import data_load
#import model as model
#import submit_csv
import logger
import logging
import numpy as np

####################################################
# ログ宣言
####################################################
log = logging.getLogger(__name__)
logger.setLogger(log)

####################################################
# データ読み込み
####################################################
log.info('start read data')
# トレーニングデータ
train_dl = data_load.DataLoad("./input/sales_train.csv")

log.info('end read data')

####################################################
# 分析
####################################################
log.info('start analysis')

### トレーニングデータ用意  ###################
# トレーニングデータを取得する
train_data = train_dl.getValues()

print(train_data)

log.info('end analysis')
