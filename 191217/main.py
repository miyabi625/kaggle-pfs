####################################################
# インポート
####################################################
import data_load
import model as model
import submit_csv
import logger
import logging
import numpy as np
import pandas as pd

####################################################
# 定数宣言
####################################################
# Windowサイズ
WINDOW_SIZE = 3 # testデータの11月も含めた期間

####################################################
# ログ宣言
####################################################
log = logging.getLogger(__name__)
logger.setLogger(log)

####################################################
# データ読み込み
####################################################
log.info('start read data')

#csvデータの読み込み
dl = data_load.DataLoad(WINDOW_SIZE)

log.info('end read data')

####################################################
# 分析
####################################################
log.info('start analysis')

### トレーニングデータ用意  ###################

# トレーニングデータを取得する
train = dl.getTrainValues()
train_ = train[((34-WINDOW_SIZE+1) <= train.date_block_num) & (train.date_block_num <= 33)].reset_index(drop=True)
train_y = train_['item_cnt_month']
train_x = train_.drop(columns=['date_block_num','item_cnt_month'])

log.info(train_y.head())
log.info(train_y.count())
log.info(train_x.head())
log.info(train_x.count())

model = model.Model()
model.fit(train_x.values,train_y.values)

pred = model.predict(train_x)
score = model.predictScore(train_y.values,pred)
log.info(score)

#テストデータに適用
test = dl.getTestValues()

test_ = train[(train.date_block_num == 34)].reset_index(drop=True)
test_x = train_.drop(columns=['date_block_num','item_cnt_month'])

log.info(test_x.head())

pred = model.predict(test_x)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')

test_x = train_.drop(columns=['date_block_num','item_cnt_month'])
test_x['item_cnt_month'] = pred
submission = pd.merge(
    test,
    test_x[['shop_id','item_id','item_cnt_month']],
    on=['shop_id','item_id'],
    how='left'
)
log.info(submission[['ID','item_cnt_month']].head())

# 提出ファイル作成
submission[['ID','item_cnt_month']].to_csv('./output/submission.csv', index=False)

log.info('end output data')
