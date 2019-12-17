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
# ログ宣言
####################################################
log = logging.getLogger(__name__)
logger.setLogger(log)

####################################################
# データ読み込み
####################################################
log.info('start read data')

#instance作成
dl = data_load.DataLoad()

# トレーニングデータ
dl.read_train_csv()
# テストデータ
dl.read_test_csv()

log.info('end read data')

####################################################
# 分析
####################################################
log.info('start analysis')

### トレーニングデータ用意  ###################
# トレーニングデータを取得する
tmp_df = dl.getTrainValues()
train_y = tmp_df[['shop_id','item_id','cnt33']]
print(train_y)
train_x = tmp_df.drop('cnt33',axis=1)
print(train_x)
val = tmp_df[['shop_id','item_id']]
print(val)

model = model.Model()
model.fit(train_x.values,train_y.values)
pred = model.predict(val.values)
output = pred[0::,2].astype(float)
print(output)

score = model.predictScore(train_y.values,output)
print(score)

#テストデータに適用
test_data = dl.getTestValues()

ids = test_data['ID']
test_data = test_data.drop('ID',axis=1)

model.fit((train_data.values)[0::, 0:(len(train_data.columns)-1)],(train_data.values)[0::, 0::])

pred = model.predict(test_data.values)
output = pred[0::,2].astype(float)
print(output)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')

sb = submit_csv.SubmitCsv("./output/pfs_submit.csv")
sb.to_csv(["ID", "item_cnt_month"],zip(ids, output))
log.info('end output data')
