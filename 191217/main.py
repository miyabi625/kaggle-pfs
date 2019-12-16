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
#train_data = tmp_df[tmp_df.date_block_num == 32].drop('date_block_num',axis=1)
#test_data = tmp_df[tmp_df.date_block_num == 33].drop('date_block_num',axis=1)
train_data = tmp_df[tmp_df.date_block_num == 33].drop('date_block_num',axis=1)
print(train_data)
test_data = dl.getTestValues()
print(test_data)

ids = test_data['ID']
test_data = test_data.drop('ID',axis=1)

model = model.Model()
model.fit((train_data.values)[0::, 0:(len(train_data.columns)-1)],(train_data.values)[0::, 0::])
#print((test_data.values)[0::, 0:(len(test_data.columns)-1)])
#output = model.predict((test_data.values)[0::, 0:(len(test_data.columns)-1)])
pred = model.predict(test_data.values)
output = pred[0::,2].astype(float)
print(output)


#score = model.predictScore((test_data.values)[0::, 0::],output)
#print(score)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')

sb = submit_csv.SubmitCsv("./output/pfs_submit.csv")
sb.to_csv(["ID", "item_cnt_month"],zip(ids, output))
log.info('end output data')
