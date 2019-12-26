from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import logger
import logging
import numpy as np
import pandas as pd

class Model:
    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    # modelパラメータの設定
    def set_param(self,param):
        # 将来用のメソッドとして用意する。今のところは未定義（pass）
        pass

    # 機械学習
    def fit(self,x_train,y_train):
        self.model = self.model.fit(x_train, y_train)

    # 結果の取得
    def predict(self,test_data):
        return self.model.predict(test_data)
    
    #評価（RMSE）
    def predictScore(self,y_true,y_pred):
        rmse_val = np.sqrt(
            np.mean(
                np.square(
                    np.array(y_true - y_pred)
                )
            )
        )
        return rmse_val