import pandas as pd
# import xgboost as xgb
from xgboost import XGBRegressor

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.visualization import visual
from utils.metric import *
from utils.data_load import *

import numpy as np
import matplotlib.pyplot as plt

'''
    참고: xgboost feature importance : https://hongl.tistory.com/131
'''

def run():   
    d_path = "../dataset/final_kimchi_dataset.csv"
    df = pd.read_csv(d_path)
    
    df_test = df.iloc[9:]
    
    train_x, train_y, test_x, test_y = data_loader(df)

    # model train
    xg_model = XGBRegressor(max_depth=2, n_estimators=100)
    xg_model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=50, verbose=False)

    # predict
    prediction = xg_model.predict(test_x)

    gt_x = df['Date'].tolist()
    gt_y = df['Price'].tolist()
    prediction_x = df_test['Date'].tolist()
    prediction_y = prediction.tolist()

    # print(gt_x)
    # print(gt_y)
    # print(prediction_x)
    # print(prediction_y)
    
    save_path = os.path.join('results', 'xgboost')
    os.makedirs(save_path, exist_ok=True)
    
    # save results
    result_df = pd.DataFrame({
        'Date' : prediction_x,
        'Prediction' : prediction_y
    })
    result_df.to_csv(os.path.join(save_path, 'results.csv'))
    
    # visual prediction
    visual(gt_x, gt_y, prediction_x, prediction_y, os.path.join(save_path, 'xgboost_prediction.png'))
    
    # cal RMSE
    rmse = np.array(cal_RMSE(gt_y[9:], np.array(prediction_y)))
    print('rmse : {}\n'.format(rmse))
    
    # # extract feature importance
    feat_importances = pd.Series(xg_model.feature_importances_, index=train_x.columns)
    feat_importances.nlargest(15).plot(kind='barh')

    plt.title('xgboost feature importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'xgboost_feature_importance.png'))
    

if __name__ == '__main__':
    run()