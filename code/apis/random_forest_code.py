
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.visualization import visual
from utils.metric import *
from utils.data_load import *

import matplotlib.pyplot as plt

def run():
    d_path = "../dataset/final_kimchi_dataset.csv"
    df = pd.read_csv(d_path)
    
    df_test = df.iloc[9:]
    
    train_x, train_y, test_x, test_y = data_loader(df)

    rf_model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf_model.fit(train_x, train_y)
    
    prediction = rf_model.predict(test_x)

    gt_x = df['Date'].tolist()
    gt_y = df['Price'].tolist()
    prediction_x = df_test['Date'].tolist()
    prediction_y = prediction.tolist()

    # print(gt_x)
    # print(gt_y)
    # print(prediction_x)
    # print(prediction_y)
    
    save_path = os.path.join('results', 'random_forest')
    os.makedirs(save_path, exist_ok=True)
    
    # save results
    result_df = pd.DataFrame({
        'Date' : prediction_x,
        'Prediction' : prediction_y
    })
    result_df.to_csv(os.path.join(save_path, 'results.csv'))
    
    # visual prediction
    visual(gt_x, gt_y, prediction_x, prediction_y, os.path.join(save_path, 'rf_prediction.png'))
    
    # cal RMSE
    rmse = np.array(cal_RMSE(gt_y[9:], np.array(prediction_y)))
    print('rmse : {}\n'.format(rmse))
    
    # extract feature importance
    
    feat_importances = pd.Series(rf_model.feature_importances_, index=train_x.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    
    plt.title('random forest feature importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rf_feature_importance.png'))
    
if __name__ == '__main__':
    run()