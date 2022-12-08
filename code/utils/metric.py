
from sklearn.metrics import mean_squared_error 
import numpy as np

def cal_RMSE(gt, pred):
    MSE = mean_squared_error(gt, pred) 
    return np.sqrt(MSE)
