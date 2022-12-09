
'''
    https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from itertools import chain
import random

from models.lstm_model import *
from utils.visualization import visual
from utils.metric import *

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # device

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
      

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

def train_model(model, train_df, num_epochs = None, lr = None, verbose = 10, patience = 10):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    nb_epochs = num_epochs
    
    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in enumerate(train_df):

            x_train, y_train = samples
            
            # seq별 hidden state reset
            model.reset_hidden_state()
            
            # H(x) 계산
            outputs = model(x_train)
                
            # cost 계산
            loss = criterion(outputs, y_train)                    
            
            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_cost += loss/total_batch
               
        train_hist[epoch] = avg_cost        
        
        if epoch % 1 == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
        '''
        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if train_hist[epoch-patience] < train_hist[epoch]:
                print('\n Early Stopping')
                
                break
        '''    
    return model.eval(), train_hist

def MAE(true, pred):
    return np.mean(np.abs(true-pred))


def main():
  ## 1. split train/test set
  # data load
  df = pd.read_csv("../dataset/final_kimchi_dataset.csv")
  
  gt_x = df['Date'].tolist()
  gt_y = df['Price'].tolist()
  prediction_x = gt_x[-3:]
  
  df = df[['Date', 'Total_Volume', 'Total_Boxes', 'Small_Boxes', 'Large_Boxes', 'XLarge_Boxes', 'Price']]
  df = df.drop(['Date'], axis=1)

  seq_length = 2 # window size
  batch = 1

  # 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
  # df = df[::-1]  
#   train_size = int(len(df)*0.7)
  train_size = 9
  train_set = df[0:train_size]  
  test_set = df[train_size-seq_length:]

  ## 2. data scaling
  # Input scale
  scaler_x = MinMaxScaler()
  scaler_x.fit(train_set.iloc[:, :-1])
  # print(scaler_x.n_samples_seen_)
  # print(scaler_x.feature_range)
  # print(scaler_x.data_min_)
  # print(scaler_x.data_max_)
    
  train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
  test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])

  # Output scale
  scaler_y = MinMaxScaler()
  scaler_y.fit(train_set.iloc[:, [-1]])
  # print(scaler_y.n_samples_seen_)
  # print(scaler_y.feature_range)
  # print(scaler_y.data_min_)
  # print(scaler_y.data_max_)

  train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:, [-1]])
  test_set.iloc[:, -1] = scaler_y.transform(test_set.iloc[:, [-1]])

  ## 3. Creating datasets and convert to tensor
  trainX, trainY = build_dataset(np.array(train_set), seq_length)
  testX, testY = build_dataset(np.array(test_set), seq_length)

  trainX_tensor = torch.FloatTensor(trainX).to(device)
  trainY_tensor = torch.FloatTensor(trainY).to(device)

  testX_tensor = torch.FloatTensor(testX).to(device)
  testY_tensor = torch.FloatTensor(testY).to(device)


  # Dataloader
  dataset = TensorDataset(trainX_tensor, trainY_tensor)
  dataloader = DataLoader(dataset,
                          batch_size=batch,
                          shuffle=True,  
                          drop_last=True)

  # LSTM params
  data_dim = 6
  hidden_dim = 10 
  output_dim = 1 
  learning_rate = 0.0001
  nb_epochs = 10
  
  # model train
  net = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)  
  model, train_hist = train_model(net, dataloader, num_epochs = nb_epochs, lr = learning_rate, verbose = 20, patience = 10)

  # loss per epoch
  save_path = os.path.join('results', 'lstm-split1')
  os.makedirs(save_path, exist_ok=True)
    
  # fig = plt.figure(figsize=(10, 4))
  # plt.plot(train_hist, label="Training loss")
  # plt.legend()
  
  # plt.xlabel('Epoch')
  # plt.ylabel('Loss')
    
  # plt.savefig(os.path.join(save_path, 'loss.png'))

  # model save   
  PATH = os.path.join(save_path, "my_lstm-split1.pth")
  torch.save(model.state_dict(), PATH)

  # model load
  model = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)  
  model.load_state_dict(torch.load(PATH), strict=False)
  model.eval()

  # test
  with torch.no_grad(): 
      pred = []
      for pr in range(len(testX_tensor)):

          model.reset_hidden_state()

          predicted = model(torch.unsqueeze(testX_tensor[pr], 0)) # gpu, tensor
          predicted = torch.flatten(predicted).item()
          pred.append(predicted)

      # INVERSE
      pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
      testY_inverse = scaler_y.inverse_transform(testY_tensor.cpu())
      
  print('MAE SCORE : ', MAE(pred_inverse, testY_inverse))

  # save results
  result_df = pd.DataFrame({
      'Date' : prediction_x,
      'Prediction' : list(chain.from_iterable(pred_inverse))
  })
  result_df.to_csv(os.path.join(save_path, 'results.csv'))
  
  # visual prediction
  plt.cla() # Clear the current axes
  # visual(gt_x, gt_y, prediction_x, pred_inverse.tolist(), os.path.join(save_path, 'lstm_prediction.png'))
  visual(gt_x, gt_y, prediction_x, list(chain.from_iterable(pred_inverse)), os.path.join(save_path, 'lstm_prediction.png'))
    
  # cal RMSE
  rmse = np.array(cal_RMSE(gt_y[9:], np.array(list(chain.from_iterable(pred_inverse)))))
  print('rmse : {}\n'.format(rmse))


if __name__ == '__main__':
  seed_everything(33)
  main()