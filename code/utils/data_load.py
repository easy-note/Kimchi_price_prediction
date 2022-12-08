
import pandas as pd

def data_loader(df):
    
    df['Date'] = df.Date.astype(str)

    # split train, test set
    df_train = df.iloc[:9]
    df_test = df.iloc[9:]

    train_y = df_train.loc[:,'Price']
    train_x = df_train.drop(labels=['Price','Date'], axis=1)
    test_y = df_test.loc[:,'Price']
    test_x = df_test.drop(labels=['Price','Date'], axis=1)
    
    return train_x, train_y, test_x, test_y