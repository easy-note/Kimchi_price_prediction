
'''
    전국 평균 데이터 생성
'''
import os
import csv
import pandas as pd
import natsort

def main():
    df = pd.read_excel("../dataset/Kimchi_dataset.xlsx", engine = "openpyxl")
    df = df.sort_values(by='Date')
    df['Date'] = df.Date.astype(str)

    # Remove outlier data, Nan value
    df = df[df['Price'] <= 5]
    dfRe = df.replace('',  0)

    print(df)

    dates = df['Date'].tolist()
    dates = set(dates)
    dates = natsort.natsorted(dates)

    final = []
    for date in dates:
        date_df = df[df['Date'] ==  date]
        print(date_df)
        
        price_mean = date_df['Price'].mean()
        total_volume_mean = date_df['Total Volume'].mean()
        total_boxes_mean = date_df['Total Boxes'].mean()
        small_mean = date_df['Small Boxes'].mean()
        large_mean = date_df['Large Boxes'].mean()
        xlarge_mean = date_df['XLarge Boxes'].mean()
        
        final.append([date, price_mean, total_volume_mean, total_boxes_mean, small_mean, large_mean, xlarge_mean])

    save_path = '../dataset'
    os.makedirs(save_path, exist_ok=True)
    
    # save csv file
    with open(os.path.join(save_path, 'preprocessing_kimchi_dataset.csv'),'w') as file :
        write = csv.writer(file)
        write.writerow(['Date','Price','Total Volume','Total Boxes','Small Boxes','Large Boxes','XLarge Boxes'])
        write.writerows(final)

if __name__ == '__main__':
    main()