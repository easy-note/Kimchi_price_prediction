
'''
    region 별 csv 확인
'''
import os
import pandas as pd

def main(is_postprocessing):
    df = pd.read_excel("../dataset/Kimchi_dataset.xlsx", engine = "openpyxl")
    df['Region'] = df.Region.str.lower()
    df = df.sort_values(by='Date')
    
    if is_postprocessing:
        df = df[df['Price'] <= 5] # 결측치 제거
    
    print(df)

    regions = df['Region'].tolist()
    regions = [x.lower() for x in regions]
    regions = set(regions)

    for region in regions:
        region_df = df[df['Region'] ==  region]
        
        print(region_df)
        if is_postprocessing:
            save_path = '../dataset/after_post_processing_resion'
        else:
            save_path = '../dataset/before_post_processing_Resion'
        
        os.makedirs(save_path, exist_ok=True)
        region_df.to_csv(os.path.join(save_path, '{}.csv'.format(region)), index=None)
        
        print(region_df)


if __name__ == '__main__':
    is_postprocessing = False
    main(is_postprocessing)
    
    is_postprocessing = True
    main(is_postprocessing)