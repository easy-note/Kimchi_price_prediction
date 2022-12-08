
import matplotlib.pyplot as plt

def visual(gt_x, gt_y, prediction_x, prediction_y, save_path):
    print('\nvisual')
    plt.plot(gt_x, gt_y, label='Actual')
    plt.plot(prediction_x, prediction_y, label='Prediction')
    
    plt.xticks(rotation=45)

    plt.legend()
    
    plt.xlabel('Date')
    plt.ylabel('Kimchi Price')
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.cla() # Clear the current axes
    
    
if __name__ == '__main__':
    gt_x = ['2018-01-07', '2018-01-14', '2018-01-21', '2018-01-28', '2018-02-04', '2018-02-11', '2018-02-18', '2018-02-25', '2018-03-04', '2018-03-11', '2018-03-18', '2018-03-25']
    gt_y = [1.91, 1.73, 1.82, 1.72, 1.83, 1.85, 1.83, 1.79, 1.92, 1.85, 1.83, 1.74]
    
    prediction_x = ['2018-03-04', '2018-03-11', '2018-03-18', '2018-03-25']
    # prediction_y = [1.7577666, 1.7577666, 1.6835145, 1.6835145]
    prediction_y = [1.7824866771697998, 1.7752342224121094, 1.7323498725891113, 1.723191499710083]
    
    visual(gt_x, gt_y, prediction_x, prediction_y)
    