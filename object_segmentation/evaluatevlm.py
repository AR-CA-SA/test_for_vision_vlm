import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score,explained_variance_score,mean_absolute_percentage_error)
import matplotlib.pyplot as plt

def pre_vs_real_plot(y_true_flat,y_pred_flat, m="", title = "", color='blue'):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, edgecolor='k')

    lims = [min(y_true_flat.min(), y_pred_flat.min()), max(y_true_flat.max(), y_pred_flat.max())]
    plt.plot(lims, lims, color=color, linestyle='--', linewidth=1.5, label='Perfect prediction (y = x)')

    plt.xlabel(f'Actual Values {m} ')
    plt.ylabel(f'Predicted Values {m}')
    plt.title(f'{title} {m}')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.grid()
    plt.show()


def evaluate_model(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true,y_pred) *100
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    res = y_true - y_pred
    relative_abs_error = np.mean(np.abs((y_true - y_pred)/y_true))*100


    return {'MSE' : mse,
            'RMSE' : rmse, "MAE": mae, 
            "MAPE" : mape,
            "R2" : r2,
            "EXPLAINED_VARIANCE" : evs, 
            "RELATIVE_AE" : relative_abs_error,
             "RESIDUALS" : res }


def get():
    df_vlm = pd.read_csv("/home/prg/test_for_vision_vlm/object_segmentation/vlm_predicted_values.csv")
    df_real = pd.read_csv("/home/prg/clip/Object_Audio_Features_correct.csv")
    print("hello")
    real_values = df_real.iloc[:,1:3]
    predicted_values_vlm = df_vlm.iloc[:,3:5]
    print(predicted_values_vlm)

    print(f"{'='*30} Frequency Metrics {'='*30}")
    print(evaluate_model(real_values.iloc[:,1], predicted_values_vlm.iloc[:,1]))
    pre_vs_real_plot(real_values.iloc[:,1], predicted_values_vlm.iloc[:,1], m="Hz", title="VLM Predicted Frequency")
    print(f"{'='*30} Intensity Metrics {'='*30}")
    print(evaluate_model(real_values.iloc[:,0], predicted_values_vlm.iloc[:,0]))
    
    pre_vs_real_plot(real_values.iloc[:,0], predicted_values_vlm.iloc[:,0], m ="dB", title="VLM Predicted Intensity")

get()