import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from formatter import to_json
import json

def average_from_json():


    to_iterate = ["recorded_outputs_w_dimensions/w_dimensions_glass_147",
                  "recorded_outputs_w_dimensions/w_dimensions_glass_149",
                  "recorded_outputs_w_dimensions/w_dimensions_glass_150"]


    keys_ent =["DF_1","DF_2","DF_3"]
    keys_enti = ["INDB_1","INDB_2","INDB_3"]
    averageVLMPredictedValues = []


    for i in range(3):
        averageVLMPredictedValues.append({keys_ent[i] : 0, keys_enti[i] : 0})
    print(averageVLMPredictedValues)
    for i in range(3):
        print(averageVLMPredictedValues)
        df = pd.read_json(to_iterate[i])
        current_dic = averageVLMPredictedValues[i]
        print(current_dic, i )
        current_dic[f"DF_{i+1}"] = np.mean(df.iloc[:, 0 ])
        current_dic[f"INDB_{i+1}"] = np.mean(df.iloc[:, 1])
        print(averageVLMPredictedValues)
    with open("recorded_outputs_w_dimensions/w_dimensions_glass_average", 'w', encoding = 'utf-8') as f:
        json.dump(averageVLMPredictedValues, f, ensure_ascii=False, indent=4)
average_from_json()