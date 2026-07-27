import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def box_plot_maker():
    to_iterate = [
        "recorded_outputs_w_dimensions/w_dimensions_metal_bowl_247",
        "recorded_outputs_w_dimensions/w_dimensions_metal_pot_741",
        "recorded_outputs_w_dimensions/w_dimensions_mug_230",
        "recorded_outputs_w_dimensions/w_dimensions_mug_212",
        "recorded_outputs_w_dimensions/w_dimensions_bnf_bottle_277",
        "recorded_outputs_w_dimensions/w_dimensions_plate_average",
        "recorded_outputs_w_dimensions/w_dimensions_glass_average"

    ]
    df = pd.read_csv("impactNoiseDataUseful.csv")

    real_data_df= []
    real_data_in = []
    predicted_data_df = []
    predicted_data_in = []
    object_names = []

    for i in range(len(to_iterate)):
        print(to_iterate[i])
        current_df = pd.read_json(to_iterate[i])
        predicted_data_df.append(current_df.iloc[:, 0].tolist())
        predicted_data_in.append(current_df.iloc[:,1].tolist())

        real_data_df.append(df.iloc[i, 4:7].to_list())
        real_data_in.append(df.iloc[i,8:11].to_list())
        object_names.append(df.iloc[i, 0])

    n = len(object_names)
    print(df.head())
    
    base_positions = np.arange(n) * 2.0
    real_positions = base_positions - 0.35
    pred_positions = base_positions + 0.35

    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(10, 6))

    bp_real = axs[0].boxplot(
        real_data_df,
        positions=real_positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor = "#514E7A", color = "black")
    )
    bp_pred = axs[0].boxplot(
        predicted_data_df,
        positions=pred_positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor = "#699265", color = "black")
    )
    axs[0].grid(True, alpha = 0.5)
    axs[1].grid(True, alpha = 0.5)
    axs[0].set_xticks(base_positions)
    axs[0].set_xticklabels(object_names, rotation=90, ha="right")
    axs[0].legend(
        [bp_real["boxes"][0], bp_pred["boxes"][0]],
        ["Real", "Predicted"],
        loc="upper right",
    )
    axs[0].set_title("Dstribution of VLM Predicted and Real Dominant Frequency Values Given Dimensions of the Object")




    bp_real = axs[1].boxplot(
        real_data_in,
        positions=real_positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor = "#514E7A", color = "black")
    )
    bp_pred = axs[1].boxplot(
        predicted_data_in,
        positions=pred_positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor = "#699265", color = "black")
    )

    axs[1].set_xticks(base_positions)
    axs[1].set_xticklabels(object_names, rotation=90, ha="right")
    axs[1].legend(
        [bp_real["boxes"][0], bp_pred["boxes"][0]],
        ["Real", "Predicted"],
        loc="upper right",
    )
    axs[0].set_title("Dominant Frequency (Hz): Predicted vs Recorded (Given Object Dimensions)")
    axs[1].set_title("Intensity (dB): Predicted vs Recorded (Given Object Dimensions)")
    axs[0].set_ylabel("Dominant Frequency (Hz)")
    axs[1].set_ylabel("Intensity (dB)")

    plt.tight_layout()
    plt.show()


box_plot_maker()