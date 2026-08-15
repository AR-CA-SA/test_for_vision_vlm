import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NOTE: order matches CSV row order (bowl, pot, mug230, mug212, bnf, GLASS, PLATE)
to_iterate = [
    "recorded_outputs_w_reference/w_reference_metal_bowl_247",
    "recorded_outputs_w_reference/w_reference_metal_pot_741",
    "recorded_outputs_w_reference/w_reference_mug_230",
    "recorded_outputs_w_reference/w_reference_mug_212",
    "recorded_outputs_w_reference/w_reference_bnf_bottle_277",
    "recorded_outputs_w_reference/w_reference_glass_average",   # was plate_average — swapped
    "recorded_outputs_w_reference/w_reference_plate_average",   # was glass_average — swapped
]

df = pd.read_csv("impactNoiseDataUseful.csv")
print("CSV shape:", df.shape)
print("CSV columns:", df.columns.tolist())

real_data_df = []
real_data_in = []
predicted_data_df = []
predicted_data_in = []
object_names = []

for i in range(len(to_iterate)):
    current_df = pd.read_json(to_iterate[i])
    pred_freq = current_df.iloc[:, 0].tolist()
    pred_in = current_df.iloc[:, 1].tolist()

    real_freq = df.iloc[i, 4:7].to_list()   # Dominant Frequency 1,2,3
    real_in = df.iloc[i, 7:10].to_list()    # Intensity 1,2,3  <-- fixed slice
    name = df.iloc[i, 0]

    if len(real_freq) != 3:
        raise ValueError(
            f"Row {i} ('{name}'): expected 3 real frequency values from "
            f"df.iloc[{i}, 4:7], got {len(real_freq)}: {real_freq}."
        )
    if len(real_in) != 3:
        raise ValueError(
            f"Row {i} ('{name}'): expected 3 real intensity values from "
            f"df.iloc[{i}, 7:10], got {len(real_in)}: {real_in} "
            f"(CSV has {df.shape[1]} columns total)."
        )
    if len(pred_freq) != 3 or len(pred_in) != 3:
        raise ValueError(
            f"Row {i} ('{name}'): predicted data from '{to_iterate[i]}' "
            f"doesn't have 3 trials (got {len(pred_freq)} freq, {len(pred_in)} in)."
        )

    predicted_data_df.append(pred_freq)
    predicted_data_in.append(pred_in)
    real_data_df.append(real_freq)
    real_data_in.append(real_in)
    object_names.append(name)

# ----- PLOTTING -----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
x_positions = np.arange(len(object_names))
jitter = np.linspace(-0.15, 0.15, 3)

# ----- FREQUENCY PLOT (Top) -----
for i in range(len(object_names)):
    for trial in range(3):
        ax1.scatter(i + jitter[trial], real_data_df[i][trial],
                    color='blue', s=80, alpha=0.7,
                    label='Real' if i == 0 and trial == 0 else "")
        ax1.scatter(i + jitter[trial], predicted_data_df[i][trial],
                    color='red', s=80, alpha=0.7,
                    label='Predicted' if i == 0 and trial == 0 else "")
        ax1.plot([i + jitter[trial], i + jitter[trial]],
                 [real_data_df[i][trial], predicted_data_df[i][trial]],
                 color='grey', alpha=0.4, linewidth=1)


ax1.set_xticks(x_positions)
ax1.set_xticklabels(object_names, rotation=45, ha='right')
ax1.set_ylabel('Dominant Frequency (Hz)', fontsize=12)
ax1.set_title("Dominant Frequency (Hz): Predicted vs Recorded (Cross-Object Reference)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# ----- INTENSITY PLOT (Bottom) -----
for i in range(len(object_names)):
    for trial in range(3):
        ax2.scatter(i + jitter[trial], real_data_in[i][trial],
                    color='blue', s=80, alpha=0.7,
                    label='Real' if i == 0 and trial == 0 else "")
        ax2.scatter(i + jitter[trial], predicted_data_in[i][trial],
                    color='red', s=80, alpha=0.7,
                    label='Predicted' if i == 0 and trial == 0 else "")
        ax2.plot([i + jitter[trial], i + jitter[trial]],
                 [real_data_in[i][trial], predicted_data_in[i][trial]],
                 color='grey', alpha=0.4, linewidth=1)


ax2.set_xticks(x_positions)
ax2.set_xticklabels(object_names, rotation=45, ha='right')
ax2.set_ylabel('Intensity (dB)', fontsize=12)
ax2.set_title("Dominant Frequency (Hz): Predicted vs Recorded (Cross-Object Reference)", fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('vlm_acoustic_comparison_w_reference.png', dpi=300, bbox_inches='tight')
plt.show()