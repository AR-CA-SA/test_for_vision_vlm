import json
import numpy as np
import pandas as pd

# NOTE: order matches CSV row order (bowl, pot, mug230, mug212, bnf, GLASS, PLATE)

to_iterate_w_reference = [
    "recorded_outputs_w_reference/w_reference_metal_bowl_247",
    "recorded_outputs_w_reference/w_reference_metal_pot_741",
    "recorded_outputs_w_reference/w_reference_mug_230",
    "recorded_outputs_w_reference/w_reference_mug_212",
    "recorded_outputs_w_reference/w_reference_bnf_bottle_277",
    "recorded_outputs_w_reference/w_reference_glass_average",   # was plate_average — swapped
    "recorded_outputs_w_reference/w_reference_plate_average",   # was glass_average — swapped
]


to_iterate_w_dimensions = [
    "recorded_outputs_w_dimensions/w_dimensions_metal_bowl_247",
    "recorded_outputs_w_dimensions/w_dimensions_metal_pot_741",
    "recorded_outputs_w_dimensions/w_dimensions_mug_230",
    "recorded_outputs_w_dimensions/w_dimensions_mug_212",
    "recorded_outputs_w_dimensions/w_dimensions_bnf_bottle_277",
    "recorded_outputs_w_dimensions/w_dimensions_glass_average",   # was plate_average — swapped
    "recorded_outputs_w_dimensions/w_dimensions_plate_average",   # was glass_average — swapped
]

to_iterate_no_dimensions = [
    "recorded_outputs_no_dimensions/no_dimensions_metal_bowl_247",
    "recorded_outputs_no_dimensions/no_dimensions_metal_pot_741",
    "recorded_outputs_no_dimensions/no_dimensions_mug_230",
    "recorded_outputs_no_dimensions/no_dimensions_mug_212",
    "recorded_outputs_no_dimensions/no_dimensions_bnf_bottle_277",
    "recorded_outputs_no_dimensions/no_dimensions_glass_average",   # was plate_average — swapped
    "recorded_outputs_no_dimensions/no_dimensions_plate_average",   # was glass_average — swapped
]

CSV_PATH = "impactNoiseDataUseful.csv"


def load_predicted(path):
    """
    Loads a predicted-output JSON file shaped like:
    [
      {"DF": 3000, "INDB": 75},
      {"DF": 3000, "INDB": 20},
      {"DF": 3000, "INDB": 85}
    ]
    Returns (pred_freq_list, pred_in_list), each length 3.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) != 3:
        raise ValueError(
            f"'{path}': expected a JSON list of 3 trial objects, "
            f"got {type(data)} with length "
            f"{len(data) if isinstance(data, list) else 'N/A'}."
        )

    pred_freq, pred_in = [], []
    for trial_idx, entry in enumerate(data):
        if "DF" not in entry or "INDB" not in entry:
            raise ValueError(
                f"'{path}' trial {trial_idx}: missing 'DF' or 'INDB' key. "
                f"Got keys: {list(entry.keys())}"
            )
        pred_freq.append(entry["DF"])
        pred_in.append(entry["INDB"])

    return pred_freq, pred_in


def error_metrics(real, pred):
    """Given two length-3 lists, return dict of error metrics."""
    real = np.array(real, dtype=float)
    pred = np.array(pred, dtype=float)
    abs_err = np.abs(real - pred)
    pct_err = np.where(real != 0, abs_err / np.abs(real) * 100, np.nan)
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    mape = np.nanmean(pct_err)
    return {
        "abs_err_per_trial": abs_err.tolist(),
        "pct_err_per_trial": pct_err.tolist(),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


results = []


def calculate_and_print_metrics(json_predictions, real_values_df):
    df = pd.read_csv(real_values_df)
    for i in range(len(json_predictions)):
        path = json_predictions[i]
        name = df.iloc[i, 0]

        try:
            pred_freq, pred_in = load_predicted(path)
        except FileNotFoundError:
            print(f"[SKIP] Row {i} ('{name}'): file not found -> {path}")
            continue

        real_freq = df.iloc[i, 4:7].to_list()   # Dominant Frequency 1,2,3
        real_in = df.iloc[i, 7:10].to_list()    # Intensity 1,2,3

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

        freq_metrics = error_metrics(real_freq, pred_freq)
        in_metrics = error_metrics(real_in, pred_in)

        results.append({
            "object": name,
            "real_freq": real_freq,
            "pred_freq": pred_freq,
            "freq_MAE": freq_metrics["MAE"],
            "freq_RMSE": freq_metrics["RMSE"],
            "freq_MAPE_%": freq_metrics["MAPE"],
            "real_in": real_in,
            "pred_in": pred_in,
            "in_MAE": in_metrics["MAE"],
            "in_RMSE": in_metrics["RMSE"],
            "in_MAPE_%": in_metrics["MAPE"],
        })

        print(f"--- {name} ---")
        print(f"  Freq   real={real_freq}  pred={pred_freq}")
        print(f"    MAE={freq_metrics['MAE']:.2f} Hz  RMSE={freq_metrics['RMSE']:.2f} Hz  MAPE={freq_metrics['MAPE']:.2f}%")
        print(f"  Intens real={real_in}  pred={pred_in}")
        print(f"    MAE={in_metrics['MAE']:.2f} dB  RMSE={in_metrics['RMSE']:.2f} dB  MAPE={in_metrics['MAPE']:.2f}%")
        print()

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("error_metrics_summary.csv", index=False)
        print("Saved per-object error metrics -> error_metrics_summary.csv")
        print()

        print("=== OVERALL (mean across objects) ===")
        print(f"Frequency  MAE:  {results_df['freq_MAE'].mean():.2f} Hz")
        print(f"Frequency  RMSE: {results_df['freq_RMSE'].mean():.2f} Hz")
        print(f"Frequency  MAPE: {results_df['freq_MAPE_%'].mean():.2f} %")
        print(f"Intensity  MAE:  {results_df['in_MAE'].mean():.2f} dB")
        print(f"Intensity  RMSE: {results_df['in_RMSE'].mean():.2f} dB")
        print(f"Intensity  MAPE: {results_df['in_MAPE_%'].mean():.2f} %")
    else:
        print("No JSON files found — nothing to compute. "
            "Place the 7 predicted-output JSON files in "
            "'recorded_outputs_no_dimensions/' next to this script.")

if __name__ == "__main__":
    print("=== ERROR METRICS FOR PREDICTIONS WITH REFERENCE ===")
    calculate_and_print_metrics(to_iterate_w_reference, CSV_PATH)
    print("\n=== ERROR METRICS FOR PREDICTIONS WITH DIMENSIONS ===")
    calculate_and_print_metrics(to_iterate_w_dimensions, CSV_PATH)
    print("\n=== ERROR METRICS FOR PREDICTIONS WITHOUT DIMENSIONS ===")
    calculate_and_print_metrics(to_iterate_no_dimensions, CSV_PATH)

    
