import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score  # 引入 r2_score
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset
from typing import List

# --- [ SCRIPT SETUP ] ---
# Allows the script to find the m3gnet package
try:
    # This allows the script to be run with `python -m m3gnet.predict`
    from .models import M3GNet
    from .graph import RadiusCutoffGraphConverter, StructureOrMolecule
    from .graph.batch import collate_list_of_graphs
except ImportError:
    # This allows the script to be run from the root directory (e.g., `python m3gnet/predict.py`)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from m3gnet.models import M3GNet
    from m3gnet.graph import RadiusCutoffGraphConverter, StructureOrMolecule
    from m3gnet.graph.batch import collate_list_of_graphs


# --- [ HELPER CLASSES & FUNCTIONS ] ---
class _PredictionDataset(Dataset):
    """Internal Dataset for prediction, yields graphs."""

    def __init__(
        self,
        structures: List[StructureOrMolecule],
        converter: RadiusCutoffGraphConverter,
    ):
        self.structures = structures
        self.converter = converter

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        # The collate function expects a tuple of (graph, target), so we provide a dummy target.
        return self.converter.convert(self.structures[idx]), torch.tensor(0.0)


def _calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error, ignoring zero true values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0
    return (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )
        )
        * 100
    )


# --- [ CORE REUSABLE FUNCTIONS ] ---
def predict_from_structures(
    model: M3GNet,
    structures: List[StructureOrMolecule],
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Core prediction logic on a list of structure objects."""
    model.to(device)
    model.eval()
    dataset = _PredictionDataset(structures, model.graph_converter)
    # The collate function for property prediction returns (graph_batch, target_batch)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_list_of_graphs
    )

    all_preds = []
    print("Performing predictions...")
    with torch.no_grad():
        for batch in loader:
            graph_batch, _ = batch
            graph_batch = graph_batch.to(device)
            preds = model(graph_batch)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def predict_from_files(
    model_dir: str, input_path: str, batch_size: int = 32, device: str = "cpu"
) -> pd.DataFrame:
    """Loads a model and performs predictions on a file or directory."""
    print(f"Loading model from directory: {model_dir}")
    model = M3GNet.load(model_dir)

    input_p = Path(input_path)
    if input_p.is_dir():
        filepaths = sorted(glob.glob(os.path.join(input_p, "*.cif")))
        if not filepaths:
            raise FileNotFoundError(f"No .cif files found in {input_p}")
    elif input_p.is_file():
        filepaths = [str(input_p)]
    else:
        raise FileNotFoundError(f"Input path not found: {input_p}")

    structures = [Structure.from_file(f) for f in filepaths]
    # We use filenames without extension for robust matching
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in filepaths]
    print(f"Loaded {len(structures)} structures.")

    predictions = predict_from_structures(model, structures, device, batch_size)
    return pd.DataFrame({"filename": filenames, "predicted_value": predictions})


# --- [ MAIN COMMAND-LINE LOGIC ] ---
def main():
    """Main command-line interface logic."""
    parser = argparse.ArgumentParser(
        description="Perform predictions or evaluations using a trained M3GNet model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the directory containing the saved model (e.g., saved_models/property_predictor/best_model).",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single CIF file or a directory of CIF files.",
    )
    parser.add_argument(
        "--targets-csv",
        type=str,
        default=None,
        help="Path to a CSV file with true values for evaluation. If not provided, will automatically look for 'id_prop.csv' in the input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files. Defaults to a new 'evaluation' sub-folder in the model directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the generation of the parity plot in evaluation mode.",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Smart Mode Detection ---
    input_dir = (
        args.input_path
        if os.path.isdir(args.input_path)
        else os.path.dirname(args.input_path)
    )
    targets_csv_path = (
        args.targets_csv if args.targets_csv else os.path.join(input_dir, "id_prop.csv")
    )
    is_evaluation_mode = os.path.exists(targets_csv_path)

    # --- Output Directory Setup ---
    output_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(args.model_dir, "evaluation_results")
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Run Prediction ---
    results_df = predict_from_files(
        args.model_dir, args.input_path, args.batch_size, device
    )

    # --- Branch based on mode ---
    if is_evaluation_mode:
        print("\n--- [EVALUATION MODE] ---")
        print(
            f"Found '{os.path.basename(targets_csv_path)}'. Comparing predictions against true values."
        )

        try:
            targets_df = pd.read_csv(targets_csv_path)
            if len(targets_df.columns) < 2:
                raise ValueError("CSV must have at least two columns.")

            # Robustly rename columns regardless of original names
            original_cols = targets_df.columns
            targets_df.rename(
                columns={original_cols[0]: "filename", original_cols[1]: "property"},
                inplace=True,
            )
            print(
                f"  - Interpreting CSV columns as: '{original_cols[0]}' -> 'filename', '{original_cols[1]}' -> 'property'"
            )

            # Robustly handle file extensions for matching
            targets_df["filename"] = (
                targets_df["filename"]
                .astype(str)
                .str.replace(r"\.cif$", "", case=False, regex=True)
            )

        except Exception as e:
            print(
                f"Error reading or parsing the targets CSV file at {targets_csv_path}: {e}"
            )
            sys.exit(1)

        # Merge prediction results with true values
        merged_df = pd.merge(results_df, targets_df, on="filename", how="inner")

        if merged_df.empty:
            print(
                "\nWarning: No matching filenames found between predictions and targets CSV."
            )
            print(
                "Please ensure the first column of your CSV contains filenames that match the CIF files (without extension)."
            )
            is_evaluation_mode = False  # Fallback to prediction-only mode
        else:
            print(f"  - Successfully matched {len(merged_df)} entries for evaluation.")

    if is_evaluation_mode:
        # If evaluation is still a go
        merged_df.rename(columns={"property": "true_value"}, inplace=True)
        merged_df["absolute_error"] = np.abs(
            merged_df["true_value"] - merged_df["predicted_value"]
        )
        final_df = merged_df[
            ["filename", "true_value", "predicted_value", "absolute_error"]
        ]

        # --- Calculate and Report Metrics ---
        true_values = final_df["true_value"]
        pred_values = final_df["predicted_value"]

        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)  # Calculate R²
        mape = _calculate_mape(true_values, pred_values)

        print("\n--- Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²):              {r2:.4f}")  # Print R²
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        # Save detailed CSV
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        final_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Detailed evaluation results saved to {csv_path}")

        # --- Plotting ---
        if not args.no_plot:
            print("Generating parity plot...")
            fig, ax = plt.subplots()
            ax.scatter(
                true_values, pred_values, alpha=0.5, s=10, label=f"MAE = {mae:.4f}"
            )
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(
                lims, lims, "k--", alpha=0.75, zorder=0, label="Perfect Match (y=x)"
            )

            # Add R² text to the plot
            text_str = f"$R^2 = {r2:.4f}$"
            ax.text(
                0.95,
                0.05,
                text_str,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
            )

            ax.set_aspect("equal")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title("Model Evaluation: Parity Plot")
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.legend()
            ax.grid(True)
            plot_path = os.path.join(output_dir, "evaluation_plot.png")
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to {plot_path}")

    else:  # This block handles original prediction mode OR fallback from a failed evaluation
        print("\n--- [PREDICTION MODE] ---")
        csv_path = os.path.join(output_dir, "prediction_results.csv")
        results_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Prediction results saved to {csv_path}")


if __name__ == "__main__":
    main()
