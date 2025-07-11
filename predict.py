# m3gnet/predict.py (The Final, Ultimate, Correct Version)

import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

# --- [ SCRIPT SETUP ] ---
# Allows the script to find the m3gnet package
try:
    from . import M3GNet
    from .graph import collate_list_of_graphs, RadiusCutoffGraphConverter, StructureOrMolecule
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from m3gnet import M3GNet
    from m3gnet.graph import collate_list_of_graphs, RadiusCutoffGraphConverter, StructureOrMolecule

# --- [ HELPER CLASSES & FUNCTIONS ] ---
class _PredictionDataset(Dataset):
    """Internal Dataset for prediction, yields graphs."""
    def __init__(self, structures: List[StructureOrMolecule], converter: RadiusCutoffGraphConverter):
        self.structures = structures
        self.converter = converter
    def __len__(self): return len(self.structures)
    def __getitem__(self, idx):
        return self.converter.convert(self.structures[idx]), torch.tensor(0.0)

def _calculate_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# --- [ CORE REUSABLE FUNCTIONS ] ---
def predict_from_structures(
    model: M3GNet, structures: List[StructureOrMolecule], 
    device: str = 'cpu', batch_size: int = 32
) -> np.ndarray:
    """Core prediction logic on a list of structure objects."""
    model.to(device)
    model.eval()
    dataset = _PredictionDataset(structures, model.graph_converter)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_list_of_graphs)
    all_preds = []
    print("Performing predictions...")
    with torch.no_grad():
        for batch in loader:
            graph, _ = batch
            graph = graph.to(device)
            preds = model(graph)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds).flatten()

def predict_from_files(
    model_dir: str, input_path: str,
    batch_size: int = 32, device: str = 'cpu'
) -> pd.DataFrame:
    """Loads a model and performs predictions on a file or directory."""
    print(f"Loading model from directory: {model_dir}")
    model = M3GNet.load(model_dir)
    
    input_p = Path(input_path)
    if input_p.is_dir():
        filepaths = sorted(glob.glob(os.path.join(input_p, "*.cif")))
        if not filepaths: raise FileNotFoundError(f"No .cif files found in {input_p}")
    elif input_p.is_file():
        filepaths = [str(input_p)]
    else:
        raise FileNotFoundError(f"Input path not found: {input_p}")
    
    structures = [Structure.from_file(f) for f in filepaths]
    filenames = [os.path.basename(f) for f in filepaths]
    print(f"Loaded {len(structures)} structures.")

    predictions = predict_from_structures(model, structures, device, batch_size)
    return pd.DataFrame({'filename': filenames, 'predicted_value': predictions})

# --- [ MAIN COMMAND-LINE LOGIC ] ---
def main():
    """Main command-line interface logic."""
    parser = argparse.ArgumentParser(
        description="Perform predictions or evaluations using a trained M3GNet model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # The two required positional arguments
    parser.add_argument("model_dir", type=str, help="Path to the directory containing the saved model (e.g., saved_models/property_predictor/best_model).")
    parser.add_argument("input_path", type=str, help="Path to a single CIF file or a directory of CIF files.")
    
    # Optional arguments
    parser.add_argument("--targets-csv", type=str, default=None, help="Path to a CSV file with true values for evaluation. If not provided, will automatically look for 'id_prop.csv' in the input directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output files. Defaults to a new 'evaluation' sub-folder in the model directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--no-plot", action="store_true", help="Suppress the generation of the parity plot in evaluation mode.")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Smart Mode Detection ---
    input_dir = args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    # Use user-provided CSV if available, otherwise check for the default name
    targets_csv_path = args.targets_csv if args.targets_csv else os.path.join(input_dir, "id_prop.csv")
    is_evaluation_mode = os.path.exists(targets_csv_path)

    # --- Output Directory Setup ---
    output_dir = args.output_dir if args.output_dir else os.path.join(args.model_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    # --- Run Prediction ---
    # This step is common to both modes
    results_df = predict_from_files(args.model_dir, args.input_path, args.batch_size, device)

    # --- Branch based on mode ---
    if is_evaluation_mode:
        print("\n--- [EVALUATION MODE] ---")
        print(f"'id_prop.csv' found. Comparing predictions against true values.")
        
        targets_df = pd.read_csv(targets_csv_path)
        results_df = pd.merge(results_df, targets_df, on="filename")
        
        # Check if 'property' column exists after merge
        if 'property' not in results_df.columns:
            raise ValueError(f"The CSV file at {targets_csv_path} must contain 'filename' and 'property' columns.")
            
        results_df.rename(columns={'property': 'true_value'}, inplace=True)
        results_df['absolute_error'] = np.abs(results_df['true_value'] - results_df['predicted_value'])
        results_df = results_df[['filename', 'true_value', 'predicted_value', 'absolute_error']]

        # Calculate and Report Metrics
        mae = mean_absolute_error(results_df['true_value'], results_df['predicted_value'])
        mape = _calculate_mape(results_df['true_value'], results_df['predicted_value'])
        
        print("\n--- Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        # Save detailed CSV
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Detailed evaluation results saved to {csv_path}")

        # Plotting
        if not args.no_plot:
            print("Generating parity plot...")
            fig, ax = plt.subplots()
            ax.scatter(results_df['true_value'], results_df['predicted_value'], alpha=0.5, label=f"MAE = {mae:.4f}")
            lims = [np.min(results_df[['true_value', 'predicted_value']].values), np.max(results_df[['true_value', 'predicted_value']].values)]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="Perfect Match")
            ax.set_aspect('equal'), ax.set_xlim(lims), ax.set_ylim(lims)
            ax.set_title("Model Evaluation: Parity Plot"), ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values"), ax.legend(), ax.grid(True)
            plot_path = os.path.join(output_dir, "evaluation_plot.png")
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to {plot_path}")
    else:
        print("\n--- [PREDICTION MODE] ---")
        print("'id_prop.csv' not found. Saving predictions only.")
        csv_path = os.path.join(output_dir, "prediction_results.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Prediction results saved to {csv_path}")

if __name__ == "__main__":
    main()