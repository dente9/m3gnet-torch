# m3gnet/test_predict.py (Final In-Package Version)

import torch
import os
import sys
from pymatgen.core import Structure, Lattice

# --- [ SCRIPT SETUP ] ---
# This block ensures that the script can find the 'm3gnet' package
# when you run it directly from inside the 'm3gnet' directory.
# It adds the parent directory (e.g., 'Desktop') to Python's path.
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# --- [ IMPORTS ] ---
# Now we can use absolute imports from the m3gnet package
from m3gnet import M3GNet
from m3gnet.predict import predict_from_structures, predict_from_files

# --- [ CONFIGURATION ] ---
# Paths are now defined relative to this script's location inside the package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor", "best_model.pt")

# --- [ MAIN FUNCTION ] ---
def main():
    """
    Demonstrates how to call prediction functions from within the package.
    """
    print("--- M3GNet Prediction Demo (run from inside the package) ---")

    # 1. Load the trained model
    print(f"\n[Step 1] Loading trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run `python run_train.py` first to train a model.")
        return
        
    model = M3GNet(is_intensive=True, n_atom_types=95)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")

    # ======================================================================
    # Method 1: Predicting from CIF files (using predict_from_files)
    # ======================================================================
    print("\n\n--- [Method 1: Predicting from CIF files] ---")
    
    cif_input_path = os.path.join(SCRIPT_DIR, "data", "cif_file")
    print(f"Targeting files in: {cif_input_path}")
    
    try:
        results_df = predict_from_files(
            model_path=MODEL_PATH,
            input_path=cif_input_path,
        )
        print("Prediction from files successful. Results:")
        print(results_df.head())
        
    except FileNotFoundError as e:
        print(f"Error during file-based prediction: {e}")

    # ======================================================================
    # Method 2: Predicting from in-memory objects (using predict_from_structures)
    # ======================================================================
    print("\n\n--- [Method 2: Predicting from in-memory Pymatgen objects] ---")
    
    print("Creating virtual structures in memory...")
    bcc_fe = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    nacl = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    super_material = Structure(Lattice.hexagonal(a=3.0, c=5.0), ["Au", "Pt", "Ag"], [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]])
    
    my_structures = [bcc_fe, nacl, super_material]
    
    predictions = predict_from_structures(
        model=model,
        structures=my_structures
    )
    
    print("Prediction from in-memory objects successful. Results:")
    for i, s in enumerate(my_structures):
        print(f"  - Predicted property for {s.formula}: {predictions[i]:.4f}")


if __name__ == "__main__":
    main()