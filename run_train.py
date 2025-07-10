# m3gnet/run_train.py (Final Simplified Version)

import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure

# --- [ SCRIPT SETUP ] ---
# This block ensures that the script can find the 'm3gnet' package
# when you run it directly from inside the 'm3gnet' directory.
# It adds the parent directory (e.g., 'Desktop') to Python's path.
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PACKAGE_ROOT)

# Now we can use absolute imports from the m3gnet package
from m3gnet import (
    M3GNet, 
    PropertyTrainer, 
    ModelCheckpoint, 
    EarlyStopping
)

# --- [ CONFIGURATION ] ---
# Paths are defined relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "aaa")
CSV_PATH = os.path.join(DATA_PATH, "id_prop.csv")
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 10 
USE_EARLY_STOPPING = True

def main():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    print(f"Models will be saved to: {os.path.abspath(SAVE_DIR)}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(DATA_PATH, x))
    structures = [Structure.from_file(f) for f in df['filepath']]
    targets = df['property'].values
    print(f"Loaded {len(structures)} structures.")

    train_structs, val_structs, train_targets, val_targets = train_test_split(
        structures, targets, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(train_structs)}")
    print(f"Validation set size: {len(val_structs)}")

    print("Initializing model and trainer...")
    model = M3GNet(is_intensive=True, n_atom_types=95)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = PropertyTrainer(model=model, optimizer=optimizer, device=DEVICE)

    print("Setting up callbacks...")
    checkpoint = ModelCheckpoint(save_dir=SAVE_DIR, monitor="val_loss", mode="min")
    
    callbacks = [checkpoint]
    if USE_EARLY_STOPPING:
        early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        callbacks.append(early_stopper)
        print("Early stopping is ENABLED.")
    else:
        print("Early stopping is DISABLED. Training will run for all epochs.")

    print("Starting training...")
    trainer.train(
        train_structures=train_structs, train_targets=train_targets,
        val_structures=val_structs, val_targets=val_targets,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(SAVE_DIR, 'best_model.pt')}")
    print(f"Last model saved to: {os.path.join(SAVE_DIR, 'last_model.pt')}")
    print("\nTo evaluate the best model, run the following command from this directory:")
    print(f"python run_evaluate.py --model-dir {os.path.relpath(SAVE_DIR)}")


if __name__ == "__main__":
    main()