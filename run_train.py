# m3gnet/run_train.py (Final Version with Embedding Type Configuration)

import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure, Lattice, Molecule
from tqdm import tqdm
import platform

# --- [ SCRIPT SETUP ] ---
try:
    from . import M3GNet, PropertyTrainer, ModelCheckpoint, EarlyStopping, RadiusCutoffGraphConverter
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from m3gnet import M3GNet, PropertyTrainer, ModelCheckpoint, EarlyStopping, RadiusCutoffGraphConverter

# ==============================================================================
# --- [ CONFIGURATION ZONE ] ---
# All hyperparameters and settings are defined here for easy modification.
# ==============================================================================

# --- Path Settings ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "cif_file")
CSV_PATH = os.path.join(DATA_PATH, "id_prop.csv")
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor")

# --- Model Hyperparameters ---
# <<<<<<<<<<<<<<<<<<<< NEW CONFIGURATION OPTION HERE <<<<<<<<<<<<<<<<<<<<
# Choose between 'attention' for context-aware embedding or 'simple' for standard embedding.
EMBEDDING_TYPE = "simple" 

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
USE_EARLY_STOPPING = True
PATIENCE = 15

# --- Performance Settings ---
NUM_WORKERS = 0 
PIN_MEMORY = True if DEVICE == "cuda" else False

def main():
    """Main training function."""
    
    global NUM_WORKERS
    if platform.system().lower() != 'windows':
        try:
            num_cores = len(os.sched_getaffinity(0))
            NUM_WORKERS = num_cores // 2
        except AttributeError:
            num_cores = os.cpu_count()
            NUM_WORKERS = num_cores // 2 if num_cores else 0
    
    config = {
        "Device": DEVICE,
        "Embedding Type": EMBEDDING_TYPE,
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Learning Rate": LEARNING_RATE,
        "Early Stopping": "Enabled" if USE_EARLY_STOPPING else "Disabled",
        "Patience": PATIENCE if USE_EARLY_STOPPING else "N/A",
        "Num Workers": NUM_WORKERS,
        "Pin Memory": PIN_MEMORY,
        "Save Directory": os.path.abspath(SAVE_DIR),
        "Data Path": os.path.abspath(DATA_PATH)
    }
    
    print("\n--- M3GNet Training Configuration ---")
    for key, value in config.items():
        print(f"{key:<20}: {value}")
    print("-------------------------------------\n")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(DATA_PATH, x))
    structures = [Structure.from_file(f) for f in df['filepath']]
    targets = df['property'].values
    print(f"Loaded {len(structures)} structures.")

    # --- Initialize Model ---
    # <<<<<<<<<<<<<<<<<<<< USING THE NEW CONFIGURATION <<<<<<<<<<<<<<<<<<<<
    print(f"Initializing model with '{EMBEDDING_TYPE}' embedding...")
    model = M3GNet(
        is_intensive=True, 
        n_atom_types=95,
        embedding_type=EMBEDDING_TYPE # Pass the configured type to the model
    )
    model.to(DEVICE)
    converter = model.graph_converter
    
    # --- Initialize Lazy Layers ---
    print("Initializing lazy layers with a dummy graph...")
    dummy_molecule = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dummy_graph = converter.convert(dummy_molecule)
    
    from m3gnet.graph.batch import collate_list_of_graphs
    dummy_batch, _ = collate_list_of_graphs([(dummy_graph, torch.tensor(0.0))])
    
    with torch.no_grad():
        model(dummy_batch.to(DEVICE))
    
    print("\n--- Model Architecture (Initialized) ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print("----------------------------------------\n")

    print("Pre-processing structures into graphs... (This may take a moment)")
    graphs = [converter.convert(s) for s in tqdm(structures, desc="Converting")]
    
    train_graphs, val_graphs, train_targets, val_targets = train_test_split(
        graphs, targets, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(train_graphs)}")
    print(f"Validation set size: {len(val_graphs)}")

    print("Initializing optimizer, scheduler, and trainer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = len(train_graphs) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1)
    
    trainer = PropertyTrainer(model=model, optimizer=optimizer, device=DEVICE)

    print("Setting up callbacks...")
    checkpoint = ModelCheckpoint(save_dir=SAVE_DIR, monitor="val_loss", mode="min")
    callbacks = [checkpoint]
    if USE_EARLY_STOPPING:
        early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        callbacks.append(early_stopper)

    print("\n--- Starting Training ---")
    trainer.train(
        train_graphs=train_graphs, train_targets=train_targets,
        val_graphs=val_graphs, val_targets=val_targets,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
        scheduler=scheduler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print("\n--- Training complete! ---")
    print(f"Best model saved to: {os.path.join(SAVE_DIR, 'best_model.pt')}")
    print(f"Last model saved to: {os.path.join(SAVE_DIR, 'last_model.pt')}")
    print("\nTo evaluate the best model, run:")
    print(f"python m3gnet/predict.py --model-path {os.path.join(SAVE_DIR, 'best_model.pt')} --input-path {DATA_PATH}")

if __name__ == "__main__":
    main()