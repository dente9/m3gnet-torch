# m3gnet/run_train.py (Final Version compatible with dual-metric Trainer)

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure, Molecule, Element
from tqdm import tqdm
import platform

# --- [ 1. ROBUST IMPORT HANDLING ] ---
try:
    from m3gnet.models import M3GNet, Potential
    from m3gnet.train import PropertyTrainer, PotentialTrainer, ModelCheckpoint, EarlyStopping
    from m3gnet.graph import RadiusCutoffGraphConverter
    from m3gnet.graph.batch import collate_list_of_graphs, collate_potential_graphs
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from m3gnet.models import M3GNet, Potential
    from m3gnet.train import PropertyTrainer, PotentialTrainer, ModelCheckpoint, EarlyStopping
    from m3gnet.graph import RadiusCutoffGraphConverter
    from m3gnet.graph.batch import collate_list_of_graphs, collate_potential_graphs


# --- [ 2. CONFIGURATION ZONE ] ---
# General settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 0 
PIN_MEMORY = True if DEVICE == "cuda" else False

# --- Task-specific settings ---
# Choose 'property' or 'potential'
# 'property' is used for predicting total energy (or any other property) from structures only.
# 'potential' is used for training on energy, forces, and stresses.
TRAINING_TYPE = "property" 

if TRAINING_TYPE == 'property':
    # Default settings for property prediction (e.g., total energy)
    DATA_PATH = os.path.join(SCRIPT_DIR, "data", "cif_file")
    CSV_PATH = os.path.join(DATA_PATH, "id_prop.csv")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor")
    TARGET_COLUMN = 'property' # Assumes 'property' column contains total energy
    IS_INTENSIVE = False     # Total energy is extensive
    FIT_ELEMENT_REFS = True  # Recommended for total energy
    EMBEDDING_TYPE = "attention"

elif TRAINING_TYPE == 'potential':
    # Settings for potential training (energy, forces, stresses)
    DATA_PATH = os.path.join(SCRIPT_DIR, "data", "efs_data")
    JSON_PATH = os.path.join(DATA_PATH, "efs.json")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "potential_predictor")
    IS_INTENSIVE = False     # Total energy is extensive
    FIT_ELEMENT_REFS = True
    EMBEDDING_TYPE = "attention"
else:
    raise ValueError(f"Unknown TRAINING_TYPE: {TRAINING_TYPE}")

# Callback settings
USE_EARLY_STOPPING = True
PATIENCE = 25


# --- [ 3. HELPER FUNCTION ] ---

def fit_element_refs(structures: list, energies: np.ndarray, n_atom_types: int) -> np.ndarray:
    """
    Fits elemental reference energies by solving a linear system.
    """
    print("Fitting elemental reference energies...")
    feature_matrix = np.zeros((len(structures), n_atom_types))
    for i, s in enumerate(structures):
        for el_key, count in s.composition.get_el_amt_dict().items():
            if isinstance(el_key, str):
                el_obj = Element(el_key)
            else:
                el_obj = el_key
            
            if el_obj.Z < n_atom_types:
                feature_matrix[i, el_obj.Z] = count
    
    ata = feature_matrix.T @ feature_matrix
    atb = feature_matrix.T @ energies
    
    try:
        element_refs = np.linalg.solve(ata, atb)
        print("Elemental reference energies fitted successfully using np.linalg.solve.")
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Using pseudo-inverse (pinv) for fitting.")
        pseudo_inverse = np.linalg.pinv(ata)
        element_refs = pseudo_inverse @ atb

    final_refs = np.zeros(n_atom_types)
    final_refs[:len(element_refs)] = element_refs
    
    print("Fitted non-zero elemental references (eV/atom):")
    for i, ref in enumerate(final_refs):
        if abs(ref) > 1e-6:
            print(f"  - {Element.from_Z(i).symbol} (Z={i}): {ref:.4f}")
            
    return final_refs


# --- [ 4. MAIN FUNCTION ] ---

def main():
    """Main training function."""
    
    global NUM_WORKERS
    if platform.system().lower() != 'windows':
        try:
            num_cores = len(os.sched_getaffinity(0))
            NUM_WORKERS = min(12, num_cores)
        except AttributeError:
            num_cores = os.cpu_count() or 1
            NUM_WORKERS = min(12, num_cores)

    config = {
        "Training Type": TRAINING_TYPE, "Device": DEVICE, "Embedding Type": EMBEDDING_TYPE, 
        "Epochs": EPOCHS, "Batch Size": BATCH_SIZE, "Learning Rate": LEARNING_RATE,
        "Is Intensive": IS_INTENSIVE, "Fit Element Refs": FIT_ELEMENT_REFS,
        "Early Stopping": "Enabled" if USE_EARLY_STOPPING else "Disabled",
        "Patience": PATIENCE if USE_EARLY_STOPPING else "N/A", "Num Workers": NUM_WORKERS,
        "Pin Memory": PIN_MEMORY, "Save Directory": os.path.abspath(SAVE_DIR)
    }
    
    print("\n--- M3GNet Training Configuration ---")
    for key, value in config.items(): print(f"{key:<20}: {value}")
    print("-------------------------------------\n")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Data Loading Logic ---
    print("Loading data...")
    if TRAINING_TYPE == 'potential':
        with open(JSON_PATH, 'r') as f:
            data_list = json.load(f)
        structures = [Structure.from_dict(d['structure']) for d in data_list]
        targets_energy = np.array([d['energy'] for d in data_list])
        targets_forces = [np.array(d['forces']) for d in data_list]
        targets_stresses = [np.array(d['stress']) for d in data_list] if 'stress' in data_list[0] else None
        print(f"Loaded {len(structures)} structures with energies, forces, and stresses.")
        targets = targets_energy
    else: # property training
        df = pd.read_csv(CSV_PATH)
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(DATA_PATH, x))
        structures = [Structure.from_file(f) for f in df['filepath']]
        targets = df[TARGET_COLUMN].values
        print(f"Loaded {len(structures)} structures for property prediction.")

    # --- Element Refs Fitting and Target Adjustment ---
    n_atom_types = 95
    element_refs_data = None
    original_targets_for_metric = targets.copy() # Backup original targets for metric calculation

    if FIT_ELEMENT_REFS:
        element_refs_data = fit_element_refs(structures, targets, n_atom_types)
        
        composition_matrix = np.zeros((len(structures), n_atom_types))
        for i, s in enumerate(structures):
            for el_key, count in s.composition.get_el_amt_dict().items():
                if isinstance(el_key, str): el_obj = Element(el_key)
                else: el_obj = el_key
                if el_obj.Z < n_atom_types:
                    composition_matrix[i, el_obj.Z] = count
        
        ref_energies_per_struct = composition_matrix @ element_refs_data
        targets = targets - ref_energies_per_struct # `targets` now becomes interaction energy
        print("\nSubtracted reference/offset energies. The model will now learn the residual.")

    # --- Model and Trainer Initialization ---
    print("\nInitializing model...")
    model = M3GNet(
        is_intensive=IS_INTENSIVE, 
        n_atom_types=n_atom_types, 
        embedding_type=EMBEDDING_TYPE,
        element_refs=element_refs_data
    )
    model.to(DEVICE)
    converter = model.graph_converter
    
    print("Initializing lazy layers with a dummy graph...")
    dummy_molecule = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dummy_graph = converter.convert(dummy_molecule)
    
    if TRAINING_TYPE == 'potential':
        dummy_targets = {'energy': torch.tensor(0.0), 'forces': torch.zeros(3, 3)}
        dummy_batch, _ = collate_potential_graphs([(dummy_graph, dummy_targets)])
        potential = Potential(model)
        potential.to(DEVICE)
        with torch.no_grad():
            potential(dummy_batch.to(DEVICE), compute_forces=False, compute_stress=False)
    else:
        dummy_batch, _ = collate_list_of_graphs([(dummy_graph, torch.tensor(0.0), torch.tensor(0.0))]) # Add dummy original_target
        with torch.no_grad():
            model(dummy_batch.to(DEVICE))

    print("\n--- Model Architecture (Initialized) ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print("----------------------------------------\n")

    print("Pre-processing structures into graphs... (This may take a moment)")
    graphs = [converter.convert(s) for s in tqdm(structures, desc="Converting")]
    
    # --- Data Splitting Logic ---
    if TRAINING_TYPE == 'potential':
        indices = list(range(len(graphs)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        train_targets_energy = targets[train_indices] # These are now interaction energies
        val_targets_energy = targets[val_indices]
        train_targets_forces = [targets_forces[i] for i in train_indices]
        val_targets_forces = [targets_forces[i] for i in val_indices]
        if targets_stresses:
            train_targets_stresses = [targets_stresses[i] for i in train_indices]
            val_targets_stresses = [targets_stresses[i] for i in val_indices]
        else:
            train_targets_stresses, val_targets_stresses = None, None
    else:
        # Split all necessary data together
        (
            train_graphs, val_graphs,
            train_targets, val_targets,
            train_original_targets, val_original_targets
        ) = train_test_split(
            graphs, targets, original_targets_for_metric,
            test_size=0.2, random_state=42
        )

    print(f"\nTraining set size: {len(train_graphs)}")
    print(f"Validation set size: {len(val_graphs)}")

    print("\nInitializing optimizer, scheduler, and trainer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = len(train_graphs) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1)

    # --- Select Trainer and Prepare Arguments ---
    if TRAINING_TYPE == 'potential':
        trainer = PotentialTrainer(potential=Potential(model), optimizer=optimizer, device=DEVICE)
        train_args = {
            "train_graphs": train_graphs, "train_energies": train_targets_energy, "train_forces": train_targets_forces, "train_stresses": train_targets_stresses,
            "val_graphs": val_graphs, "val_energies": val_targets_energy, "val_forces": val_targets_forces, "val_stresses": val_targets_stresses
        }
    else: # property training
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=DEVICE)
        train_args = {
            "train_graphs": train_graphs, 
            "train_targets": train_targets, # This is the interaction energy
            "train_original_targets": train_original_targets, # This is the original total energy
            "val_graphs": val_graphs, 
            "val_targets": val_targets,
            "val_original_targets": val_original_targets
        }

    print("\nSetting up callbacks...")
    checkpoint = ModelCheckpoint(save_dir=SAVE_DIR, monitor="val_loss", mode="min")
    callbacks = [checkpoint]
    if USE_EARLY_STOPPING:
        early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        callbacks.append(early_stopper)

    print("\n--- Starting Training ---")
    trainer.train(
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
        scheduler=scheduler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        **train_args
    )

    print("\n--- Training complete! ---")
    best_model_dir = os.path.join(SAVE_DIR, 'best_model')
    print(f"Best model saved to: {best_model_dir}")

if __name__ == "__main__":
    main()