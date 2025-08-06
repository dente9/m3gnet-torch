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
    from m3gnet.train import (
        PropertyTrainer,
        PotentialTrainer,
        ModelCheckpoint,
        EarlyStopping,
    )
    from m3gnet.graph import RadiusCutoffGraphConverter
    from m3gnet.graph.batch import collate_list_of_graphs, collate_potential_graphs
except ImportError:
    # This allows the script to be run from the root directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from m3gnet.models import M3GNet, Potential
    from m3gnet.train import (
        PropertyTrainer,
        PotentialTrainer,
        ModelCheckpoint,
        EarlyStopping,
    )
    from m3gnet.graph import RadiusCutoffGraphConverter
    from m3gnet.graph.batch import collate_list_of_graphs, collate_potential_graphs

# --- [ 2. CONFIGURATION ZONE ] ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 96
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False
USE_EARLY_STOPPING = True
PATIENCE = 25

# --- Task-specific settings ---
# 'property': Loads from pre-split train/ and val/ folders.
# 'potential': Loads from a single JSON and splits it in memory.
TRAINING_TYPE = "property"

if TRAINING_TYPE == "property":
    PROPERTY_DATA_ROOT = os.path.join(SCRIPT_DIR, "data", "split_s")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor")
    IS_INTENSIVE = False
    FIT_ELEMENT_REFS = True
    USE_NORMALIZATION = True
    EMBEDDING_TYPE = "attention"

elif TRAINING_TYPE == "potential":
    DATA_PATH = os.path.join(SCRIPT_DIR, "data", "efs_data")
    JSON_PATH = os.path.join(DATA_PATH, "efs.json")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "potential_predictor")
    IS_INTENSIVE = False
    FIT_ELEMENT_REFS = True
    USE_NORMALIZATION = True
    EMBEDDING_TYPE = "attention"
else:
    raise ValueError(
        f"Unknown TRAINING_TYPE: '{TRAINING_TYPE}'. Must be 'property' or 'potential'."
    )


# --- [ 3. HELPER FUNCTIONS ] ---


def load_structures_and_targets(data_path: str, csv_filename: str = "id_prop.csv"):
    """
    Robustly loads Structure objects and their corresponding targets from a directory.
    """
    csv_path = os.path.join(data_path, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Required CSV file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    # Robustly handle column names
    filename_col, target_col = df.columns[0], df.columns[1]

    df["filepath"] = df[filename_col].apply(
        lambda fn: os.path.join(
            data_path, f"{fn}.cif" if not str(fn).lower().endswith(".cif") else str(fn)
        )
    )

    structures, valid_indices = [], []
    print(f"Loading structures from {os.path.basename(data_path)}...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading files"):
        try:
            structures.append(Structure.from_file(row["filepath"]))
            valid_indices.append(index)
        except Exception as e:
            print(f"\nWarning: Skipping file {row['filepath']} due to error: {e}")

    targets = df.iloc[valid_indices][target_col].values
    return structures, targets


def fit_element_refs(
    structures: list, energies: np.ndarray, n_atom_types: int
) -> np.ndarray:
    """
    Fits elemental reference energies using a list of Structure objects.
    This is the original, most straightforward implementation.
    """
    print("Fitting elemental reference energies from Structure objects...")
    feature_matrix = np.zeros((len(structures), n_atom_types))
    for i, s in enumerate(structures):
        for el_key, count in s.composition.get_el_amt_dict().items():
            el_obj = Element(el_key)
            if el_obj.Z < n_atom_types:
                feature_matrix[i, el_obj.Z] = count
    try:
        element_refs = np.linalg.solve(
            feature_matrix.T @ feature_matrix, feature_matrix.T @ energies
        )
        print("Elemental reference energies fitted successfully using np.linalg.solve.")
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Using pseudo-inverse (pinv) for fitting.")
        element_refs = np.linalg.pinv(feature_matrix) @ energies

    final_refs = np.zeros(n_atom_types)
    limit = min(len(element_refs), n_atom_types)
    final_refs[:limit] = element_refs[:limit]

    print("Fitted non-zero elemental references (eV/atom):")
    for i, ref in enumerate(final_refs):
        if abs(ref) > 1e-6:
            print(f"  - {Element.from_Z(i).symbol} (Z={i}): {ref:.4f}")
    return final_refs


# --- [ 4. MAIN FUNCTION (ROBUST LOGIC VERSION) ] ---
def main():
    """Main training function."""

    global NUM_WORKERS
    if platform.system().lower() != "windows":
        try:
            num_cores = len(os.sched_getaffinity(0))
            NUM_WORKERS = min(12, num_cores)
        except AttributeError:
            num_cores = os.cpu_count() or 1
            NUM_WORKERS = min(12, num_cores)

    # --- Configuration Logging ---
    config = {
        "Training Type": TRAINING_TYPE,
        "Device": DEVICE,
        "Embedding Type": EMBEDDING_TYPE,
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Learning Rate": LEARNING_RATE,
        "Is Intensive": IS_INTENSIVE,
        "Fit Element Refs": FIT_ELEMENT_REFS,
        "Use Normalization": USE_NORMALIZATION,
        "Early Stopping": "Enabled" if USE_EARLY_STOPPING else "Disabled",
        "Patience": PATIENCE if USE_EARLY_STOPPING else "N/A",
        "Num Workers": NUM_WORKERS,
        "Pin Memory": PIN_MEMORY,
        "Save Directory": os.path.abspath(SAVE_DIR),
    }
    if TRAINING_TYPE == "property":
        config["Property Data Root"] = os.path.abspath(PROPERTY_DATA_ROOT)
    elif TRAINING_TYPE == "potential":
        config["Potential Data JSON"] = os.path.abspath(JSON_PATH)

    print("\n--- M3GNet Training Configuration ---")
    for key, value in config.items():
        print(f"{key:<22}: {value}")
    print("-------------------------------------\n")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Data Loading (Step 1: Load all structures into memory) ---
    if TRAINING_TYPE == "property":
        train_data_path = os.path.join(PROPERTY_DATA_ROOT, "train")
        val_data_path = os.path.join(PROPERTY_DATA_ROOT, "val")
        train_structures, train_targets_total = load_structures_and_targets(
            train_data_path
        )
        val_structures, val_targets_total = load_structures_and_targets(val_data_path)

    elif TRAINING_TYPE == "potential":
        with open(JSON_PATH, "r") as f:
            data_list = json.load(f)
        all_structures = [Structure.from_dict(d["structure"]) for d in data_list]
        all_targets = np.array([d["energy"] for d in data_list])
        all_forces = [np.array(d["forces"]) for d in data_list]
        all_stresses = (
            [np.array(d["stress"]) for d in data_list]
            if "stress" in data_list[0]
            else None
        )

        indices = list(range(len(all_structures)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )

        train_structures = [all_structures[i] for i in train_indices]
        val_structures = [all_structures[i] for i in val_indices]
        train_targets_total = all_targets[train_indices]
        val_targets_total = all_targets[val_indices]
        train_forces = [all_forces[i] for i in train_indices]
        val_forces = [all_forces[i] for i in val_indices]
        train_stresses = (
            [all_stresses[i] for i in train_indices] if all_stresses else None
        )
        val_stresses = [all_stresses[i] for i in val_indices] if all_stresses else None

    print(f"\nTraining set size: {len(train_structures)}")
    print(f"Validation set size: {len(val_structures)}")

    # --- Stats Calculation ---
    # This happens before model creation, ensuring robust initialization
    element_refs_data, mean_interaction, std_interaction = None, 0.0, 1.0
    all_structures = train_structures + val_structures
    all_targets_total = np.concatenate([train_targets_total, val_targets_total])

    if FIT_ELEMENT_REFS:
        element_refs_data = fit_element_refs(train_structures, train_targets_total, 95)

        composition_matrix = np.zeros((len(all_structures), 95))
        for i, s in enumerate(all_structures):
            for el, count in s.composition.get_el_amt_dict().items():
                if Element(el).Z < 95:
                    composition_matrix[i, Element(el).Z] = count

        ref_energies_per_struct = composition_matrix @ element_refs_data
        interaction_energies = all_targets_total - ref_energies_per_struct

        if USE_NORMALIZATION:
            train_interaction_energies = interaction_energies[: len(train_structures)]
            mean_interaction = np.mean(train_interaction_energies)
            std_interaction = np.std(train_interaction_energies)
            if std_interaction < 1e-6:
                std_interaction = 1.0
            print(
                f"\nNormalization stats (on train set): Mean={mean_interaction:.4f}, Std={std_interaction:.4f}"
            )
            targets_for_training = (
                interaction_energies - mean_interaction
            ) / std_interaction
        else:
            targets_for_training = interaction_energies
    else:
        targets_for_training = all_targets_total

    # --- Model Initialization (using final, correct stats) ---
    print("\nInitializing model with all statistics...")
    model = M3GNet(
        is_intensive=IS_INTENSIVE,
        n_atom_types=95,
        embedding_type=EMBEDDING_TYPE,
        element_refs=element_refs_data,
        mean=mean_interaction,
        std=std_interaction,
    )
    model.to(DEVICE)
    converter = model.graph_converter

    # --- Graph Conversion (Step 2: Convert all structures to graphs) ---
    print("\nPre-processing structures into graphs...")
    train_graphs = [
        converter.convert(s) for s in tqdm(train_structures, desc="Converting train")
    ]
    val_graphs = [
        converter.convert(s) for s in tqdm(val_structures, desc="Converting val")
    ]

    # --- Prepare Trainer Arguments ---
    train_indices_range = slice(0, len(train_structures))
    val_indices_range = slice(len(train_structures), len(all_structures))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if TRAINING_TYPE == "potential":
        trainer = PotentialTrainer(
            potential=Potential(model), optimizer=optimizer, device=DEVICE
        )
        train_args = {
            "train_graphs": train_graphs,
            "train_energies": targets_for_training[train_indices_range],
            "train_forces": train_forces,
            "train_stresses": train_stresses,
            "val_graphs": val_graphs,
            "val_energies": targets_for_training[val_indices_range],
            "val_forces": val_forces,
            "val_stresses": val_stresses,
        }
    else:  # property
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=DEVICE)
        train_args = {
            "train_graphs": train_graphs,
            "train_targets": targets_for_training[train_indices_range],
            "train_original_targets": train_targets_total,
            "val_graphs": val_graphs,
            "val_targets": targets_for_training[val_indices_range],
            "val_original_targets": val_targets_total,
        }

    # --- Initialize Lazy Layers ---
    print("\nInitializing lazy layers with a dummy graph...")
    dummy_molecule = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dummy_graph = converter.convert(dummy_molecule)
    if TRAINING_TYPE == "potential":
        dummy_batch, _ = collate_potential_graphs(
            [(dummy_graph, {"energy": torch.tensor(0.0), "forces": torch.zeros(3, 3)})]
        )
        potential = Potential(model)
        potential.to(DEVICE)
        with torch.no_grad():
            potential(
                dummy_batch.to(DEVICE), compute_forces=False, compute_stress=False
            )
    else:
        dummy_batch, _ = collate_list_of_graphs(
            [(dummy_graph, torch.tensor(0.0), torch.tensor(0.0))]
        )
        with torch.no_grad():
            model(dummy_batch.to(DEVICE))

    print(
        f"\n--- Model Architecture ---\n{model}\nTotal Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n--------------------------\n"
    )

    # --- Setup Callbacks and Scheduler ---
    callbacks = [ModelCheckpoint(save_dir=SAVE_DIR, monitor="val_loss", mode="min")]
    if USE_EARLY_STOPPING:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        )

    steps_per_epoch = len(train_graphs) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1,
        eta_min=LEARNING_RATE * 0.01,
    )

    # --- Start Training ---
    print("\n--- Starting Training ---")
    trainer.train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        scheduler=scheduler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        **train_args,
    )

    print(
        f"\n--- Training complete! Best model saved to: {os.path.join(SAVE_DIR, 'best_model')} ---"
    )


if __name__ == "__main__":
    main()
