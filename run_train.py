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
# (Configuration remains the same)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False
USE_EARLY_STOPPING = True
PATIENCE = 25
TRAINING_TYPE = "property"
if TRAINING_TYPE == "property":
    PROPERTY_DATA_ROOT = os.path.join(SCRIPT_DIR, "data", "pre_split")
    SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models", "property_predictor")
    TARGET_COLUMN = "property"
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
    raise ValueError(f"Unknown TRAINING_TYPE: {TRAINING_TYPE}")


# --- [ 3. HELPER FUNCTIONS ] ---


# CORRECTED HELPER FUNCTION
def load_from_df_and_path(data_path: str, csv_filename: str):
    """
    Loads structures and targets. Assumes 1st col is filename, 2nd is target property.
    Automatically adds '.cif' extension if it's missing from the filename in the CSV.
    """
    csv_path = os.path.join(data_path, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Required CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if len(df.columns) < 2:
        raise ValueError(
            f"CSV at {csv_path} must contain at least two columns: one for filenames and one for properties."
        )

    filename_col = df.columns[0]
    target_col = df.columns[1]

    print(f"  - Reading filenames from column: '{filename_col}'")
    print(f"  - Reading targets from column: '{target_col}'")

    # --- THIS IS THE FIX ---
    # Smartly add .cif extension. Handles cases where .cif is present or missing.
    df["filepath"] = df[filename_col].apply(
        lambda fn: os.path.join(
            data_path, f"{fn}.cif" if not str(fn).lower().endswith(".cif") else str(fn)
        )
    )
    # --- END OF FIX ---

    print(f"Loading structures from {data_path}...")
    structures = []
    valid_rows_mask = pd.Series([True] * len(df), index=df.index)

    for index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Loading from {os.path.basename(data_path)}"
    ):
        filepath = row["filepath"]
        try:
            structures.append(Structure.from_file(filepath))
        except Exception as e:
            # Now the error should only happen for genuinely missing or corrupted files
            print(
                f"\nWarning: Error loading structure file: {filepath}. Skipping this entry. Error: {e}"
            )
            valid_rows_mask[index] = False

    df_filtered = df[valid_rows_mask]
    targets = df_filtered[target_col].values

    if len(structures) != len(targets):
        raise RuntimeError(
            "Mismatch between number of loaded structures and targets after handling errors. Please check data."
        )

    return structures, targets


def fit_element_refs(
    structures: list, energies: np.ndarray, n_atom_types: int
) -> np.ndarray:
    """(This function remains unchanged)"""
    print("Fitting elemental reference energies...")
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


# --- [ 4. MAIN FUNCTION (SIMPLIFIED) ] ---
def main():
    """Main training function."""

    # (The rest of the script is unchanged as the fix was in the helper function)
    global NUM_WORKERS
    if platform.system().lower() != "windows":
        try:
            num_cores = len(os.sched_getaffinity(0))
            NUM_WORKERS = min(12, num_cores)
        except AttributeError:
            num_cores = os.cpu_count() or 1
            NUM_WORKERS = min(12, num_cores)
    config = {
        "Training Type": TRAINING_TYPE,
        "Device": DEVICE,
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Save Directory": os.path.abspath(SAVE_DIR),
    }
    print("\n--- M3GNet Training Configuration ---")
    [print(f"{k:<20}: {v}") for k, v in config.items()]
    print("-------------------------------------\n")
    os.makedirs(SAVE_DIR, exist_ok=True)

    if TRAINING_TYPE == "potential":
        print("Loading EFS data from single JSON file...")
        with open(JSON_PATH, "r") as f:
            data_list = json.load(f)
        structures = [Structure.from_dict(d["structure"]) for d in data_list]
        targets_total_energy = np.array([d["energy"] for d in data_list])
        targets_forces = [np.array(d["forces"]) for d in data_list]
        targets_stresses = (
            [np.array(d["stress"]) for d in data_list]
            if "stress" in data_list[0]
            else None
        )

        print("\nSplitting EFS data into training and validation sets...")
        indices = list(range(len(structures)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )

        train_structures = [structures[i] for i in train_indices]
        val_structures = [structures[i] for i in val_indices]
        train_targets_total = targets_total_energy[train_indices]
        val_targets_total = targets_total_energy[val_indices]

    elif TRAINING_TYPE == "property":
        train_data_path = os.path.join(PROPERTY_DATA_ROOT, "train")
        val_data_path = os.path.join(PROPERTY_DATA_ROOT, "val")

        if not os.path.isdir(train_data_path) or not os.path.isdir(val_data_path):
            raise FileNotFoundError(
                f"Required 'train' and 'val' subdirectories not found in '{PROPERTY_DATA_ROOT}'."
            )

        print(
            f"Loading data from pre-split directories: '{train_data_path}' and '{val_data_path}'"
        )
        train_structures, train_targets_total = load_from_df_and_path(
            train_data_path, "id_prop.csv"
        )
        val_structures, val_targets_total = load_from_df_and_path(
            val_data_path, "id_prop.csv"
        )

        structures = train_structures + val_structures
        targets_total_energy = np.concatenate([train_targets_total, val_targets_total])
        train_indices = list(range(len(train_structures)))
        val_indices = list(range(len(train_structures), len(structures)))

    print(f"\nTraining set size: {len(train_structures)}")
    print(f"Validation set size: {len(val_structures)}")

    n_atom_types = 95
    element_refs_data = None
    mean_interaction = 0.0
    std_interaction = 1.0
    train_original_targets = train_targets_total
    val_original_targets = val_targets_total
    if FIT_ELEMENT_REFS:
        element_refs_data = fit_element_refs(
            train_structures, train_targets_total, n_atom_types
        )
        composition_matrix = np.zeros((len(structures), n_atom_types))
        for i, s in enumerate(structures):
            for el, count in s.composition.get_el_amt_dict().items():
                if Element(el).Z < n_atom_types:
                    composition_matrix[i, Element(el).Z] = count
        ref_energies_per_struct = composition_matrix @ element_refs_data
        interaction_energies = targets_total_energy - ref_energies_per_struct
        if USE_NORMALIZATION:
            train_interaction_energies = interaction_energies[train_indices]
            mean_interaction = np.mean(train_interaction_energies)
            std_interaction = np.std(train_interaction_energies)
            if std_interaction < 1e-6:
                print(
                    "Warning: Std dev of interaction energy is near zero. Normalization skipped."
                )
                std_interaction = 1.0
                mean_interaction = 0.0
            print(
                f"\nNormalization stats (on train set): Mean={mean_interaction:.4f}, Std={std_interaction:.4f}"
            )
            targets_for_training = (
                interaction_energies - mean_interaction
            ) / std_interaction
        else:
            targets_for_training = interaction_energies
    else:
        targets_for_training = targets_total_energy

    if TRAINING_TYPE == "property":
        train_targets = targets_for_training[train_indices]
        val_targets = targets_for_training[val_indices]
    elif TRAINING_TYPE == "potential":
        train_targets_energy = targets_for_training[train_indices]
        val_targets_energy = targets_for_training[val_indices]
        train_targets_forces = [targets_forces[i] for i in train_indices]
        val_targets_forces = [targets_forces[i] for i in val_indices]
        if "targets_stresses" in locals() and targets_stresses:
            train_targets_stresses = [targets_stresses[i] for i in train_indices]
            val_targets_stresses = [targets_stresses[i] for i in val_indices]
        else:
            train_targets_stresses, val_targets_stresses = None, None

    print("\nInitializing model...")
    model = M3GNet(
        is_intensive=IS_INTENSIVE,
        n_atom_types=n_atom_types,
        embedding_type=EMBEDDING_TYPE,
        element_refs=element_refs_data,
        mean=mean_interaction,
        std=std_interaction,
    )
    model.to(DEVICE)
    converter = model.graph_converter
    print("Initializing lazy layers...")
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

    print("Pre-processing structures into graphs...")
    all_graphs = [converter.convert(s) for s in tqdm(structures, desc="Converting")]
    train_graphs = [all_graphs[i] for i in train_indices]
    val_graphs = [all_graphs[i] for i in val_indices]

    print("\nInitializing optimizer, scheduler, and trainer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = len(train_graphs) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1,
        eta_min=LEARNING_RATE * 0.01,
    )

    if TRAINING_TYPE == "potential":
        trainer = PotentialTrainer(
            potential=Potential(model), optimizer=optimizer, device=DEVICE
        )
        train_args = {
            "train_graphs": train_graphs,
            "train_energies": train_targets_energy,
            "train_forces": train_targets_forces,
            "train_stresses": train_targets_stresses,
            "val_graphs": val_graphs,
            "val_energies": val_targets_energy,
            "val_forces": val_targets_forces,
            "val_stresses": val_targets_stresses,
        }
    else:
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=DEVICE)
        train_args = {
            "train_graphs": train_graphs,
            "train_targets": train_targets,
            "train_original_targets": train_original_targets,
            "val_graphs": val_graphs,
            "val_targets": val_targets,
            "val_original_targets": val_original_targets,
        }

    print("\nSetting up callbacks...")
    checkpoint = ModelCheckpoint(save_dir=SAVE_DIR, monitor="val_loss", mode="min")
    callbacks = [checkpoint]
    if USE_EARLY_STOPPING:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        )

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
