# m3gnet/train/tests/test_trainer.py

import unittest
import torch
import numpy as np
import os
import glob
import pandas as pd
from pymatgen.core import Structure

# Import everything we need to test
from m3gnet.models import M3GNet, Potential
from m3gnet.train import PropertyTrainer, PotentialTrainer, ModelCheckpoint, EarlyStopping

# Determine the root directory of the project to locate the data folder
# This makes the test runnable from any location
try:
    # Assumes the script is run from the project root (e.g., `python -m m3gnet.train.tests.test_trainer`)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
except NameError:
    # Fallback for interactive environments
    _PROJECT_ROOT = "."

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "cif_file")


class TestTrainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load structures from CIF files and generate dummy properties for training.
        This method is run once before all tests in this class.
        """
        print("\n--- Setting up for Trainer Tests ---")
        
        # 1. Load all structures from the data directory
        cls.cif_files = glob.glob(os.path.join(DATA_DIR, "*.cif"))
        if not cls.cif_files:
            raise FileNotFoundError(f"No CIF files found in {DATA_DIR}. Please ensure data is available.")
        
        cls.structures = [Structure.from_file(f) for f in cls.cif_files]
        print(f"Loaded {len(cls.structures)} structures for testing.")

        # 2. Generate dummy data for PropertyTrainer (e.g., band gaps)
        cls.property_targets = np.random.rand(len(cls.structures)) * 5  # Random "band gaps" from 0 to 5 eV
        
        # Create and save id_prop.csv as requested
        prop_df = pd.DataFrame({
            "filename": [os.path.basename(f) for f in cls.cif_files],
            "property": cls.property_targets
        })
        cls.id_prop_path = os.path.join(DATA_DIR, "id_prop.csv")
        prop_df.to_csv(cls.id_prop_path, index=False)
        print(f"Generated and saved dummy properties to {cls.id_prop_path}")

        # 3. Generate dummy data for PotentialTrainer (energy, forces, stresses)
        cls.potential_energies = np.random.rand(len(cls.structures)) * -10 # Random negative energies
        cls.potential_forces = [np.random.randn(len(s), 3) for s in cls.structures]
        cls.potential_stresses = [np.random.randn(3, 3) for _ in cls.structures]

        # 4. Set up a small subset for quick training tests
        cls.train_structs = cls.structures[:4]
        cls.val_structs = cls.structures[4:6]
        
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {cls.device}")

    def test_01_property_trainer(self):
        """Test the PropertyTrainer for a simple scalar property prediction."""
        print("\nTesting: PropertyTrainer...")
        
        # Prepare data for this specific test
        train_targets = self.property_targets[:4]
        val_targets = self.property_targets[4:6]

        # Initialize model, optimizer, and trainer
        model = M3GNet(is_intensive=True, n_atom_types=95) # Intensive property like band gap
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=self.device)

        # Run a short training session
        try:
            trainer.train(
                train_structures=self.train_structs,
                train_targets=train_targets,
                val_structures=self.val_structs,
                val_targets=val_targets,
                epochs=2,
                batch_size=2,
            )
            print("...PropertyTrainer ran successfully.")
        except Exception as e:
            self.fail(f"PropertyTrainer failed with an exception: {e}")

    def test_02_potential_trainer(self):
        """Test the PotentialTrainer for energy, forces, and stress fitting."""
        print("\nTesting: PotentialTrainer...")
        
        # Prepare data for this specific test
        train_energies, val_energies = self.potential_energies[:4], self.potential_energies[4:6]
        train_forces, val_forces = self.potential_forces[:4], self.potential_forces[4:6]
        train_stresses, val_stresses = self.potential_stresses[:4], self.potential_stresses[4:6]

        # Initialize model, potential, optimizer, and trainer
        model = M3GNet(is_intensive=False, n_atom_types=95) # Extensive property
        potential = Potential(model=model)
        optimizer = torch.optim.Adam(potential.parameters(), lr=1e-4)
        trainer = PotentialTrainer(potential=potential, optimizer=optimizer, device=self.device)

        # Run a short training session
        try:
            trainer.train(
                train_structures=self.train_structs,
                train_energies=train_energies,
                train_forces=train_forces,
                train_stresses=train_stresses,
                val_structures=self.val_structs,
                val_energies=val_energies,
                val_forces=val_forces,
                val_stresses=val_stresses,
                epochs=2,
                batch_size=2,
            )
            print("...PotentialTrainer ran successfully.")
        except Exception as e:
            self.fail(f"PotentialTrainer failed with an exception: {e}")
            
    def test_03_callbacks(self):
        """Test the ModelCheckpoint and EarlyStopping callbacks."""
        print("\nTesting: Callbacks (ModelCheckpoint and EarlyStopping)...")
        
        # --- Test ModelCheckpoint ---
        checkpoint_path = os.path.join(DATA_DIR, "test_model.pt")
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", mode="min", verbose=False)
        
        # Re-run a short property training
        model = M3GNet(is_intensive=True, n_atom_types=95)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=self.device)
        trainer.train(
            train_structures=self.train_structs, train_targets=self.property_targets[:4],
            val_structures=self.val_structs, val_targets=self.property_targets[4:6],
            epochs=2, batch_size=2, callbacks=[checkpoint]
        )
        self.assertTrue(os.path.exists(checkpoint_path), "ModelCheckpoint failed to create a model file.")
        os.remove(checkpoint_path) # Clean up
        print("...ModelCheckpoint OK.")

        # --- Test EarlyStopping ---
        # We simulate a non-improving validation loss to trigger early stopping
        class NonImprovingTrainer(PropertyTrainer):
            def _validate_one_epoch(self, loader):
                # Force validation loss to be constant
                return {"val_loss": 1.0}
        
        early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=False)
        model = M3GNet(is_intensive=True, n_atom_types=95)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ni_trainer = NonImprovingTrainer(model=model, optimizer=optimizer, device=self.device)
        
        # This training should stop after epoch 3 (1 initial + 2 patience)
        ni_trainer.train(
            train_structures=self.train_structs, train_targets=self.property_targets[:4],
            val_structures=self.val_structs, val_targets=self.property_targets[4:6],
            epochs=10, batch_size=2, callbacks=[early_stopper]
        )
        self.assertTrue(early_stopper.stop_training, "EarlyStopping failed to trigger.")
        print("...EarlyStopping OK.")

if __name__ == "__main__":
    unittest.main()