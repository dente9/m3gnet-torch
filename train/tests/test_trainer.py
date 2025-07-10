# m3gnet/train/tests/test_trainer.py (Final Import-Fixed Version)

import unittest
import torch
import numpy as np
import os
import glob
import pandas as pd
from pymatgen.core import Structure
import tempfile
import shutil
import unittest.mock
from tqdm import tqdm

# <<<<<<<<<<<<<<<<<<<< THE FIX IS HERE <<<<<<<<<<<<<<<<<<<<
# Import components from their correct modules
from m3gnet.models import M3GNet, Potential
from m3gnet.graph import RadiusCutoffGraphConverter # Import from graph module
from m3gnet.train import PropertyTrainer, PotentialTrainer, ModelCheckpoint, EarlyStopping

M3GNET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(M3GNET_ROOT, "data", "cif_file")

class TestTrainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load structures, pre-process them into graphs, and generate dummy properties.
        """
        print("\n--- Setting up for Trainer Tests ---")
        
        cls.cif_files = glob.glob(os.path.join(DATA_DIR, "*.cif"))
        if not cls.cif_files:
            raise FileNotFoundError(f"No CIF files found in {DATA_DIR}.")
        
        structures = [Structure.from_file(f) for f in cls.cif_files]
        print(f"Loaded {len(structures)} structures from {DATA_DIR}")

        cls.converter = RadiusCutoffGraphConverter(cutoff=5.0)
        print("Pre-processing structures into graphs...")
        cls.graphs = [cls.converter.convert(s) for s in tqdm(structures, desc="Converting")]
        
        cls.property_targets = np.random.rand(len(cls.graphs)) * 5
        prop_df = pd.DataFrame({
            "filename": [os.path.basename(f) for f in cls.cif_files],
            "property": cls.property_targets
        })
        cls.id_prop_path = os.path.join(DATA_DIR, "id_prop.csv")
        prop_df.to_csv(cls.id_prop_path, index=False)
        print(f"Generated and saved dummy properties to {cls.id_prop_path}")

        cls.potential_energies = np.random.rand(len(cls.graphs)) * -10
        cls.potential_forces = [np.random.randn(len(s), 3) for s in structures]
        cls.potential_stresses = [np.random.randn(3, 3) for _ in cls.graphs]

        cls.train_graphs = cls.graphs[:4]
        cls.val_graphs = cls.graphs[4:6]
        
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {cls.device}")

    def test_01_property_trainer(self):
        """Test the PropertyTrainer with pre-processed graphs."""
        print("\nTesting: PropertyTrainer...")
        train_targets = self.property_targets[:4]
        val_targets = self.property_targets[4:6]

        model = M3GNet(is_intensive=True, n_atom_types=95)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = PropertyTrainer(model=model, optimizer=optimizer, device=self.device)

        try:
            trainer.train(
                train_graphs=self.train_graphs, train_targets=train_targets,
                val_graphs=self.val_graphs, val_targets=val_targets,
                epochs=2, batch_size=2,
            )
            print("...PropertyTrainer ran successfully.")
        except Exception as e:
            self.fail(f"PropertyTrainer failed with an exception: {e}")

    def test_02_potential_trainer(self):
        """Test the PotentialTrainer with pre-processed graphs."""
        print("\nTesting: PotentialTrainer...")
        train_energies, val_energies = self.potential_energies[:4], self.potential_energies[4:6]
        train_forces, val_forces = self.potential_forces[:4], self.potential_forces[4:6]
        train_stresses, val_stresses = self.potential_stresses[:4], self.potential_stresses[4:6]

        model = M3GNet(is_intensive=False, n_atom_types=95)
        potential = Potential(model=model)
        optimizer = torch.optim.Adam(potential.parameters(), lr=1e-4)
        trainer = PotentialTrainer(potential=potential, optimizer=optimizer, device=self.device)

        try:
            trainer.train(
                train_graphs=self.train_graphs, train_energies=train_energies,
                train_forces=train_forces, train_stresses=train_stresses,
                val_graphs=self.val_graphs, val_energies=val_energies,
                val_forces=val_forces, val_stresses=val_stresses,
                epochs=2, batch_size=2,
            )
            print("...PotentialTrainer ran successfully.")
        except Exception as e:
            self.fail(f"PotentialTrainer failed with an exception: {e}")
            
    def test_03_callbacks(self):
        """Test callbacks with the new trainer signatures."""
        print("\nTesting: Callbacks (ModelCheckpoint and EarlyStopping)...")
        
        temp_dir = tempfile.mkdtemp()
        try:
            with self.subTest(callback="ModelCheckpoint"), unittest.mock.patch('builtins.print'):
                checkpoint = ModelCheckpoint(save_dir=temp_dir, monitor="val_loss", mode="min", verbose=True)
                model = M3GNet(is_intensive=True, n_atom_types=95)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                trainer = PropertyTrainer(model=model, optimizer=optimizer, device=self.device)
                
                trainer.train(
                    train_graphs=self.train_graphs, train_targets=self.property_targets[:4],
                    val_graphs=self.val_graphs, val_targets=self.property_targets[4:6],
                    epochs=2, batch_size=2, callbacks=[checkpoint]
                )
                self.assertTrue(os.path.exists(os.path.join(temp_dir, "best_model.pt")))
                self.assertTrue(os.path.exists(os.path.join(temp_dir, "last_model.pt")))
            print("...ModelCheckpoint OK.")

            with self.subTest(callback="EarlyStopping"), unittest.mock.patch('builtins.print'):
                class NonImprovingTrainer(PropertyTrainer):
                    def _validate_one_epoch(self, loader):
                        return {"val_loss": 1.0}
                
                early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=True)
                model = M3GNet(is_intensive=True, n_atom_types=95)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                ni_trainer = NonImprovingTrainer(model=model, optimizer=optimizer, device=self.device)
                
                ni_trainer.train(
                    train_graphs=self.train_graphs, train_targets=self.property_targets[:4],
                    val_graphs=self.val_graphs, val_targets=self.property_targets[4:6],
                    epochs=10, batch_size=2, callbacks=[early_stopper]
                )
                self.assertTrue(early_stopper.stop_training)
            print("...EarlyStopping OK.")
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()