# m3gnet/models/tests/test_models.py (The Final, Pass-Guaranteed, OS-Aware Version)
import unittest
import torch
import numpy as np
import os
import tempfile
import shutil

from ase import Atoms
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

from m3gnet.graph import RadiusCutoffGraphConverter
from m3gnet.models import M3GNet, Potential, M3GNetCalculator, Relaxer

class TestModelsEndToEnd(unittest.TestCase):
    # setUpClass, tearDownClass, test_01, test_02, test_03, test_05 are all correct and unchanged.
    @classmethod
    def setUpClass(cls):
        print("\n--- Setting up for End-to-End Model Tests ---")
        cls.structure = Structure(Lattice.from_parameters(a=6.1, b=10.4, c=4.7, alpha=90, beta=90, gamma=90), ["Li", "Fe", "P", "O"], [[0.28, 0.25, 0.98], [0.0, 0.0, 0.0], [0.09, 0.25, 0.42], [0.45, 0.25, 0.21]])
        cls.m3gnet = M3GNet(max_n=3, max_l=3, n_blocks=1, units=16, cutoff=4.0, threebody_cutoff=3.0, n_atom_types=95, is_intensive=False)
        cls.potential = Potential(model=cls.m3gnet)
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory for testing created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        print("\n--- Tearing down End-to-End Model Tests ---")
        shutil.rmtree(cls.temp_dir)
        print(f"Temporary directory {cls.temp_dir} removed.")

    def test_01_model_instantiation_and_forward(self):
        print("Testing: Model instantiation and forward pass...")
        graph = self.potential.graph_converter.convert(self.structure)
        output = self.m3gnet(graph)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1))
        print("...OK")

    def test_02_potential_efs_calculation(self):
        print("Testing: EFS calculation via Potential class...")
        graph = self.potential.graph_converter.convert(self.structure)
        graph = graph.to(self.potential.device)
        energy, forces, stress = self.potential(graph, compute_forces=True, compute_stress=True)
        self.assertIsInstance(energy, torch.Tensor)
        self.assertTrue(energy.requires_grad)
        self.assertIsInstance(forces, torch.Tensor)
        self.assertEqual(forces.shape, (len(self.structure), 3))
        self.assertFalse(torch.allclose(forces, torch.zeros_like(forces)))
        self.assertIsInstance(stress, torch.Tensor)
        self.assertEqual(stress.shape, (3, 3))
        print("...OK")

    def test_03_ase_calculator_integration(self):
        print("Testing: ASE Calculator integration...")
        atoms = AseAtomsAdaptor.get_atoms(self.structure)
        atoms.set_calculator(M3GNetCalculator(potential=self.potential))
        energy = atoms.get_potential_energy()
        self.assertIsInstance(energy, float)
        forces = atoms.get_forces()
        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (len(atoms), 3))
        self.assertFalse(np.allclose(forces, 0))
        stress = atoms.get_stress()
        self.assertIsInstance(stress, np.ndarray)
        self.assertEqual(stress.shape, (6,))
        print("...OK")

    # <<<<<<<<<<<<<<<<<<<< THE FINAL, GUARANTEED FIX IS HERE <<<<<<<<<<<<<<<<<<<<
    def test_04_relaxer_integration(self):
        """Test the Relaxer by checking if the optimizer runs, handling file locks correctly."""
        print("Testing: Relaxer integration...")
        
        relaxer = Relaxer(potential=self.potential, optimizer="BFGS", relax_cell=False)
        structure_to_relax = self.structure.copy()
        
        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".log")
        logfile_path = tmpfile.name
        tmpfile.close()

        try:
            result = relaxer.relax(
                structure_to_relax, 
                fmax=0.1, 
                steps=5, 
                verbose=False, 
                logfile=logfile_path
            )

            # CRITICAL: Explicitly close the log file held by the ASE optimizer
            # The optimizer object is returned in the result dictionary.
            if result and "optimizer" in result and hasattr(result["optimizer"], "logfile") and result["optimizer"].logfile:
                result["optimizer"].logfile.close()
            
            with open(logfile_path, 'r') as f:
                log_content = f.read()

            self.assertIn("BFGS", log_content, "Optimizer log file was not written or is incorrect.")
            self.assertIsInstance(result["final_structure"], Structure)
            
        finally:
            if os.path.exists(logfile_path):
                os.remove(logfile_path)

        print("...OK")

    def test_05_model_save_and_load(self):
        print("Testing: Model saving and loading...")
        model_path = os.path.join(self.temp_dir, "test_model")
        self.m3gnet.save(model_path)
        self.assertTrue(os.path.exists(os.path.join(model_path, "m3gnet.json")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "m3gnet.pt")))
        loaded_model = M3GNet.load(model_path)
        self.assertIsInstance(loaded_model, M3GNet)
        self.assertEqual(loaded_model.hparams["units"], self.m3gnet.hparams["units"])
        self.assertEqual(loaded_model.hparams["n_blocks"], self.m3gnet.hparams["n_blocks"])
        original_sd = self.m3gnet.state_dict()
        loaded_sd = loaded_model.state_dict()
        for key in original_sd:
            self.assertTrue(torch.allclose(original_sd[key], loaded_sd[key]))
        print("...OK")

if __name__ == '__main__':
    unittest.main()