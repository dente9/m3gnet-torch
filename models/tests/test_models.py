import unittest
import torch
import numpy as np
import os
import tempfile
import shutil

# ASE and Pymatgen for creating test structures
from ase import Atoms
from pymatgen.core import Structure, Lattice

# Import the final, real modules
from m3gnet.graph import RadiusCutoffGraphConverter
from m3gnet.models import M3GNet, Potential, M3GNetCalculator, Relaxer

class TestModelsEndToEnd(unittest.TestCase):
    """
    End-to-end test suite for the m3gnet.models module, using real layers.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple structure and a real M3GNet model."""
        print("\n--- Setting up for End-to-End Model Tests ---")
        
        # 1. Create a test structure (e.g., LiFePO4 unit cell)
        # Using a more complex structure to ensure all graph logic is triggered
        cls.structure = Structure(
            Lattice.from_parameters(a=6.1, b=10.4, c=4.7, alpha=90, beta=90, gamma=90),
            ["Li", "Fe", "P", "O"],
            [[0.28, 0.25, 0.98], [0.0, 0.0, 0.0], [0.09, 0.25, 0.42], [0.45, 0.25, 0.21]]
        )
        
        # 2. Instantiate a real M3GNet model with a small configuration for fast testing
        cls.m3gnet = M3GNet(
            max_n=3, max_l=3, n_blocks=1, units=16,
            cutoff=4.0, threebody_cutoff=3.0,
            n_atom_types=95, # Ensure it's large enough for Fe, P, O, Li
            is_intensive=False # Energy is an extensive property
        )
        
        # 3. Create a Potential object
        cls.potential = Potential(model=cls.m3gnet)

        # 4. Create a temporary directory for saving/loading tests
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory for testing created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests are done."""
        print("\n--- Tearing down End-to-End Model Tests ---")
        shutil.rmtree(cls.temp_dir)
        print(f"Temporary directory {cls.temp_dir} removed.")

    def test_01_model_instantiation_and_forward(self):
        """Test if the real M3GNet model can be instantiated and run a forward pass."""
        print("Testing: Model instantiation and forward pass...")
        
        graph = self.potential.graph_converter.convert(self.structure)
        output = self.m3gnet(graph)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1)) # Extensive model with ReduceReadOut("sum") gives (1,1)
        print("...OK")

    def test_02_potential_efs_calculation(self):
        """Test the Potential class's ability to compute real energy, forces, and stress."""
        print("Testing: EFS calculation via Potential class...")
        
        graph = self.potential.graph_converter.convert(self.structure)
        
        # We need to move the graph to the model's device
        graph = graph.to(self.potential.device)
        
        energy, forces, stress = self.potential(graph, compute_forces=True, compute_stress=True)
        
        # Check energy
        self.assertIsInstance(energy, torch.Tensor)
        self.assertTrue(energy.requires_grad) # Should have a grad_fn
        
        # Check forces
        self.assertIsInstance(forces, torch.Tensor)
        self.assertEqual(forces.shape, (len(self.structure), 3))
        # A non-trivial model should produce non-zero forces on a random structure
        self.assertFalse(torch.allclose(forces, torch.zeros_like(forces)))
        
        # Check stress
        self.assertIsInstance(stress, torch.Tensor)
        self.assertEqual(stress.shape, (3, 3))
        # Stress might be close to zero for a random model, but shouldn't be exactly zero
        # unless the gradients are null, which we test with forces.
        
        print("...OK")

    def test_03_ase_calculator_integration(self):
        """Test the M3GNetCalculator with a real model."""
        print("Testing: ASE Calculator integration...")
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(self.structure)
        calculator = M3GNetCalculator(potential=self.potential)
        atoms.set_calculator(calculator)
        
        # Get energy
        energy = atoms.get_potential_energy()
        self.assertIsInstance(energy, float)
        
        # Get forces
        forces = atoms.get_forces()
        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (len(atoms), 3))
        self.assertFalse(np.allclose(forces, 0))
        
        # Get stress
        stress = atoms.get_stress()
        self.assertIsInstance(stress, np.ndarray)
        self.assertEqual(stress.shape, (6,)) # ASE Voigt notation
        
        print("...OK")

    def test_04_relaxer_integration(self):
        """Test the Relaxer class with a real model to see if atoms move."""
        print("Testing: Relaxer integration...")
        
        relaxer = Relaxer(potential=self.potential, optimizer="BFGS", relax_cell=False)
        
        structure_to_relax = self.structure.copy()
        initial_pos = structure_to_relax.cart_coords.copy()
        
        # Run a short relaxation. With real forces, the atoms should move.
        result = relaxer.relax(structure_to_relax, fmax=100.0, steps=2, verbose=False)
        
        self.assertIn("final_structure", result)
        final_pos = result["final_structure"].cart_coords
        
        # Check that atom positions have changed after relaxation steps
        self.assertFalse(np.allclose(initial_pos, final_pos))
        print("...OK")

    def test_05_model_save_and_load(self):
        """Test saving a model and loading it back."""
        print("Testing: Model saving and loading...")
        
        # Save the model
        model_path = os.path.join(self.temp_dir, "test_model")
        self.m3gnet.save(model_path)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(model_path, "m3gnet.json")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "m3gnet.pt")))
        
        # Load the model back
        loaded_model = M3GNet.load(model_path)
        
        self.assertIsInstance(loaded_model, M3GNet)
        
        # Verify that the loaded model has the same configuration
        self.assertEqual(loaded_model.hparams["units"], self.m3gnet.hparams["units"])
        self.assertEqual(loaded_model.hparams["n_blocks"], self.m3gnet.hparams["n_blocks"])
        
        # Verify that the state dicts are the same
        original_sd = self.m3gnet.state_dict()
        loaded_sd = loaded_model.state_dict()
        for key in original_sd:
            self.assertTrue(torch.allclose(original_sd[key], loaded_sd[key]))
            
        print("...OK")

if __name__ == '__main__':
    unittest.main()