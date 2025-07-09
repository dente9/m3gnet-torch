import unittest
import torch
import numpy as np
from pathlib import Path

# Import classes from your new module
from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter, MaterialGraphDataset, collate_fn

# Dependencies for creating test structures
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from ase import Atoms
from torch.utils.data import DataLoader

class TestGraphAndBatch(unittest.TestCase):
    """
    Comprehensive test suite for the m3gnet.graph module.
    """

    def setUp(self):
        """
        Set up common structures and converters for tests.
        This method is run before each test function.
        """
        # 1. A simple periodic structure (BCC Iron) using pymatgen
        lattice = Lattice.cubic(2.87)
        self.periodic_struct_pmg = Structure(
            lattice,
            ["Fe", "Fe"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )

        # 2. A simple non-periodic molecule (Water) using ASE
        self.molecule_ase = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.molecule_ase.set_cell([10, 10, 10]) # Set a large box to ensure non-periodicity is handled
        self.molecule_ase.pbc = [False, False, False]
        
        # 3. Converters
        self.converter_with_3body = RadiusCutoffGraphConverter(cutoff=5.0, threebody_cutoff=4.0)
        self.converter_no_3body = RadiusCutoffGraphConverter(cutoff=5.0, threebody_cutoff=None)

        # 4. Dummy target properties
        self.energy1 = -1.0
        self.forces1 = np.random.rand(2, 3).astype(np.float32)
        self.stress1 = np.random.rand(3, 3).astype(np.float32)

        self.energy2 = -2.0
        self.forces2 = np.random.rand(3, 3).astype(np.float32)
        self.stress2 = np.random.rand(3, 3).astype(np.float32)
    
    def test_converter_periodic_pmg(self):
        """Test conversion of a periodic pymatgen structure."""
        graph = self.converter_with_3body.convert(self.periodic_struct_pmg)
        
        self.assertIsInstance(graph, MaterialGraph)
        self.assertEqual(graph.n_atoms.item(), 2)
        self.assertTrue(graph.n_bonds.item() > 0)
        self.assertIsNotNone(graph.lattices)
        self.assertEqual(graph.lattices.shape, (1, 3, 3))
        
        # Check dtypes
        self.assertEqual(graph.atom_features.dtype, torch.long)
        self.assertEqual(graph.bond_features.dtype, torch.float32)
        self.assertEqual(graph.atom_positions.dtype, torch.float32)
        
        # Check shapes
        self.assertEqual(graph.atom_features.shape, (2, 1))
        self.assertEqual(graph.bond_atom_indices.shape, (graph.n_bonds.item(), 2))


    def test_converter_non_periodic_ase(self):
        """Test conversion of a non-periodic ASE Atoms object."""
        graph = self.converter_with_3body.convert(self.molecule_ase)

        self.assertIsInstance(graph, MaterialGraph)
        self.assertEqual(graph.n_atoms.item(), 3)
        self.assertTrue(graph.n_bonds.item() > 0)
        self.assertIsNone(graph.lattices, "Lattice should be None for non-periodic systems")
        self.assertEqual(graph.atom_features.shape, (3, 1))

    def test_three_body_logic(self):
        """Test if three-body interactions are correctly included or excluded."""
        # With three-body cutoff
        graph_with_3b = self.converter_with_3body.convert(self.periodic_struct_pmg)
        self.assertTrue(graph_with_3b.has_three_body)
        self.assertIsNotNone(graph_with_3b.triple_bond_indices)
        self.assertIsNotNone(graph_with_3b.triple_features)
        self.assertTrue(graph_with_3b.n_triples.item() > 0)
        self.assertEqual(graph_with_3b.triple_features.shape, (graph_with_3b.n_triples.item(), 1))

        # Without three-body cutoff
        graph_no_3b = self.converter_no_3body.convert(self.periodic_struct_pmg)
        self.assertFalse(graph_no_3b.has_three_body)
        self.assertIsNone(graph_no_3b.triple_bond_indices)
        self.assertIsNone(graph_no_3b.triple_features)
        self.assertEqual(graph_no_3b.n_triples.item(), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available for testing")
    def test_graph_to_device(self):
        """Test moving the graph to a CUDA device."""
        graph = self.converter_no_3body.convert(self.molecule_ase)
        cuda_graph = graph.to('cuda')
        self.assertEqual(cuda_graph.atom_features.device.type, 'cuda')
        self.assertEqual(cuda_graph.bond_features.device.type, 'cuda')
        self.assertIsNone(cuda_graph.lattices) # Check None is preserved

    def test_dataset_and_getitem(self):
        """Test the MaterialGraphDataset initialization and __getitem__ method."""
        dataset = MaterialGraphDataset(
            structures=[self.periodic_struct_pmg, self.molecule_ase],
            converter=self.converter_with_3body,
            energies=[self.energy1, self.energy2],
            forces=[self.forces1, self.forces2],
            stresses=[self.stress1, self.stress2]
        )
        
        self.assertEqual(len(dataset), 2)
        
        # Check first item
        graph, energy, forces, stress = dataset[0]
        self.assertIsInstance(graph, MaterialGraph)
        self.assertEqual(graph.n_atoms.item(), 2)
        self.assertTrue(torch.allclose(energy, torch.tensor(self.energy1)))
        self.assertTrue(torch.allclose(forces, torch.from_numpy(self.forces1)))
        self.assertTrue(torch.allclose(stress, torch.from_numpy(self.stress1)))
        
    def test_collate_fn(self):
        """Test the custom collate function for batching graphs."""
        # Create a batch of data
        graph1 = self.converter_with_3body.convert(self.periodic_struct_pmg)
        graph2 = self.converter_with_3body.convert(self.molecule_ase)
        
        batch = [
            (graph1, torch.tensor(self.energy1), torch.from_numpy(self.forces1), torch.from_numpy(self.stress1)),
            (graph2, torch.tensor(self.energy2), torch.from_numpy(self.forces2), torch.from_numpy(self.stress2))
        ]
        
        batched_graph, (batched_energy, batched_forces, batched_stress) = collate_fn(batch)
        
        # -- Validate batched graph --
        total_atoms = graph1.n_atoms.item() + graph2.n_atoms.item()
        total_bonds = graph1.n_bonds.item() + graph2.n_bonds.item()
        total_triples = graph1.n_triples.item() + graph2.n_triples.item()
        
        self.assertEqual(batched_graph.atom_features.shape[0], total_atoms)
        self.assertEqual(batched_graph.bond_features.shape[0], total_bonds)
        self.assertEqual(batched_graph.triple_features.shape[0], total_triples)
        
        # Crucial check: are indices correctly shifted?
        self.assertTrue(batched_graph.bond_atom_indices.max() < total_atoms)
        self.assertTrue(batched_graph.triple_bond_indices.max() < total_bonds)

        # Check that lattices are correctly batched (only one periodic struct in this case)
        self.assertEqual(batched_graph.lattices.shape[0], 1)
        
        # -- Validate batched targets --
        self.assertEqual(batched_energy.shape, (2,))
        self.assertTrue(torch.allclose(batched_energy, torch.tensor([self.energy1, self.energy2])))
        
        self.assertEqual(batched_forces.shape, (total_atoms, 3))
        self.assertEqual(batched_stress.shape, (2, 3, 3))

    def test_load_from_cif_files(self):
        """Test the end-to-end pipeline: loading CIF -> Dataset -> DataLoader -> Batch."""
        cif_dir = Path(__file__).parent.parent.parent / "data" / "cif_files"
        
        if not cif_dir.exists():
            self.skipTest(f"CIF directory not found at {cif_dir}, skipping this test.")

        cif_files = list(cif_dir.glob("*.cif"))
        if not cif_files:
            self.skipTest(f"No CIF files found in {cif_dir}, skipping this test.")
            
        # Use a small subset for a quick test
        structures_from_files = [Structure.from_file(f) for f in cif_files[:4]]
        
        # Mock energies for testing
        energies = [float(i) for i in range(len(structures_from_files))]

        dataset = MaterialGraphDataset(
            structures=structures_from_files,
            converter=self.converter_with_3body,
            energies=energies
        )
        
        # Batch size > 1 to test collation
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        # Get one batch
        batched_graph, (batched_energies,) = next(iter(dataloader))

        self.assertIsInstance(batched_graph, MaterialGraph)
        self.assertEqual(len(batched_energies), 2)
        
        # Check atom count in batch
        num_atoms_in_batch = batched_graph.atom_features.shape[0]
        expected_atoms = len(structures_from_files[0]) + len(structures_from_files[1])
        self.assertEqual(num_atoms_in_batch, expected_atoms)

        print("\nSuccessfully tested data loading and batching from CIF files.")


if __name__ == '__main__':
    unittest.main()