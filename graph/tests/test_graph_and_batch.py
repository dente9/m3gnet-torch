# m3gnet/graph/tests/test_graph_and_batch.py (Final Fixed Version)

import unittest
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import DataLoader

# Now we can import everything from the graph package level
from m3gnet.graph import (
    MaterialGraph, 
    RadiusCutoffGraphConverter, 
    MaterialGraphDataset, 
    collate_list_of_graphs
)

class TestGraphAndBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up two simple structures for testing."""
        print("\n--- Setting up for Graph and Batch Tests ---")
        cls.s1 = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
        cls.s2 = Structure(Lattice.cubic(5.4), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        cls.converter = RadiusCutoffGraphConverter(cutoff=4.0)
        
    def test_01_graph_conversion(self):
        """Test the conversion from a Pymatgen structure to a MaterialGraph."""
        print("Testing: Structure to Graph conversion...")
        graph = self.converter.convert(self.s1)
        self.assertIsInstance(graph, MaterialGraph)
        self.assertEqual(graph.n_atoms.item(), 1)
        print("...OK")

    def test_02_dataset(self):
        """Test the MaterialGraphDataset."""
        print("Testing: MaterialGraphDataset...")
        dataset = MaterialGraphDataset(
            structures=[self.s1, self.s2],
            converter=self.converter,
            energies=[1.0, 2.0]
        )
        self.assertEqual(len(dataset), 2)
        graph, energy = dataset[0]
        self.assertIsInstance(graph, MaterialGraph)
        self.assertAlmostEqual(energy.item(), 1.0)
        print("...OK")

    def test_03_collate_and_dataloader(self):
        """Test the collate function and its use in a DataLoader."""
        print("Testing: Collate function and DataLoader...")
        dataset = MaterialGraphDataset(
            structures=[self.s1, self.s2],
            converter=self.converter,
            energies=[1.0, 2.0]
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_list_of_graphs)
        
        batched_graph, (batched_energies,) = next(iter(loader))

        self.assertIsInstance(batched_graph, MaterialGraph)
        self.assertEqual(batched_graph.n_atoms.sum().item(), 3) # 1 + 2 atoms
        self.assertEqual(batched_energies.shape, (2, 1))
        self.assertTrue(torch.allclose(batched_energies.view(-1), torch.tensor([1.0, 2.0])))
        print("...OK")

if __name__ == "__main__":
    unittest.main()