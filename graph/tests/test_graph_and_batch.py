# m3gnet/graph/tests/test_graph_and_batch.py (Final Self-Contained Version)

import unittest
import torch
import numpy as np
from pymatgen.core import Lattice, Structure
from torch.utils.data import Dataset, DataLoader

# Import ONLY from the graph module we are testing
from m3gnet.graph import (
    MaterialGraph, 
    RadiusCutoffGraphConverter, 
    collate_list_of_graphs,
    collate_potential_graphs
)

# <<<<<<<<<<<<<<<<<<<< THE FIX IS HERE <<<<<<<<<<<<<<<<<<<<
# Define a minimal, temporary Dataset class *inside the test file*.
# This makes the test self-contained and breaks the import cycle with the train module.
class MinimalPropertyDataset(Dataset):
    """A minimal dataset for testing property prediction data handling."""
    def __init__(self, structures, targets, converter):
        self.structures = structures
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.converter = converter
    def __len__(self):
        return len(self.structures)
    def __getitem__(self, idx):
        graph = self.converter.convert(self.structures[idx])
        return graph, self.targets[idx]

class MinimalPotentialDataset(Dataset):
    """A minimal dataset for testing potential data handling."""
    def __init__(self, structures, energies, forces, converter):
        self.structures = structures
        self.energies = torch.tensor(energies, dtype=torch.float32)
        self.forces = [torch.tensor(f, dtype=torch.float32) for f in forces]
        self.converter = converter
    def __len__(self):
        return len(self.structures)
    def __getitem__(self, idx):
        graph = self.converter.convert(self.structures[idx])
        targets = {"energy": self.energies[idx], "forces": self.forces[idx]}
        return graph, targets

class TestGraphAndBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Setting up for Graph and Batch Tests ---")
        cls.s1 = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
        cls.s2 = Structure(Lattice.cubic(5.4), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        cls.converter = RadiusCutoffGraphConverter(cutoff=4.0)
        
    def test_01_graph_conversion(self):
        print("Testing: Structure to Graph conversion...")
        graph = self.converter.convert(self.s1)
        self.assertIsInstance(graph, MaterialGraph)
        self.assertEqual(graph.n_atoms.item(), 1)
        print("...OK")

    def test_02_property_dataloader(self):
        print("Testing: Property DataLoader...")
        dataset = MinimalPropertyDataset(
            structures=[self.s1, self.s2],
            targets=np.array([1.0, 2.0]),
            converter=self.converter
        )
        # Use the collate function designed for simple property targets
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_list_of_graphs)
        batched_graph, (batched_targets,) = next(iter(loader))
        self.assertEqual(batched_graph.n_atoms.sum().item(), 3)
        self.assertEqual(batched_targets.shape, (2, 1))
        print("...OK")

    def test_03_potential_dataloader(self):
        print("Testing: Potential DataLoader...")
        dataset = MinimalPotentialDataset(
            structures=[self.s1, self.s2],
            energies=np.array([1.0, 2.0]),
            forces=[np.random.rand(1, 3), np.random.rand(2, 3)],
            converter=self.converter
        )
        # Use the collate function designed for potential (dictionary) targets
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_potential_graphs)
        batched_graph, targets_dict = next(iter(loader))
        self.assertEqual(batched_graph.n_atoms.sum().item(), 3)
        self.assertEqual(targets_dict["energy"].shape, (2, 1))
        self.assertEqual(targets_dict["forces"].shape, (3, 3))
        print("...OK")

if __name__ == "__main__":
    unittest.main()