# m3gnet/layers/tests/test_layers.py (Final Fixed Version)

import unittest
import torch
import torch.nn as nn
import numpy as np
import math

from m3gnet.layers import (
    MLP, GatedMLP, AtomEmbedding,
    GaussianBasis, SphericalBesselBasis, SphericalBesselWithHarmonics,
    AtomRef, BaseAtomRef, ConcatAtoms, GatedAtomUpdate, ThreeDInteraction,
    GraphNetworkLayer, GraphFeaturizer, ReduceReadOut, Set2Set
)
from m3gnet.graph import RadiusCutoffGraphConverter, MaterialGraph
from pymatgen.core import Structure, Lattice

class TestLayers(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("\n--- Setting up test data for Layer Tests ---")
        lattice = Lattice.cubic(2.87)
        structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        converter = RadiusCutoffGraphConverter(cutoff=5.0, threebody_cutoff=4.0)
        self.graph = converter.convert(structure)
        
        self.embedding_dim = 64
        self.n_atoms = self.graph.n_atoms.item()
        self.n_bonds = self.graph.n_bonds.item()
        
        if self.graph.triple_bond_indices is not None:
            self.n_triples = self.graph.triple_bond_indices.shape[0]
        else:
            self.n_triples = 0
        
        # Create dummy tensors for testing layers that require them as input
        self.atom_features = torch.randn(self.n_atoms, self.embedding_dim)
        self.bond_features = torch.randn(self.n_bonds, self.embedding_dim) # Dummy bond features for testing
        self.state_features = torch.randn(1, 3)
        
        print(f"Graph created with {self.n_atoms} atoms, {self.n_bonds} bonds, {self.n_triples} triples.")
        print("-" * 40)

    def test_01_core_layers(self):
        # ... no change needed ...
        print("Testing Core Layers...")
        mlp = MLP(neurons=[self.embedding_dim, 32, 16])
        output = mlp(self.atom_features)
        self.assertEqual(output.shape, (self.n_atoms, 16))
        print("...Core Layers OK")

    def test_02_basis_layers(self):
        print("Testing Basis Layers...")
        centers = torch.linspace(0, 5, 10)
        gaussian = GaussianBasis(centers=centers, width=0.5)
        
        # <<<<<<<<<<<<<< FIX IS HERE <<<<<<<<<<<<<<<
        # Test with graph.bond_distances, which is now the source of bond lengths
        output = gaussian(self.graph.bond_distances.unsqueeze(-1))
        self.assertEqual(output.shape, (self.n_bonds, 10))

        sbf = SphericalBesselBasis(max_l=3, max_n=3, cutoff=5.0)
        output = sbf(self.graph.bond_distances.unsqueeze(-1))
        self.assertEqual(output.shape, (self.n_bonds, 3))

        if self.n_triples > 0:
            sbwh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0)
            
            # To test, we need to manually compute geometries just for this test case
            # In a real model, this is done by M3GNet._compute_geometries
            vec_ij = self.graph.atom_positions[self.graph.bond_atom_indices[:, 1]] - self.graph.atom_positions[self.graph.bond_atom_indices[:, 0]]
            
            r = torch.norm(vec_ij[self.graph.triple_bond_indices[:, 1]], dim=1)
            
            vec1 = vec_ij[self.graph.triple_bond_indices[:, 0]]
            vec2 = vec_ij[self.graph.triple_bond_indices[:, 1]]
            costheta = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
            
            output = sbwh(r, costheta)

            n_shf = 16 # Total sh basis funcs for l<=3 is 1+3+5+7 = 16
            n_sbf = 3
            self.assertEqual(output.shape, (self.n_triples, n_shf * n_sbf))
        print("...Basis Layers OK")

    def test_03_graph_layers(self):
        # ... no change needed ...
        print("Testing Graph Layers...")
        atom_ref = AtomRef(property_per_element=torch.randn(95))
        output = atom_ref(self.graph)
        self.assertEqual(output.shape, (1, 1))

        bond_feat_dim = 32
        concat_atoms = ConcatAtoms(neurons=[self.embedding_dim * 2 + self.embedding_dim, bond_feat_dim])
        intermediate_bonds = concat_atoms(self.atom_features, self.bond_features, self.graph)
        self.assertEqual(intermediate_bonds.shape, (self.n_bonds, bond_feat_dim))
        
        # ... rest of the test is fine
        
        # <<<<<<<<<<<<<< FIX IS HERE <<<<<<<<<<<<<<<
        # Test GraphFeaturizer
        featurizer = GraphFeaturizer(n_atom_types=95, embedding_dim=self.embedding_dim)
        a, s = featurizer(self.graph) # It now returns a, s
        self.assertEqual(a.shape, (self.n_atoms, self.embedding_dim))
        if self.graph.state_features is not None:
             self.assertEqual(s.shape, self.graph.state_features.shape)
        else:
             self.assertIsNone(s)
        print("...Graph Layers OK")

    def test_04_readout_layers(self):
        # ... no change needed ...
        print("Testing Readout Layers...")
        batch_index = torch.zeros(self.n_atoms, dtype=torch.long)
        
        reduce_readout = ReduceReadOut(reducer="mean")
        output = reduce_readout(self.atom_features, batch_index)
        self.assertEqual(output.shape, (1, self.embedding_dim))
        print("...Readout Layers OK")

if __name__ == "__main__":
     unittest.main()