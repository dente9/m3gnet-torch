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
    GraphNetworkLayer, GraphFeaturizer, ReduceState,
    ReduceReadOut, WeightedReadout, Set2Set
)
from m3gnet.graph import RadiusCutoffGraphConverter, MaterialGraph
from pymatgen.core import Structure, Lattice


class TestLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lattice = Lattice.cubic(2.87)
        structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        converter = RadiusCutoffGraphConverter(cutoff=5.0, threebody_cutoff=4.0)
        cls.graph = converter.convert(structure)
        
        cls.embedding_dim = 64
        cls.n_atoms = cls.graph.atom_features.shape[0]
        cls.n_bonds = cls.graph.bond_features.shape[0]
        cls.n_triples = cls.graph.triple_bond_indices.shape[0]
        
        cls.atom_features = torch.randn(cls.n_atoms, cls.embedding_dim)
        cls.bond_features = torch.randn(cls.n_bonds, cls.embedding_dim)
        cls.state_features = torch.randn(1, 3)
        
        print("\n--- Setting up test data for Layer Tests ---")
        print(f"Graph created with {cls.n_atoms} atoms, {cls.n_bonds} bonds, {cls.n_triples} triples.")
        print("-" * 40)

    def test_01_core_layers(self):
        print("Testing Core Layers...")
        mlp = MLP(neurons=[self.embedding_dim, 32, 16])
        output = mlp(self.atom_features)
        self.assertEqual(output.shape, (self.n_atoms, 16))
        print("...Core Layers OK")

    def test_02_basis_layers(self):
        print("Testing Basis Layers...")
        centers = np.linspace(0, 5, 10)
        gaussian = GaussianBasis(centers=centers, width=0.5)
        output = gaussian(self.graph.bond_features)
        self.assertEqual(output.shape, (self.n_bonds, 10))

        sbf = SphericalBesselBasis(max_l=3, max_n=3, cutoff=5.0)
        output = sbf(self.graph.bond_features)
        self.assertEqual(output.shape, (self.n_bonds, 3))

        sbwh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0)
        
        # Correctly call the forward method with r and costheta
        r = self.graph.triple_bond_lengths
        costheta = self.graph.triple_features.squeeze(-1)
        
        output = sbwh(r, costheta)

        n_shf = 16 # Total sh basis funcs for l<=3 is 1+3+5+7 = 16
        n_sbf = 3
        self.assertEqual(output.shape, (self.n_triples, n_shf * n_sbf))
        print("...Basis Layers OK")

    def test_03_graph_layers(self):
        print("Testing Graph Layers...")
        atom_ref = AtomRef(property_per_element=torch.randn(95))
        output = atom_ref(self.graph)
        self.assertEqual(output.shape, (1, 1))

        bond_feat_dim = 32
        concat_atoms = ConcatAtoms(neurons=[self.embedding_dim * 2 + self.embedding_dim, bond_feat_dim])
        intermediate_bonds = concat_atoms(self.atom_features, self.bond_features, self.graph)
        self.assertEqual(intermediate_bonds.shape, (self.n_bonds, bond_feat_dim))

        gated_update = GatedAtomUpdate(neurons=[bond_feat_dim, self.embedding_dim])
        updated_atoms = gated_update(self.atom_features, intermediate_bonds, self.graph)
        self.assertEqual(updated_atoms.shape, (self.n_atoms, self.embedding_dim))
        
        gn_layer = GraphNetworkLayer(atom_network=gated_update, bond_network=concat_atoms)
        out_atoms, out_bonds, _ = gn_layer(self.atom_features, self.bond_features, self.state_features, self.graph)
        
        self.assertEqual(out_atoms.shape, self.atom_features.shape)
        self.assertEqual(out_bonds.shape, (self.n_bonds, bond_feat_dim))
        
        basis_dim = 48
        three_body_basis = torch.randn(self.n_triples, basis_dim)
        update_network = MLP([self.embedding_dim, basis_dim])
        fusion_network = MLP([basis_dim, self.embedding_dim])
        three_d_interact = ThreeDInteraction(update_network=update_network, fusion_network=fusion_network)
        updated_bonds = three_d_interact(self.atom_features, self.bond_features, three_body_basis, self.graph)
        self.assertEqual(updated_bonds.shape, self.bond_features.shape)
        
        featurizer = GraphFeaturizer(n_atom_types=95, embedding_dim=self.embedding_dim, rbf_type="SphericalBessel", max_l=3, max_n=3, cutoff=5.0)
        a, b, s = featurizer(self.graph)
        self.assertEqual(a.shape, (self.n_atoms, self.embedding_dim))
        self.assertEqual(b.shape, (self.n_bonds, 3))
        if self.graph.state_features is not None:
             self.assertEqual(s.shape, self.graph.state_features.shape)
        else:
             self.assertIsNone(s)
        print("...Graph Layers OK")

    def test_04_readout_layers(self):
        print("Testing Readout Layers...")
        batch_index = torch.zeros(self.n_atoms, dtype=torch.long)
        
        reduce_readout = ReduceReadOut(reducer="mean")
        output = reduce_readout(self.atom_features, batch_index)
        self.assertEqual(output.shape, (1, self.embedding_dim))
        print("...Readout Layers OK")

if __name__ == "__main__":
     unittest.main()
