# m3gnet/graph/struct_to_graph.py (Final Version with Type Hint)

"""
Tools for converting atomic structures into graph representations.
"""
from dataclasses import dataclass, fields
from typing import Optional, Union
from itertools import combinations

import torch
import numpy as np
from ase import Atoms
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

# <<<<<<<<<<<<<<<<<<<< TYPE HINT IS NOW DEFINED HERE <<<<<<<<<<<<<<<<<<<<
StructureOrMolecule = Union[Structure, Molecule, Atoms]

@dataclass
class MaterialGraph:
    atom_features: torch.Tensor
    bond_atom_indices: torch.Tensor
    bond_features: torch.Tensor
    atom_positions: torch.Tensor
    pbc_offsets: torch.Tensor
    n_atoms: torch.Tensor
    n_bonds: torch.Tensor
    state_features: Optional[torch.Tensor] = None
    lattices: Optional[torch.Tensor] = None
    has_three_body: bool = False
    triple_bond_indices: Optional[torch.Tensor] = None
    triple_features: Optional[torch.Tensor] = None
    triple_bond_lengths: Optional[torch.Tensor] = None
    n_triples: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "MaterialGraph":
        new_attrs = {}
        for field in fields(self):
            attr = getattr(self, field.name)
            if isinstance(attr, torch.Tensor):
                new_attrs[field.name] = attr.to(device)
            else:
                new_attrs[field.name] = attr
        return self.__class__(**new_attrs)

def _get_fixed_radius_bonding(structure: StructureOrMolecule, cutoff: float, numerical_tol: float = 1e-8) -> tuple:
    if isinstance(structure, (Structure, Molecule)):
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords, dtype=float).copy())
        if isinstance(structure, Structure):
            lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix, dtype=float).copy())
            pbc = np.array([1, 1, 1], dtype=int)
        else:
            lattice_matrix = np.eye(3) * 1000.0
            pbc = np.array([0, 0, 0], dtype=int)
    elif isinstance(structure, Atoms):
        cart_coords = np.ascontiguousarray(np.array(structure.get_positions(), dtype=float).copy())
        lattice_matrix = np.ascontiguousarray(np.array(structure.cell[:], dtype=float).copy())
        pbc = np.array(structure.pbc, dtype=int)
        if not np.any(pbc):
             lattice_matrix = np.eye(3) * 1000.0
    else:
        raise ValueError("Unsupported structure type.")
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=cutoff, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return (
        center_indices[exclude_self], neighbor_indices[exclude_self],
        images[exclude_self], distances[exclude_self],
    )

def _compute_3body(bond_atom_indices: np.ndarray) -> np.ndarray:
    n_bonds = len(bond_atom_indices)
    if n_bonds == 0: return np.empty((0, 2), dtype=int)
    center_atoms = bond_atom_indices[:, 0]
    unique_centers, center_counts = np.unique(center_atoms, return_counts=True)
    valid_centers = unique_centers[center_counts > 1]
    
    bond_indices_by_center = {center: np.where(center_atoms == center)[0] for center in valid_centers}
    
    triple_bond_indices = []
    for center in valid_centers:
        for b1, b2 in combinations(bond_indices_by_center[center], 2):
            triple_bond_indices.extend([[b1, b2], [b2, b1]])
            
    return np.array(triple_bond_indices, dtype=int) if triple_bond_indices else np.empty((0, 2), dtype=int)

class RadiusCutoffGraphConverter:
    def __init__(self, cutoff: float = 5.0, threebody_cutoff: Optional[float] = 4.0):
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        if self.threebody_cutoff is not None and self.threebody_cutoff > self.cutoff:
            raise ValueError("Three-body cutoff cannot be larger than two-body cutoff.")

    def convert(self, structure: StructureOrMolecule, state_features: Optional[np.ndarray] = None) -> MaterialGraph:
        if isinstance(structure, Atoms):
            atom_features = structure.get_atomic_numbers()
            atom_positions = structure.get_positions()
            lattices = structure.cell[:] if np.any(structure.pbc) else None
        else:
            atom_features = [s.specie.Z for s in structure]
            atom_positions = structure.cart_coords
            lattices = structure.lattice.matrix if isinstance(structure, Structure) else None

        center_indices, neighbor_indices, pbc_offsets, bond_distances = _get_fixed_radius_bonding(structure, self.cutoff)
        bond_atom_indices = np.stack([center_indices, neighbor_indices], axis=1)
        
        has_three_body = self.threebody_cutoff is not None and len(bond_distances) > 0
        n_triples = 0
        triple_bond_indices, triple_features, triple_bond_lengths = None, None, None

        if has_three_body:
            three_body_mask = bond_distances <= self.threebody_cutoff
            three_body_bonds_indices_map = np.where(three_body_mask)[0]
            
            if len(three_body_bonds_indices_map) > 1:
                relevant_bond_atom_indices = bond_atom_indices[three_body_mask]
                raw_triple_indices = _compute_3body(relevant_bond_atom_indices)
                
                if raw_triple_indices.shape[0] > 0:
                    triple_bond_indices = three_body_bonds_indices_map[raw_triple_indices]
                    n_triples = len(triple_bond_indices)
                    
                    triple_bond_lengths = bond_distances[triple_bond_indices[:, 1]]

                    center_indices_3b = relevant_bond_atom_indices[raw_triple_indices[:, 0], 0]
                    neighbor_1_indices_3b = relevant_bond_atom_indices[raw_triple_indices[:, 0], 1]
                    neighbor_2_indices_3b = relevant_bond_atom_indices[raw_triple_indices[:, 1], 1]

                    vec1 = atom_positions[neighbor_1_indices_3b] - atom_positions[center_indices_3b]
                    vec2 = atom_positions[neighbor_2_indices_3b] - atom_positions[center_indices_3b]
                    
                    if lattices is not None:
                        pbc1 = pbc_offsets[triple_bond_indices[:, 0]]
                        pbc2 = pbc_offsets[triple_bond_indices[:, 1]]
                        vec1 += np.dot(pbc1, lattices)
                        vec2 += np.dot(pbc2, lattices)

                    cos_theta = np.einsum('ij,ij->i', vec1, vec2) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
                    triple_features = np.clip(cos_theta, -1.0, 1.0).reshape(-1, 1)

        graph_attrs = {
            "atom_features": torch.tensor(atom_features, dtype=torch.long).view(-1, 1),
            "bond_atom_indices": torch.tensor(bond_atom_indices, dtype=torch.long),
            "bond_features": torch.tensor(bond_distances, dtype=torch.float32).view(-1, 1),
            "atom_positions": torch.tensor(atom_positions, dtype=torch.float32),
            "pbc_offsets": torch.tensor(pbc_offsets, dtype=torch.float32),
            "n_atoms": torch.tensor([len(atom_features)], dtype=torch.long),
            "n_bonds": torch.tensor([len(bond_atom_indices)], dtype=torch.long),
            "state_features": torch.tensor(state_features, dtype=torch.float32).view(1, -1) if state_features is not None else None,
            "lattices": torch.tensor(lattices, dtype=torch.float32).view(1, 3, 3) if lattices is not None else None,
            "has_three_body": n_triples > 0,
            "triple_bond_indices": torch.tensor(triple_bond_indices, dtype=torch.long) if n_triples > 0 else None,
            "triple_features": torch.tensor(triple_features, dtype=torch.float32) if n_triples > 0 else None,
            "triple_bond_lengths": torch.tensor(triple_bond_lengths, dtype=torch.float32) if n_triples > 0 else None,
            "n_triples": torch.tensor([n_triples], dtype=torch.long),
        }
        return MaterialGraph(**graph_attrs)