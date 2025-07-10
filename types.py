# m3gnet/types.py

"""
Defines shared type hints for the M3GNet project to improve code clarity and maintainability.
"""
from typing import Union
from ase import Atoms
from pymatgen.core import Structure, Molecule

# A type hint for objects that represent an atomic structure.
# This includes Pymatgen Structures and Molecules, as well as ASE Atoms objects.
StructureOrMolecule = Union[Structure, Molecule, Atoms]