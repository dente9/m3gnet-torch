# m3gnet/type.py
from typing import Union
from pymatgen.core import Structure, Molecule
from ase import Atoms

# 定义一个在整个项目中可复用的类型别名
StructureOrMolecule = Union[Structure, Molecule, Atoms]