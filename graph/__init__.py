# m3gnet/graph/__init__.py (Final Version with Updated Export)

"""
This module contains classes and functions for graph representations of materials.
"""

# Export the core classes and the type hint
from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter, StructureOrMolecule

# Export the specific collate functions
from .batch import collate_list_of_graphs, collate_potential_graphs

# Define what is exposed when a user does `from m3gnet.graph import *`
__all__ = [
    "MaterialGraph", 
    "RadiusCutoffGraphConverter", 
    "StructureOrMolecule",
    "collate_list_of_graphs",
    "collate_potential_graphs",
]