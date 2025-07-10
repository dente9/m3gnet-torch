# m3gnet/graph/__init__.py (Final Correct Version)

"""
This module contains classes and functions for graph representations of materials.
"""

from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter

# We now export two separate collate functions for different trainer needs.
from .batch import collate_list_of_graphs, collate_potential_graphs

# MaterialGraphDataset is part of the training module, so it is NOT exported here.

# Define what is exposed when a user does `from m3gnet.graph import *`
__all__ = [
    "MaterialGraph", 
    "RadiusCutoffGraphConverter", 
    "collate_list_of_graphs",
    "collate_potential_graphs",
]