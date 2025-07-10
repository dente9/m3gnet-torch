# m3gnet/graph/__init__.py (Final Fixed Version)

"""
This module contains classes and functions for graph representations of materials.
"""

from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter
# Now we can safely import and expose the correctly named function
from .batch import collate_list_of_graphs, MaterialGraphDataset

# Define what is exposed when a user does `from m3gnet.graph import *`
__all__ = ["MaterialGraph", "RadiusCutoffGraphConverter", "collate_list_of_graphs", "MaterialGraphDataset"]