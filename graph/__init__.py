# m3gnet/graph/__init__.py

"""
m3gnet.graph
============

This package contains tools for converting atomic structures to graph representations
and batching them for use in PyTorch models.

The key components are:
- MaterialGraph: A dataclass for holding graph data.
- RadiusCutoffGraphConverter: A utility to convert pymatgen/ASE structures to MaterialGraph objects.
- MaterialGraphDataset: A PyTorch Dataset for handling collections of structures.
- collate_fn: A function to batch multiple MaterialGraph objects into a single large graph.
"""

from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter
from .batch import MaterialGraphDataset, collate_fn

__all__ = [
    "MaterialGraph",
    "RadiusCutoffGraphConverter",
    "MaterialGraphDataset",
    "collate_fn",
]