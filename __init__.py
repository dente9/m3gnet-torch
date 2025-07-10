# m3gnet/__init__.py (Fixed Version)

"""
The M3GNet project for deep learning potentials for molecules and crystals.
This is a modern, PyTorch-native re-implementation.
"""

# Define the package version
__version__ = "1.0.0-refactored"

# Expose key classes from submodules to the top-level namespace
# This allows users to do `from m3gnet import M3GNet`
from .graph import MaterialGraph, RadiusCutoffGraphConverter
from .models import M3GNet, Potential, M3GNetCalculator, Relaxer
from .train import PotentialTrainer, PropertyTrainer, ModelCheckpoint, EarlyStopping
from .types import StructureOrMolecule

# The collate function is removed from the top-level namespace as it's a
# specific utility for data loading, not a primary API component.
# Users who need it for custom DataLoaders can import it directly:
# from m3gnet.graph.batch import collate_list_of_graphs

# Define what is exposed when a user does `from m3gnet import *`
__all__ = [
    "__version__",
    "MaterialGraph",
    "RadiusCutoffGraphConverter",
    "M3GNet",
    "Potential",
    "M3GNetCalculator",
    "Relaxer",
    "PotentialTrainer",
    "PropertyTrainer",
    "ModelCheckpoint",
    "EarlyStopping",
    "StructureOrMolecule"
]