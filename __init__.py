# m3gnet/__init__.py (Final Version with Updated Export)

"""
The M3GNet project for deep learning potentials for molecules and crystals.
This is a modern, PyTorch-native re-implementation.
"""

__version__ = "1.0.0-refactored"

# Import from the new definitive locations
from .graph import MaterialGraph, RadiusCutoffGraphConverter, StructureOrMolecule
from .models import M3GNet, Potential, M3GNetCalculator, Relaxer
from .train import PotentialTrainer, PropertyTrainer, ModelCheckpoint, EarlyStopping

# collate functions are considered implementation details of the trainer,
# so we don't expose them at the top level.

__all__ = [
    "__version__",
    "MaterialGraph",
    "RadiusCutoffGraphConverter",
    "StructureOrMolecule",
    "M3GNet",
    "Potential",
    "M3GNetCalculator",
    "Relaxer",
    "PotentialTrainer",
    "PropertyTrainer",
    "ModelCheckpoint",
    "EarlyStopping",
]