# m3gnet/models/dynamics.py (Final Version)

"""
Tools for running molecular dynamics (MD) and structure relaxations
using the Atomic Simulation Environment (ASE).
"""
import sys
import contextlib
import io
from typing import Optional, Union

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE, BFGS, LBFGS

from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from .m3gnet import M3GNet, Potential

OPTIMIZERS = { "FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS }


class M3GNetCalculator(Calculator):
    """ M3GNet ASE Calculator. """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, potential: Potential, **kwargs):
        super().__init__(**kwargs)
        self.potential = potential

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        super().calculate(atoms, properties, system_changes)
        
        compute_forces = "forces" in self.properties
        compute_stress = "stress" in self.properties

        graph = self.potential.graph_converter.convert(self.atoms)
        graph = graph.to(self.potential.device)

        energy, forces, stress = self.potential(
            graph,
            compute_forces=compute_forces,
            compute_stress=compute_stress
        )

        self.results["energy"] = energy.detach().cpu().numpy().item()
        
        if compute_forces and forces is not None:
            self.results["forces"] = forces.detach().cpu().numpy()
        
        if compute_stress and stress is not None:
            # ASE expects stress in a 6-element Voigt form: [xx, yy, zz, yz, xz, xy]
            # and in units of eV/A^3
            s = stress.detach().cpu().numpy()
            self.results["stress"] = s[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]


class Relaxer:
    """ Wrapper for ASE-based structure relaxation. """
    def __init__(
        self,
        potential: Union[Potential, str] = "MP-2021.2.8-EFS",
        optimizer: str = "FIRE",
        relax_cell: bool = True,
    ):
        if isinstance(potential, str):
            model = M3GNet.load(potential)
            potential = Potential(model=model)
        
        self.potential = potential
        self.calculator = M3GNetCalculator(potential=self.potential)
        self.optimizer_class = OPTIMIZERS.get(optimizer)
        if self.optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer}' not supported. Available options: {list(OPTIMIZERS.keys())}")
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Union[Atoms, Structure, Molecule],
        fmax: float = 0.1,
        steps: int = 500,
        verbose: bool = True,
        **kwargs
    ) -> dict:
        if not isinstance(atoms, Atoms):
            atoms = self.ase_adaptor.get_atoms(atoms)
        
        atoms.set_calculator(self.calculator)

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            if self.relax_cell and atoms.pbc.any():
                 dyn_atoms = ExpCellFilter(atoms, hydrostatic_strain=True)
            else:
                 dyn_atoms = atoms
            optimizer = self.optimizer_class(dyn_atoms, **kwargs)
            optimizer.run(fmax=fmax, steps=steps)
        
        final_atoms = dyn_atoms.atoms if isinstance(dyn_atoms, ExpCellFilter) else dyn_atoms
        final_structure = self.ase_adaptor.get_structure(final_atoms)
        
        return {"final_structure": final_structure, "optimizer": optimizer}

class MolecularDynamics:
    """ Wrapper for ASE-based Molecular Dynamics. """
    def __init__(
        self,
        atoms: Union[Atoms, Structure],
        potential: Union[Potential, str] = "MP-2021.2.8-EFS",
        ensemble: str = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        **kwargs
    ):
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.npt import NPT
        
        if isinstance(potential, str):
            model = M3GNet.load(potential)
            potential = Potential(model)

        if not isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        
        self.atoms = atoms
        self.atoms.set_calculator(M3GNetCalculator(potential=potential))
        
        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(self.atoms, timestep * units.fs, temperature_K=temperature, **kwargs)
        elif ensemble.lower() == "npt":
            pressure_au = kwargs.get("pressure_au", 1.01325 * units.bar)
            self.dyn = NPT(self.atoms, timestep * units.fs, temperature_K=temperature, external_stress=pressure_au, **kwargs)
        else:
            raise ValueError(f"Ensemble '{ensemble}' not supported.")

    def run(self, steps: int):
        self.dyn.run(steps)