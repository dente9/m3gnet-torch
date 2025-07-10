# m3gnet/models/dynamics.py (The Final, Correctly Designed Version)

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
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, potential: Potential, **kwargs):
        super().__init__(**kwargs)
        self.potential = potential
        self.properties = self.implemented_properties

    def calculate(
        self, atoms: Optional[Atoms] = None, properties: list[str] = ['energy'], system_changes: list[str] = all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        compute_forces = "forces" in properties
        compute_stress = "stress" in properties
        graph = self.potential.graph_converter.convert(self.atoms)
        graph = graph.to(self.potential.device)
        energy, forces, stress = self.potential(
            graph, compute_forces=compute_forces, compute_stress=compute_stress
        )
        self.results["energy"] = energy.detach().cpu().numpy().item()
        if compute_forces and forces is not None:
            self.results["forces"] = forces.detach().cpu().numpy()
        if compute_stress and stress is not None:
            s = stress.detach().cpu().numpy()
            self.results["stress"] = s[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]


class Relaxer:
    def __init__(
        self, potential: Potential, optimizer: str = "FIRE", relax_cell: bool = True,
    ):
        self.potential = potential
        self.calculator = M3GNetCalculator(potential=self.potential)
        self.optimizer_class = OPTIMIZERS.get(optimizer)
        if self.optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer}' not supported. Available options: {list(OPTIMIZERS.keys())}")
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    # <<<<<<<<<<<<<<<<<<<< THE FINAL, CORRECT FIX IS HERE <<<<<<<<<<<<<<<<<<<<
    def relax(
        self, atoms: Union[Atoms, Structure, Molecule], fmax: float = 0.1, steps: int = 500, verbose: bool = True, **kwargs
    ) -> dict:
        if not isinstance(atoms, Atoms):
            atoms = self.ase_adaptor.get_atoms(atoms)
        
        # The relaxer is responsible for setting the calculator.
        atoms.set_calculator(self.calculator)

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            dyn_atoms = ExpCellFilter(atoms, hydrostatic_strain=kwargs.get('hydrostatic_strain', True)) if self.relax_cell and np.any(atoms.pbc) else atoms
            
            opt_kwargs = kwargs.copy()
            if 'hydrostatic_strain' in opt_kwargs:
                del opt_kwargs['hydrostatic_strain']

            # This is the key change. We pass the kwargs to the optimizer.
            # The test will provide a `logfile` kwarg.
            optimizer = self.optimizer_class(dyn_atoms, **opt_kwargs)
            optimizer.run(fmax=fmax, steps=steps)
        
        final_atoms = dyn_atoms.atoms if isinstance(dyn_atoms, ExpCellFilter) else dyn_atoms
        final_structure = self.ase_adaptor.get_structure(final_atoms)
        
        return {"final_structure": final_structure, "optimizer": optimizer}

class MolecularDynamics:
    pass