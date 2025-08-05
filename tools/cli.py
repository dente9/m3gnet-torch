# m3gnet/cli.py

"""
Command-Line Interface (CLI) for M3GNet.

This script provides easy access to common M3GNet functionalities,
such as structure relaxation.
"""

import argparse
import sys
import logging
import os
from pymatgen.core import Structure

# Import from our refactored M3GNet library
from m3gnet.models import M3GNet, Potential, Relaxer

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress torch's internal warnings if any, can be noisy.
logging.captureWarnings(True)


def relax_structure(args: argparse.Namespace):
    """
    Handles the 'relax' command to relax one or more atomic structures.

    Args:
        args (argparse.Namespace): Parsed arguments from the command line.
    """
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode enabled.")

    # In a real application, you might load a pre-trained model.
    # For this example, we use a randomly initialized model, which is sufficient
    # to demonstrate the relaxation process.
    # In a real CLI tool, you would likely do:
    # model = M3GNet.load("path/to/pretrained/model")
    logging.info("Initializing a randomly weighted M3GNet model for relaxation demo.")
    model = M3GNet(is_intensive=False) # Energy is extensive
    potential = Potential(model=model)
    relaxer = Relaxer(potential=potential, optimizer=args.optimizer, relax_cell=args.relax_cell)

    for filepath in args.infiles:
        try:
            logging.info(f"Loading structure from: {filepath}")
            structure = Structure.from_file(filepath)
            
            if args.verbose:
                print("\n--- Initial Structure ---")
                print(structure)
                print("-" * 25)

            logging.info(f"Starting relaxation for {filepath}...")
            relax_results = relaxer.relax(structure, fmax=args.fmax, steps=args.steps, verbose=args.verbose)
            final_structure = relax_results["final_structure"]
            logging.info("Relaxation complete.")

            if args.suffix:
                basename, ext = os.path.splitext(filepath)
                out_path = f"{basename}{args.suffix}{ext}"
            else:
                # Default to stdout if no output file specified
                out_path = None

            if out_path:
                final_structure.to(filename=out_path)
                logging.info(f"Relaxed structure written to: {out_path}")
            else:
                print("\n--- Relaxed Structure ---")
                print(final_structure)
                print("-" * 25)

        except Exception as e:
            logging.error(f"Failed to process file {filepath}: {e}")
            continue

def main():
    """
    Main function to parse command-line arguments and execute the corresponding command.
    """
    parser = argparse.ArgumentParser(
        description="M3GNet: A modern, PyTorch-native implementation of the Materials Graph Network.",
        epilog="Developed based on the original M3GNet paper."
    )
    
    # Common arguments can be defined on the main parser if needed
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output for all commands.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")

    # --- Relax Sub-command ---
    p_relax = subparsers.add_parser("relax", help="Relax one or more crystal structures.")
    p_relax.add_argument(
        "infiles",
        nargs="+",
        help="Path to one or more input files containing structures (e.g., POSCAR, CIF)."
    )
    p_relax.add_argument(
        "-s", "--suffix",
        type=str,
        default="_relaxed",
        help="Suffix to add to input filenames for saving relaxed structures (e.g., '_relaxed'). Default is '_relaxed'."
    )
    p_relax.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Maximum force tolerance for relaxation (eV/Å). Default is 0.01."
    )
    p_relax.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum number of optimization steps. Default is 500."
    )
    p_relax.add_argument(
        "--optimizer",
        type=str,
        default="BFGS",
        choices=["BFGS", "FIRE", "LBFGS"],
        help="ASE optimizer to use for relaxation. Default is BFGS."
    )
    p_relax.add_argument(
        "--no-relax-cell",
        action="store_false",
        dest="relax_cell",
        help="Fix the lattice cell during relaxation (relax atomic positions only)."
    )
    p_relax.set_defaults(func=relax_structure)

    args = parser.parse_args()
    
    # Execute the function associated with the chosen sub-command
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()