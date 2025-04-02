#!/usr/bin/env python3
import numpy as np
import h5py
import argparse
import os

def read_and_reshape_h5(file_path, key, reshape_dim=(2191, 3)):
    """
    Reads the dataset with the given key from the HDF5 file.
    If the loaded data is 2D, reshapes it to (N, *reshape_dim).
    Otherwise, returns the data as is.
    """
    with h5py.File(file_path, 'r') as hf:
        data = np.array(hf[key])
        print(f"Data shape from {file_path} (key: {key}): {data.shape}")
    if len(data.shape) == 2:
        # Assume data is concatenated coordinates for multiple structures.
        N = data.shape[0] // reshape_dim[0]
        reshaped_data = data.reshape(N, *reshape_dim)
        return reshaped_data
    else:
        return data

def load_pdb(pdb_path):
    """
    Reads a PDB file and returns all its lines and the atom lines (those starting with ATOM or HETATM).
    """
    with open(pdb_path, 'r') as file:
        lines = file.readlines()
    atom_lines = [line for line in lines if line.startswith("ATOM") or line.startswith("HETATM")]
    return lines, atom_lines

def generate_pdb_files(atom_lines, reshaped_data, output_dir, num_files=10, prefix="pdbO_"):
    """
    For each structure in reshaped_data (assumed shape: [N, num_atoms, 3]),
    update the atom coordinates in the first num_atoms lines of the reference PDB 
    and write out a new PDB file.
    """
    num_files = min(num_files, reshaped_data.shape[0])
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        coordinates = reshaped_data[i]
        new_pdb_lines = []
        # Update only as many atom lines as there are coordinates
        for j, line in enumerate(atom_lines[:coordinates.shape[0]]):
            x, y, z = coordinates[j]
            # Update columns 31-54 with new coordinates (typical PDB coordinate format)
            new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            new_pdb_lines.append(new_line)
        output_path = os.path.join(output_dir, f"{prefix}{i+1}.pdb")
        with open(output_path, 'w') as file:
            file.writelines(new_pdb_lines)
        print(f"PDB file saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate PDB files from an HDF5 file containing reconstructions."
    )
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to the HDF5 file with reconstruction data.")
    parser.add_argument("--key", type=str, required=True,
                        help="Key in the HDF5 file to use for reconstruction (e.g., 'reconstructions').")
    parser.add_argument("--pdb_file", type=str, required=True,
                        help="Path to the base/reference PDB file (e.g., base.pdb).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory where generated PDB files will be saved.")
    parser.add_argument("--num_files", type=int, default=10,
                        help="Number of PDB files to generate (default: 10).")
    args = parser.parse_args()

    # Check that the reference PDB exists
    if not os.path.exists(args.pdb_file):
        print(f"Reference PDB file not found: {args.pdb_file}")
        return

    # Load the base PDB file
    all_lines, atom_lines = load_pdb(args.pdb_file)
    print(f"Loaded {len(atom_lines)} atom lines from the reference PDB.")

    # Read and reshape the HDF5 data using the provided key
    reshaped_data = read_and_reshape_h5(args.h5_file, key=args.key, reshape_dim=(2191, 3))
    print(f"Reshaped data has shape: {reshaped_data.shape}")

    # Create the output directory (if it doesn't exist)
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate PDB files
    print("Generating PDB files from reconstruction data...")
    generate_pdb_files(atom_lines, reshaped_data, args.output_dir, num_files=args.num_files, prefix="generated_")

if __name__ == "__main__":
    main()