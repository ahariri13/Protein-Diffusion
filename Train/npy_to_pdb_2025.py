import numpy as np

def read_and_reshape_npy(file_path, reshape_dim=(2191, 3)):
    # Load the .npy file
    data = np.load(file_path)
    print(data.shape)
    # Calculate N and reshape
    N = data.shape[0] #// reshape_dim[0]
    # reshaped_data = data.reshape(N, 2191,3)
    return data#reshaped_data

def load_pdb(pdb_path):
    # Read the PDB file and store lines
    with open(pdb_path, 'r') as file:
        lines = file.readlines()
    # Filter atom lines (ATOM/HETATM)
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    return lines, atom_lines

def generate_pdb_files(atom_lines, reshaped_data, output_dir, num_files=10):
    # Ensure the number of files doesn't exceed the number of available slices
    num_files = min(num_files, reshaped_data.shape[0])

    for i in range(num_files):
        # Get the coordinates for the current slice
        coordinates = reshaped_data[i]

        # Create new PDB content with updated coordinates
        new_pdb_lines = []
        for j, line in enumerate(atom_lines[:coordinates.shape[0]]):
            x, y, z = coordinates[j]
            new_line = (
                f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"  # Update X, Y, Z columns
            )
            new_pdb_lines.append(new_line)
        
        # return new_pdb_lines

        # Save the new PDB file
        output_path = f"{output_dir}/Pred{i+1}.pdb"
        # output_path = f"./Reconstructions_Pooling/PDB/PredK12{i+1}.pdb"
        # output_path = f"./Sequential.pdb"
        # output_path = output_dir
        with open(output_path, 'w') as file:
            file.writelines(new_pdb_lines)

        # print(f"PDB file saved: {output_path}")


# # Example Usage
# npy_file = "./preds_sequential.npy"
# # Paths to reference PDB and base directory
# pdb_file  = f"./heavy_chain.pdb"  # Adjust path

# output_directory = "./"

# # Step 1: Read and reshape .npy
# reshaped_array = read_and_reshape_npy(npy_file)

# # Step 2: Load PDB file
# all_lines, atom_lines = load_pdb(pdb_file)

# # Step 3: Generate 10 PDB files
# generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)