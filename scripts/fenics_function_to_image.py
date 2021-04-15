import argparse

import dolfin as pde
import nibabel as nib
import numpy as np
from tqdm import tqdm

import brainsim.mesh_tools as mesh_tools
from brainsim.image_tools import create_image_interpolator

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=str, help="FEniCS mesh file")
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("surface", type=str, help="Freesurfer surface file to get shift of mesh (e.g. lh.pial)")
parser.add_argument("hdf5_file", type=str, help="File storing the FEniCS function")
parser.add_argument("hdf5_name", type=str, help="Name of function inside the HDF5 file")
parser.add_argument("output", type=str, help="Nifti file to save the function to (e.g. shear_modulus.nii)")
parser.add_argument("--function_space", type=str, default="CG")
parser.add_argument("--function_degree", type=int, default=1)
parser.add_argument("--extrapolation_value", type=float, default=float('nan'))

args = parser.parse_args()

# Load data
nii_img = nib.load(args.image)
points, faces, metadata = nib.freesurfer.read_geometry(args.surface, read_metadata=True)
mesh = pde.Mesh(args.mesh) 
transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(metadata, nii_img)


# Setup function
V = pde.FunctionSpace(mesh, args.function_space, args.function_degree) 
f = pde.Function(V)
hdf5 = pde.HDF5File(mesh.mpi_comm(), args.hdf5_file, "r")
hdf5_name = args.hdf5_name
if not hdf5_name.startswith("/"):
    hdf5_name = "/" + hdf5_name
hdf5.read(f, hdf5_name)


# Populate image
def eval_fenics(f, coords):
    try:
        return f(*coords)
    except RuntimeError:
        return args.extrapolation_value

output_data = np.empty_like(nii_img.get_fdata())
progress = tqdm(total=len(output_data.ravel()))
for coords, _ in np.ndenumerate(output_data):
    mesh_coords = mesh_tools.transform_coords(coords, transformation_matrix, inverse=True)
    output_data[coords] = eval_fenics(f, mesh_coords)
    progress.update(1)


# Save output
output_nii = nib.Nifti1Image(output_data, nii_img.affine, nii_img.header)
nib.save(output_nii, args.output)
