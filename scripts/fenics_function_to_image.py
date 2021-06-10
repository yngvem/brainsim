import argparse
import itertools
from time import time

import dolfin as pde
import nibabel as nib
import numpy as np
from tqdm import tqdm

import brainsim.mesh_tools as mesh_tools
from brainsim.image_tools import create_image_interpolator
from numba import jit

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=str, help="FEniCS mesh file")
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("hdf5_file", type=str, help="File storing the FEniCS function")
parser.add_argument("hdf5_name", type=str, help="Name of function inside the HDF5 file")
parser.add_argument("output", type=str, help="Nifti file to save the function to (e.g. shear_modulus.nii)")
parser.add_argument("--function_space", type=str, default="CG")
parser.add_argument("--function_degree", type=int, default=1)
parser.add_argument("--extrapolation_value", type=float, default=float('nan'))
parser.add_argument("--mask", type=str, help="Mask used to specify which image voxels to evaluate")
parser.add_argument("--skip_value", type=float, help="Voxel value indicating that a voxel should be skipped in the mask. If unspecified, it's the same as the extrapolation value.")
parser.add_argument(
    "--mesh_surface",
    type=str,
    help=("The a surface mesh used to generate the 3D FEniCS mesh. "
          "If supplied, it is used to correctly place the mesh in the RAS coordinate system.")
)
parser.add_argument('--allow_full_mask', dest='allow_full_mask', default=False, action='store_true')

args = parser.parse_args()

# Load data
nii_img = nib.load(args.image)
mesh = pde.Mesh(args.mesh) 
if args.mesh_surface is not None:
    points, faces, metadata = nib.freesurfer.read_geometry(args.mesh_surface, read_metadata=True)
else:
    metadata = None
transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(nii_img, surface_metadata=metadata)


# Setup function
V = pde.FunctionSpace(mesh, args.function_space, args.function_degree) 
f = pde.Function(V)
hdf5 = pde.HDF5File(mesh.mpi_comm(), args.hdf5_file, "r")
hdf5_name = args.hdf5_name
if not hdf5_name.startswith("/"):
    hdf5_name = "/" + hdf5_name
hdf5.read(f, hdf5_name)


# Populate image
def eval_fenics(f, coords, extrapolation_value):
    try:
        return f(*coords)
    except RuntimeError:
        return extrapolation_value

output_data = np.ones_like(nii_img.get_fdata()) * args.extrapolation_value

## Code to get a bounding box for the mesh, used to not iterate over all the voxels in the image
if args.mask is None:
    imap = V.dofmap().index_map()
    num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
    xyz = V.tabulate_dof_coordinates()
    xyz = xyz.reshape((num_dofs_local, -1))
    image_coords = mesh_tools.transform_coords(xyz, transformation_matrix)
    lower_bounds = np.maximum(0, np.floor(image_coords.min(axis=0)).astype(int))
    upper_bounds = np.minimum(output_data.shape, np.ceil(image_coords.max(axis=0)).astype(int))
    all_relevant_indices = itertools.product(
        *(range(start, stop+1) for start, stop in zip(lower_bounds, upper_bounds))
    )
    num_voxels_in_mask = np.product(1 + upper_bounds - lower_bounds)
    fraction_of_image = num_voxels_in_mask / np.product(output_data.shape)
    print(f"Computed mesh bounding box, evaluating {fraction_of_image:.0%} of all image voxels")
    print(f"There are {num_voxels_in_mask} voxels in the bounding box")
else:
    mask = nib.load(args.mask).get_fdata()
    if args.skip_value is None:
        skip_value = args.extrapolation_value
    else:
        skip_value = args.skip_value
    if np.isnan(skip_value):
        print("Extrapolation value is NaN")
        mask = ~np.isnan(mask)
    else:
        mask = ~np.isclose(mask, skip_value)
    nonzeros = np.nonzero(mask)
    num_voxels_in_mask = len(nonzeros[0])
    all_relevant_indices = zip(*nonzeros)
    fraction_of_image = num_voxels_in_mask / np.product(output_data.shape)
    print(f"Using mask, evaluating {fraction_of_image:.0%} of all image voxels")
    print(f"There are {num_voxels_in_mask} voxels in the mask")
    if fraction_of_image > 1 - 1e-10 and not args.allow_full_mask:
        raise ValueError("The supplied mask covers the whole image so you are probably doing something wrong. To allow for this behaviour, run with --allow_full_mask")


progress = tqdm(total=num_voxels_in_mask)
for coords in all_relevant_indices:
    mesh_coords = mesh_tools.transform_coords(coords, transformation_matrix, inverse=True)
    output_data[coords] = eval_fenics(f, mesh_coords, args.extrapolation_value)
    progress.update(1)

# Save output
output_nii = nib.Nifti1Image(output_data, nii_img.affine, nii_img.header)
nib.save(output_nii, args.output)
