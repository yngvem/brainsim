import dolfin as pde
import argparse
import nibabel as nib
import brainsim.mesh_tools as mesh_tools
from brainsim.image_tools import create_image_interpolator


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=str, help="FEniCS mesh file")
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("surface", type=str, help="Freesurfer surface file to get shift of mesh")
parser.add_argument("output", type=str, help="File to save the function to (e.g. shear_modulus.pvd)")
parser.add_argument(
    "--interpolation_method",
    type=str,
    default="linear",
    help="How to interpolate the image: linear or nearest"
)
parser.add_argument(
    "--out_of_bounds",
    type=str,
    default="nan",
    help="How to handle points outside the image: nan, nearest, mean or error"
)
parser.add_argument("--function_space", type=str, default="DG")
parser.add_argument("--function_degree", type=int, default=0)

args = parser.parse_args()

# Load data
Gd = nib.load(args.image)
points, faces, metadata = nib.freesurfer.read_geometry(args.surface, read_metadata=True)
mesh = pde.Mesh(args.mesh) 


# Setup function
V = pde.FunctionSpace(mesh, args.function_space, args.function_degree) 
f = pde.Function(V)
Gd_interpolator = create_image_interpolator(
    Gd.get_fdata(),
    method=args.interpolation_method,
    out_of_bounds=args.out_of_bounds
)


# Get coordinates where function must be evaluated
imap = V.dofmap().index_map()
num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
xyz = V.tabulate_dof_coordinates()
xyz = xyz.reshape((num_dofs_local, -1))

# Interpolate image onto mesh coordinates
transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(metadata, Gd)
image_coords = mesh_tools.transform_coords(xyz, transformation_matrix)
f.vector()[:] = Gd_interpolator(image_coords)

# Save output
output_file = pde.File(args.output)
output_file << f

