import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=str, help="FEniCS mesh file")
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("output", type=str, help="File to save the function to (e.g. shear_modulus.pvd)")
parser.add_argument(
    "--mesh_surface",
    type=str,
    help=("The a surface mesh used to generate the 3D FEniCS mesh. "
          "If supplied, it is used to correctly place the mesh in the RAS coordinate system.")
)
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
parser.add_argument("--hdf5_name", type=str, default=None)

args = parser.parse_args()


# Start script after parsing arguments since importing FEniCS is slow
# This allows us to get help without waiting for fenics
import dolfin as pde
import nibabel as nib
import brainsim.mesh_tools as mesh_tools
from brainsim.image_tools import create_image_interpolator


# Load data
nii_img = nib.load(args.image)
mesh = pde.Mesh(args.mesh) 
if args.mesh_surface is not None:
    points, faces, metadata = nib.freesurfer.read_geometry(args.mesh_surface, read_metadata=True)
else:
    metadata = None


# Setup function
V = pde.FunctionSpace(mesh, args.function_space, args.function_degree) 
f = pde.Function(V)
nii_interpolator = create_image_interpolator(
    nii_img.get_fdata(),
    method=args.interpolation_method,
    out_of_bounds=args.out_of_bounds
)

# Get coordinates where function must be evaluated
imap = V.dofmap().index_map()
num_dofs_local = (imap.local_range()[1] - imap.local_range()[0])
xyz = V.tabulate_dof_coordinates()
xyz = xyz.reshape((num_dofs_local, -1))

# Interpolate image onto mesh coordinates
transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(nii_img, surface_metadata=metadata)
image_coords = mesh_tools.transform_coords(xyz, transformation_matrix)
f.vector()[:] = nii_interpolator(image_coords)

# Save output
if args.output.endswith(".h5") and args.hdf5_name is not None:
    output_file = pde.HDF5File(mesh.mpi_comm(), args.output, "w")

    hdf5_name = args.hdf5_name
    if not hdf5_name.startswith("/"):
        hdf5_name = f"/{hdf5_name}"
    output_file.write(f, hdf5_name)
elif args.output.endswith(".h5"): 
    raise ValueError("Must specify --hdf5_name for hdf5 files")
else:
    output_file = pde.File(args.output)
    output_file << f

