import argparse
import pathlib
import shutil
from tempfile import TemporaryDirectory

import meshio

from brainsim import mesh_tools


class Path(type(pathlib.Path())):
    def append_suffix(self, suffix):
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return self.with_suffix(self.suffix + suffix)


def boolean(v):
    """Cast strings to boolean variables

    Function by Maxim at stack overflow, edited by Knight71.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_file", type=Path)
parser.add_argument("output_file", type=Path)

parser.add_argument("--remesh", type=boolean, default=True, help="Remesh for better quality")
parser.add_argument("--remesh_target_length", type=float, default=1.0, help="Remesh target edge length")
parser.add_argument("--remesh_num_iterations", type=int, default=3, help="Number of remeshing iterations")
parser.add_argument("--remesh_preserve_boundary_edges", type=boolean, default=False, help="Preserve boundary edges during remeshing")

parser.add_argument("--separate_narrow_gaps", type=boolean, default=True, help="Widen narrow gaps to prevent bridges")
parser.add_argument("--separate_narrow_gaps_adjustment_magnitude", type=float, default=-0.33, help="The factor used to widen the gap")

parser.add_argument("--fill_holes", type=boolean, default=True, help="If true, then holes in the mesh will be filled after separating the gaps")

parser.add_argument("--smoothen_surface", type=boolean, default=False, help="Smooth the mesh surface")
parser.add_argument("--smoothen_num_iterations", type=int, default=1, help="Number of smoothing iterations")
parser.add_argument("--smoothen_strength", type=float, default=1.0, help="Strength of the smoothing operation, only applicable if --smoothen_preserve_volume=False")
parser.add_argument("--smoothen_preserve_volume", type=boolean, default=True, help="If True, then Taubin smoothing is used and the mesh volume is preserved, otherwise, Laplacian smoothing is used.")

parser.add_argument("--collapse_edges", type=boolean, default=True, help="If true, then small triangles will be combined")
parser.add_argument("--collapse_edges_target_length", type=float, default=1.0, help="Target length of the collapsed edges")


args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

stem = output_file.stem
suffix = input_file.suffix

with mesh_tools.process_surface(input_file, output_file) as surface:
    # Remesh surface
    if args.remesh:
        print("Remeshing...")
        surface.isotropic_remeshing(
            args.remesh_target_length,
            args.remesh_num_iterations,
            args.remesh_preserve_boundary_edges
        )

    # Separate narrow gaps
    if args.separate_narrow_gaps:
        print("Separating narrow gaps...")
        surface.separate_narrow_gaps(args.separate_narrow_gaps_adjustment_magnitude)

    # Fill holes
    if args.fill_holes:
        print("Filling mesh holes...")
        surface.fill_holes()

    # Smoothen surface
    if args.smoothen_surface and args.preserve_volume:
        print("Smoothing...")
        surface.smooth_taubin(args.smoothen_num_iterations)
    elif args.smoothen_surface:
        print("Smoothing...")
        surface.smooth_laplacian(args.smoothen_strength, args.smoothen_num_iterations)
    
    # Collapse edges
    if args.collapse_edges:
        print("Collapsing edges...")
        surface.collapse_edges(args.collapse_edges_target_length)

