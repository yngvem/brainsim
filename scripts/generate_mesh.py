import pathlib
from tempfile import TemporaryDirectory
import argparse
from brainsim import mesh_tools
import meshio


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


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path, help="Path to input file or directory containing all input files.")
parser.add_argument("output_file", type=Path)
parser.add_argument("--resolution", type=int, default=16)


args = parser.parse_args()
output_file = args.output_file

stem = output_file.stem


with TemporaryDirectory() as parent:
    parent = Path(parent)
    if output_file.suffix != ".mesh":
        temp_output_file = (parent / output_file.stem).append_suffix(".mesh")
    else:
        temp_output_file = output_file

    print("Creating volume mesh...")
    mesh_tools.create_volume_mesh(
        input=args.input,
        output_file=temp_output_file,
        resolution=args.resolution
    )

    if output_file.suffix != ".mesh":
        print("Converting mesh format")
        meshio.read(temp_output_file).write(output_file)
