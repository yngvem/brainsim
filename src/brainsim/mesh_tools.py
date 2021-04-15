from contextlib import contextmanager
from pathlib import Path
import numpy as np
import SVMTK as svmtk


def transform_coords(coords, transform, inverse=False):
    if inverse:
        transform = np.linalg.inv(transform)
    coords = np.asarray(coords)
    if coords.ndim == 2:
        coords = coords.T
        coords = transform[:3, :3]@coords + transform[:3, 3, np.newaxis]
        coords = coords.T
    else:
        coords = transform[:3, :3] @ coords + transform[:3, 3]

    return coords


def get_cras(image_nii):
    """Get the center of the RAS coordinate system.

    Based on: https://neurostars.org/t/freesurfer-cras-offset/5587/2
    Which links to: https://github.com/nipy/nibabel/blob/d1518aa71a8a80f5e7049a3509dfb49cf6b78005/nibabel/freesurfer/mghformat.py#L630-L643
    """
    shape = np.array(image_nii.shape)
    center = shape / 2
    center_homogeneous = np.hstack((center, [1]))
    transform = image_nii.affine
    return (transform @ center_homogeneous)[:3]


def get_surface_ras_to_image_coordinates_transform(image_nii):
    """Convert from freesurfer surface coordinates to scanner coordinates.

    Freesurfer uses (at least) three different coordinate systems, the RAS system,
    the surface RAS system (which is the RAS system shifted so the center is in the origin)
    and the image coordinates. This function creates a transformation matrix that transforms
    the surface RAS system into image coordinates. To accomplish this, it first computes the
    c_ras (center of RAS system) to translate the surface RAS into the correct RAS coordinates.
    Then it uses the inverse of the image coordinate to RAS transformation to transform the 
    RAS coordinates into image coordinates.

    The image coordinate transform is the affine transform in the nifti file whose image
    coordinates we want to transform into.

    Example
    -------

    >>> import nibabel as nib
    ... img = nib.load("T2W.nii")
    ... translation_matrix = get_surface_ras_to_image_coordinates_transform(img)
    """
    translation_matrix = np.eye(4)
    translation_matrix[:3, -1] = get_cras(image_nii)

    return np.linalg.inv(image_nii.affine)@translation_matrix


@contextmanager
def process_surface(input_file, output_file):
    surface = svmtk.Surface(str(input_file))
    yield surface
    surface.save(str(output_file))


def create_volume_mesh(input, output_file, resolution=16):
    if Path(input).is_dir():
        surface = [svmtk.Surface(str(p)) for p in input.iterdir()]
    else:
        surface = svmtk.Surface(str(input))

    domain = svmtk.Domain(surface)
    domain.create_mesh(resolution)

    domain.save(str(output_file))

