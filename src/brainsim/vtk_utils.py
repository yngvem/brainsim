import numpy as np
import pyvista as pv


def set_transform_matrix(numpy_array, transform):
    transform.SetMatrix(np.ascontiguousarray(numpy_array,).ravel())

def get_transform_matrix(transform):
    M = [[transform.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)]
    return np.array(M)

def to_pyvista_grid(image_stack, name="Scalars_", spacing=(1, 1, 1)):
    image = pv.UniformGrid()
    image.dimensions = np.array(image_stack.shape)
    image.origin = np.array(image_stack.shape)/2
    image.spacing = spacing
    image.point_arrays[name] = image_stack.ravel('F')
    return image
    