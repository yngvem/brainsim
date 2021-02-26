import vtk
from brainsim import vtk_utils
import numpy as np


def test_set_transform_matrix():
    A = np.ascontiguousarray(np.random.standard_normal((4, 4)))
    transform = vtk.vtkTransform()
    vtk_utils.set_transform_matrix(A, transform)
    assert np.allclose(A, vtk_utils.get_transform_matrix(transform))

    A = np.asfortranarray(np.random.standard_normal((4, 4)))
    transform = vtk.vtkTransform()
    vtk_utils.set_transform_matrix(A, transform)
    assert np.allclose(A, vtk_utils.get_transform_matrix(transform))

    A = np.random.standard_normal((8, 8))[::2, ::2]
    transform = vtk.vtkTransform()
    vtk_utils.set_transform_matrix(A, transform)
    assert np.allclose(A, vtk_utils.get_transform_matrix(transform))