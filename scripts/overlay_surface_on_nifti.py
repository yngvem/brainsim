import argparse
import vtk
import pyvista as pv
import nibabel as nib
import numpy as np
import brainsim.mesh_tools as mesh_tools
import brainsim.vtk_utils as vtk_utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("surface", type=str, help="Freesurfer surface file to get shift of mesh (e.g. lh.pial)")
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

args = parser.parse_args()

mesh = pv.UnstructuredGrid()

nii_img = nib.load(args.image)
points, faces, metadata = nib.freesurfer.read_geometry(args.surface, read_metadata=True)

transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(metadata, nii_img)

mesh = pv.UnstructuredGrid(
    np.concatenate([np.ones_like(faces[:, :1])*3, faces], axis=1),
    np.ones(len(faces), dtype=np.uint8)*vtk.VTK_TRIANGLE,
    mesh_tools.transform_coords(points, transformation_matrix)
)
image = vtk_utils.to_pyvista_grid(nii_img.get_fdata())
image.origin = np.array([0, 0, 0])


def plane_cutter(plotter, meshes, normal='x', generate_triangles=False,
                 widget_color=None, assign_to_axis=None,
                 tubing=False, origin_translation=True,
                 outline_translation=False, implicit=True,
                 normal_rotation=True, **kwargs):
    from pyvista.utilities import generate_plane
    algs = [vtk.vtkCutter() for mesh in meshes] # Construct the cutter object
    plane_sliced_meshes = []
    for alg, mesh in zip(algs, meshes):
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        if not hasattr(plotter, "plane_sliced_meshes"):
            plotter.plane_sliced_meshes = []
        
        plane_sliced_meshes.append(pv.wrap(alg.GetOutput()))
        plotter.plane_sliced_meshes.append(plane_sliced_meshes[-1])

    def callback(normal, origin):
        # create the plane for clipping
        plane = generate_plane(normal, origin)
        for alg, plane_sliced_mesh in zip(algs, plane_sliced_meshes):
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())


    plotter.add_plane_widget(callback=callback, bounds=meshes[0].bounds,
                            factor=1.25, normal=normal,
                            color=widget_color, tubing=tubing,
                            assign_to_axis=assign_to_axis,
                            origin_translation=origin_translation,
                            outline_translation=outline_translation,
                            implicit=implicit, origin=meshes[0].center,
                            normal_rotation=normal_rotation)
    actors = []
    for plane_sliced_mesh in plane_sliced_meshes:
        actors.append(plotter.add_mesh(plane_sliced_mesh, **kwargs))

def my_plane_func(normal, origin):
    print(normal, origin)
    mesh_slice = mesh.slice(normal=normal, origin=origin)
    image_slice = image.slice(normal=normal, origin=origin)
    plotter.add_mesh(mesh_slice)
    plotter.add_mesh(image_slice)

plotter = pv.Plotter()
plane_cutter(plotter, [mesh, image, ])
#plotter.add_plane_widget(my_plane_func, bounds=(0, 80, 0, 80, 0, 80))
#plotter.add_axes()
plotter.show()