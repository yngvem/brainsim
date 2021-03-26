import argparse
import vtk
import pyvista as pv
import nibabel as nib
import numpy as np
import brainsim.mesh_tools as mesh_tools
import brainsim.vtk_utils as vtk_utils
import meshio


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="Nifti file to get transformation matrix from")
parser.add_argument("surface", type=str, help="Freesurfer surface file to get shift of mesh (e.g. lh.pial)")
parser.add_argument("--mesh", type=str, help="FEniCS mesh file, optional, if not supplied, then the surface file is used")


args = parser.parse_args()

nii_img = nib.load(args.image)
points, faces, metadata = nib.freesurfer.read_geometry(args.surface, read_metadata=True)
transformation_matrix = mesh_tools.get_surface_ras_to_image_coordinates_transform(metadata, nii_img)


if args.mesh is not None:
    fem_mesh = meshio.read(args.mesh)
    points = fem_mesh.points
    faces = fem_mesh.cells[0][1]

    mesh_types = {
        'triangle': vtk.VTK_TRIANGLE,
        'tetra': vtk.VTK_TETRA
    }
    mesh_type = np.ones(len(faces), dtype=np.uint8)
    mesh_type[:] = mesh_types[fem_mesh.cells[0][0]]
    extract_edges = fem_mesh.cells[0][0] == 'tetra'

    mesh = pv.UnstructuredGrid(
        np.concatenate([np.ones_like(faces[:, :1])*faces.shape[1], faces], axis=1),
        mesh_type,
        mesh_tools.transform_coords(points, transformation_matrix)
    )
else:
    mesh = pv.UnstructuredGrid(
        np.concatenate([np.ones_like(faces[:, :1])*3, faces], axis=1),
        np.ones(len(faces), dtype=np.uint8)*vtk.VTK_TRIANGLE,
        mesh_tools.transform_coords(points, transformation_matrix)
    )
    extract_edges = False


image = vtk_utils.to_pyvista_grid(nii_img.get_fdata())
image.origin = np.array([0, 0, 0])


def plane_cutter(plotter, image, volume, extract_edges=True, 
                 normal='x', generate_triangles=False,
                 widget_color=None, assign_to_axis=None,
                 tubing=False, origin_translation=True,
                 outline_translation=False, implicit=True,
                 normal_rotation=True, transformations=None, **kwargs):
    from pyvista.utilities import generate_plane
    image_cutter = vtk.vtkCutter()
    volume_cutter = vtk.vtkCutter()
    plane_sliced_meshes = []
    image_cutter.SetInputDataObject(image)
    volume_cutter.SetInputDataObject(volume)
    if not generate_triangles:
        image_cutter.GenerateTrianglesOff()
        volume_cutter.GenerateTrianglesOff()
    
    if extract_edges:
        volume_edges = vtk.vtkExtractEdges()
        volume_edges.SetInputDataObject(volume_cutter.GetOutput())
    else:
        volume_edges = volume_cutter

    if not hasattr(plotter, "plane_sliced_meshes"):
        plotter.plane_sliced_meshes = []
        
    image_sliced_mesh = pv.wrap(image_cutter.GetOutput())
    volume_sliced_mesh = pv.wrap(volume_edges.GetOutput())
    plotter.plane_sliced_meshes.extend([image_sliced_mesh, volume_sliced_mesh])

    def callback(normal, origin):
        # create the plane for clipping
        plane = generate_plane(normal, origin)
        image_cutter.SetCutFunction(plane)
        image_cutter.Update()
        image_sliced_mesh.shallow_copy(image_cutter.GetOutput())

        volume_cutter.SetCutFunction(plane)
        volume_cutter.Update()
        volume_edges.Update()
        volume_sliced_mesh.shallow_copy(volume_edges.GetOutput())


    plotter.add_plane_widget(callback=callback, bounds=volume.bounds,
                            factor=1.25, normal=normal,
                            color=widget_color, tubing=tubing,
                            assign_to_axis=assign_to_axis,
                            origin_translation=origin_translation,
                            outline_translation=outline_translation,
                            implicit=implicit, origin=volume.center,
                            normal_rotation=normal_rotation)
    plotter.add_mesh(image_sliced_mesh, **kwargs)
    plotter.add_mesh(volume_sliced_mesh, **kwargs)

def my_plane_func(normal, origin):
    print(normal, origin)
    mesh_slice = mesh.slice(normal=normal, origin=origin)
    image_slice = image.slice(normal=normal, origin=origin)
    plotter.add_mesh(mesh_slice)
    plotter.add_mesh(image_slice)


plotter = pv.Plotter()
plane_cutter(plotter, image, mesh, extract_edges=extract_edges)
plotter.show()