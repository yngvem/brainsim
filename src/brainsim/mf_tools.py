import fenics as pde
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


def extrapolate_nans(f, num_neighbours=1, query_size=100, query_increase_factor=2, progress=True):
    V = f.function_space()

    imap = V.dofmap().index_map()
    num_dofs_local =  imap.local_range()[1]-imap.local_range()[0]
    xyz = V.tabulate_dof_coordinates().reshape((num_dofs_local,-1))
    
    kd_tree = KDTree(xyz)
    vec = np.array(f.vector())

    # Iterate over all nan-indices
    indices = np.where(np.isnan(vec))
    assert len(indices) == 1  # indices is a 1-tuple with the relevant indices

    if progress:
        iterate = tqdm
    else:
        iterate = lambda x: x
    for index in iterate(indices[0]):
        k = query_size
        neighbour_values = np.empty(num_neighbours)*np.nan

        while np.any(np.isnan(neighbour_values)):
            if k > vec.shape[0]:
                raise ValueError(f"Cannot find {k} non-nan values")

            dist, indices = kd_tree.query(xyz[index], int(k))
            indices = indices[1:]
            
            neighbour_idx = 0
            for vector_idx in indices:
                if not np.isnan(vec[vector_idx]):
                    neighbour_values[neighbour_idx] = vec[vector_idx]
                    neighbour_idx += 1
                    if neighbour_idx == num_neighbours:
                        break

            k *= query_increase_factor

        vec[index] = np.median(neighbour_values)

    new_f = pde.Function(V)
    new_f.vector()[:] = vec
    return new_f


