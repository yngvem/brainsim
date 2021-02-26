import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator


def get_image_coordinates(image):
    return [np.arange(l) for l in image.shape]


def get_image_coordinate_grid(image):
    return np.meshgrid(*get_image_coordinates(image), indexing='ij')


def create_image_interpolator(image, method='linear', out_of_bounds='nan', ):
    coordinates = get_image_coordinates(image)
    coordinate_grids = get_image_coordinate_grid(image)
    interpolate_function = RegularGridInterpolator(
            coordinates,
            image,
            method=method,
            bounds_error=False,
            fill_value=np.nan
        )
    def interpolator(coords):
        coords = np.asarray(coords)
        one_dim = False
        if coords.ndim == 1:
            one_dim = True
            coords = coords.reshape(1, coords.shape[0])
        out = interpolate_function(coords)
        #griddata(
        #    np.stack([grid.ravel() for grid in coordinate_grids], axis=-1),
        #    np.ascontiguousarray(image).ravel(),
        #    coords,
        #    method=method
        #)
        mask = np.isnan(out)
        if np.any(mask) and out_of_bounds == 'nearest':
            out[mask] = griddata(
                np.stack([grid.ravel() for grid in coordinate_grids], axis=-1),
                np.ascontiguousarray(image).ravel(),
                coords[mask, :],
                method='nearest'
            )
        elif np.any(mask) and out_of_bounds =='mean':
            out[mask] = np.mean(image)
        elif out_of_bounds not in {'nearest', 'mean', 'nan'}:
            raise ValueError(f"`out_of_bounds` must be either `'nearest'` `'mean'` or `'nan'`, not `'{out_of_bounds}'`")
            
        if one_dim:
            out = out[0]

        return out
    return interpolator