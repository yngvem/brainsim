from brainsim import image_tools
from pytest import approx
from itertools import product
import numpy as np


def test_image_interpolator():
    shape = (3, 4, 5)
    random_image = np.random.standard_normal(shape)
    image_interpolator = image_tools.create_image_interpolator(random_image, method='linear', out_of_bounds='nearest')
    for i, j, k in [[0, 0, 0]]:
        assert random_image[i, j, k] == approx(image_interpolator(((i, j, k))).item())
    
    assert random_image[0, 0, 0] == image_interpolator((-1, -1, -1))
    np.testing.assert_allclose(
        [random_image[0, 0, 0], random_image[-1, -1, -1]],
        image_interpolator([[-1, -1, -1], [1000, 1000, 1000]])
    )

    np.testing.assert_allclose(
        [
            0.5*random_image[0, 0, 0] + 0.5*random_image[1, 0, 0],
            0.25*random_image[0, 0, 0] + 0.75*random_image[0, 1, 0],
            random_image[1, 0, 0]
        ],
        image_interpolator([
            [0.5, 0, 0],
            [0, 0.75, 0],
            [1, -1, 0],
        ])
    )