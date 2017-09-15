from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__author__ = "Joseph O'Connor"

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform_3d(image,
                         alpha_x, sigma_x,
                         alpha_y, sigma_y,
                         random_state,
                         fill_mode="constant",
                         cval=0):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       From https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape) == 2

    random_state = np.random.RandomState(seed=random_state)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma_x, mode="constant", cval=0) * alpha_x
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma_y, mode="constant", cval=0) * alpha_y

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)))

    return map_coordinates(image, indices, order=1, mode=fill_mode, cval=cval).reshape(shape)
