import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from numpy import real

def container_indicator_cube(array_shape):
    """Helper function for defining a cube container
    """

    return np.ones(array_shape).astype(bool)



def container_indicator_sphere(array_shape):
    """Helper function for defining spherical container
    """

    ndim = len(array_shape)

    npts = np.mgrid[ [slice(0,array_shape[n]) for n in xrange(ndim)] ]

    # Normalize
    npts_normalized = [2.0 * n / n.max() - 1 for n in npts]
    npts_squared = [n ** 2 for n in npts_normalized]

    R = reduce(lambda x, y : x + y, npts_squared)
    R = np.sqrt(R)

    return R <= 1.0



def find_normalize_matrix(indices_container):
    """
    """

    array_shape = indices_container.shape
    s = [2 * i - 1 for i in array_shape]
    a = range(len(array_shape))

    indices_container_fft = fftn(indices_container.astype(int), shape=s, axes=a)
    normalize_matrix = real(fftshift(ifftn(np.abs(indices_container_fft) ** 2,
        shape=s, axes=a)))
    normalize_matrix = np.rint(normalize_matrix).astype(int)
    normalize_matrix[normalize_matrix == 0] = 1

    return normalize_matrix



def sum_raw(vol):
    """
    """

    array_shape = vol.shape
    s = [2 * i - 1 for i in array_shape]
    a = xrange(len(array_shape))

    vol_fft = fftn(vol, shape=s, axes=a)
    vol_fft_ifft = real(fftshift(ifftn(np.abs(vol_fft) ** 2, shape=s, axes=a)))

    return vol_fft_ifft