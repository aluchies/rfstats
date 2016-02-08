import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from numpy import real
from itertools import product, repeat
from blocks import spanning_blocks
from sets import Set

def container_indicator_cube(array_shape):
    """Helper function for defining a cube container
    """

    return np.ones(array_shape).astype(bool)



def container_indicator_sphere(array_shape):
    """Helper function for defining spherical container

    Arrays have shape MxNxP... depending on the number of dimensions. The
    ability to analyze subsets of the array, for example a hyper sphere included
    inside the array is desirable. The subset of the array is called a
    *container*.
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

    indices_container_fft = fftn(indices_container.astype(int), s=s, axes=a)
    normalize_matrix = real(fftshift(ifftn(np.abs(indices_container_fft) ** 2,
        s=s, axes=a)))
    normalize_matrix = np.rint(normalize_matrix).astype(int)
    normalize_matrix[normalize_matrix == 0] = 1

    return normalize_matrix



def sum_raw(vol):
    """Fast convolution using the FFT.

    See Understanding DSP by Lyons, p 515 or DSP by Proakis p 426
    """

    array_shape = vol.shape
    s = [2 * i - 1 for i in array_shape]
    a = xrange(len(array_shape))

    vol_fft = fftn(vol, s=s, axes=a)
    vol_fft_ifft = real(fftshift(ifftn(np.abs(vol_fft) ** 2, s=s, axes=a)))

    return vol_fft_ifft



def sum_one_moving_block(vol, center_block, block_set=None):
    """Find correlation between one block and all other spanning blocks 
    for the array
    """

    # set params for summation
    array_shape = vol.shape
    ndim = len(vol.shape)
    block_shape = center_block.shape
    step = 1

    # Find spanning blocks if necessary
    if not block_set:
        block_set = spanning_blocks(array_shape=array_shape,
                block_shape=block_shape, step=1)

    # find correlation value and lag values for each
    R, lags = zip(*map(_sum_two_blocks, block_set, repeat(center_block,
        len(block_set)), repeat(vol, len(block_set))))
    return R, lags




def _sum_two_blocks(block0, block1, vol):
    """Helper function that beats two blocks against each and finds summation
    of the resulting product.
    """

    b0 = vol[block0]
    b0 = b0 - b0.mean()

    b1 = vol[block1]
    b1 = b1 - b1.mean()

    CC = np.sum(b0 * b1)

    lags = []
    for n in xrange(block0.ndim):
        lags.append(block0[n].start - block1[n].start)

    return CC, lags



def sin(f, x, u):
    """Sin transform using trapezoids.
    """
    assert len(f) == len(x), 'Length of f and x vectors differ'
    out = np.zeros(len(u))
    for i, ui in enumerate(u):
        y = f * np.sin(2 * np.pi * ui * x)
        out[i] = np.trapz(y=y, x=x)
    return out



def fourier_transform_isotropic_3D(y, x, u0, u1, N=2**7):
    """
    """

    u = np.linspace(u0, u1, N)
    k = 2 * np.pi * u

    nonzero_i = np.nonzero(u)[0]
    zero_i = np.nonzero(u == 0.)[0]

    S = np.ones(N)
    S[nonzero_i] = sin(y * x, x, u[nonzero_i]) / u[nonzero_i]
    S[zero_i] = np.trapz(y=y * x ** 2, x=x) * 2 * np.pi
    S = S / S.max()

    return S, k



# def sum_all_moving_blocks(vol, block_set):
#     """Find correlation by considering dot product between every block having
#     block_shape in the volume and every other block having block_shape in
#     volume.
#     """

#     array_shape = vol.shape
#     Ny, Nx = array_shape
#     step = 1

#     # block set
#     bs = spanning_blocks(array_shape=array_shape, block_shape=block_shape, step=step)

#     center_block = center_bs[0]
#     ndim = len(center_block)
#     s = [array_shape[n] - block_shape[n] + 1 for n in xrange(ndim)]

#     vals = []
#     lags = []

#     lags = [Set() for n in xrange(ndim)]

#     for block0, block1 in bs_pairs:

#         b0 = vol[block0]
#         b0 = b0 - b0.mean()

#         b1 = vol[block1]
#         b1 = b1 - b1.mean()

#         vals.append(b0 * b1)

#         lags_list = []
#         for n in xrange(ndim):
#             lags_list.append(block0[n].start - block1[n].start)
#         lags_squared = map(lambda x: x ** 2, lags_list)
#         lag = sum(lags_squared)

#         lags.append(lag)

#     return vals, lags