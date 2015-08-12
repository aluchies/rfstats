import unittest
import numpy as np
from rfstats import container_indicator_cube, container_indicator_sphere, \
    find_normalize_matrix, sum_raw

class TestCode(unittest.TestCase):



    def test_container_indicator_cube(self):
        """
        """

        array_shape = (2, 2)
        indices = container_indicator_cube(array_shape)
        indices_real = np.asarray([[True, True], [True, True]])
        self.assertTrue(np.allclose(indices, indices_real))

    def test_container_indicator_sphere(self):

        array_shape = (3, 3)
        indices = container_indicator_sphere(array_shape)
        indices_real = np.asarray([[False,  True, False],
                                    [ True,  True,  True],
                                    [False,  True, False]])
        self.assertTrue(np.allclose(indices, indices_real))

    def test_find_normalize_matrix(self):
        array_shape = (3, 3)
        indices = container_indicator_sphere(array_shape)
        nm = find_normalize_matrix(indices)
        nm_real = np.asarray([[1, 1, 1, 1, 1],
                                [1, 2, 2, 2, 1],
                                [1, 2, 5, 2, 1],
                                [1, 2, 2, 2, 1],
                                [1, 1, 1, 1, 1]])
        self.assertTrue(np.allclose(nm, nm_real))


    def test_sum_raw(self):
        vol = np.ones( (3, 3) )
        R = sum_raw(vol)
        R_real = np.asarray([[ 1.,  2.,  3.,  2.,  1.],
                            [ 2.,  4.,  6.,  4.,  2.],
                            [ 3.,  6.,  9.,  6.,  3.],
                            [ 2.,  4.,  6.,  4.,  2.],
                            [ 1.,  2.,  3.,  2.,  1.]])
        self.assertTrue(np.allclose(R, R_real))



if __name__ == '__main__':
    print 'Running unit tests for rfstats.py'
    unittest.main()