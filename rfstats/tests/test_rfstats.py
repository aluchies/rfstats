import unittest
import numpy as np
from rfstats import container_indicator_cube, container_indicator_sphere, \
    find_normalize_matrix, sum_raw, sum_one_moving_block
from blocks import Block, spanning_blocks

class TestCode(unittest.TestCase):



    def test_container_indicator_cube(self):
        """
        """

        array_shape = (2, 2)
        indices = container_indicator_cube(array_shape)
        indices_real = np.asarray([[True, True], [True, True]])
        self.assertTrue(np.allclose(indices, indices_real))

    def test_container_indicator_sphere(self):
        """
        """
        array_shape = (3, 3)
        indices = container_indicator_sphere(array_shape)
        indices_real = np.asarray([[False,  True, False],
                                    [ True,  True,  True],
                                    [False,  True, False]])
        self.assertTrue(np.allclose(indices, indices_real))

    def test_find_normalize_matrix(self):
        """
        """
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
        """
        """
        vol = np.ones( (3, 3) )
        R = sum_raw(vol)
        R_real = np.asarray([[ 1.,  2.,  3.,  2.,  1.],
                            [ 2.,  4.,  6.,  4.,  2.],
                            [ 3.,  6.,  9.,  6.,  3.],
                            [ 2.,  4.,  6.,  4.,  2.],
                            [ 1.,  2.,  3.,  2.,  1.]])
        self.assertTrue(np.allclose(R, R_real))


    def test1_sum_one_moving_block(self):
        """
        """
        vol = np.asarray( [1, 3, 4] )

        one_block = Block([ slice(0, 2) ])
        block_set = spanning_blocks(array_shape=vol.shape,
            block_shape=one_block.shape, step=1)

        R, lags = sum_one_moving_block(vol, one_block, block_set)

        R_real = (2.0, 1.0)
        lags_real = ([0], [1])

        self.assertEqual(R, R_real)
        self.assertEqual(lags, lags_real)



    def test2_sum_one_moving_block(self):
        """
        """
        vol = np.asarray( [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]] )

        one_block = Block([slice(0,2), slice(0, 2)])
        block_set = spanning_blocks(array_shape=vol.shape,
            block_shape=one_block.shape, step=1)

        R, lags = sum_one_moving_block(vol, one_block, block_set)
        R_real = (10.0, 10.0, 10.0, 10.0)
        lags_real = ([0, 0], [0, 1], [1, 0], [1, 1])

        self.assertEqual(R, R_real)
        self.assertEqual(lags, lags_real)



if __name__ == '__main__':
    print 'Running unit tests for rfstats.py'
    unittest.main()