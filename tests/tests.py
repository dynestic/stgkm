import unittest
import numpy as np
from stgkm.distance_functions import s_journey
from stgkm.helper_functions import similarity_measure, similarity_matrix


class Tests(unittest.TestCase):
    """
    Test class for tkm
    """

    def __init__(self, *args, **kwargs):
        """Initialize test class."""
        super(Tests, self).__init__(*args, **kwargs)

        self.two_cluster_connectivity_matrix = np.array(
            [
                [
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, 0],
                ],
                [
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0],
                ],
            ]
        )

    def test_temporal_graph_distance(self):
        """
        Test temporal graph distance
        """

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            ]
        )

        timesteps, _, _ = connectivity_matrix.shape
        # Ensure test cases are symmetric
        for i in range(timesteps):
            assert np.all(
                connectivity_matrix[i, :, :] == connectivity_matrix[i, :, :].T
            )

        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 4, 2], [1, 0, 1, 2], [2, 1, 0, 1], [3, 3, 1, 0]],
                    [[0, 1, 2, 2], [1, 0, 3, 1], [2, 2, 0, 1], [3, 1, 1, 0]],
                    [
                        [0, np.inf, np.inf, 1],
                        [2, 0, 1, 1],
                        [2, 1, 0, np.inf],
                        [1, 1, 2, 0],
                    ],
                    [
                        [0, 1, 1, np.inf],
                        [1, 0, 1, np.inf],
                        [1, 1, 0, np.inf],
                        [np.inf, np.inf, np.inf, 0],
                    ],
                ]
            )
        )

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            ]
        )
        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 1], [2, 1, 1, 0]],
                    [[0, 2, 1, 2], [np.inf, 0, 1, 1], [1, 1, 0, np.inf], [2, 1, 2, 0]],
                    [
                        [0, 1, np.inf, np.inf],
                        [1, 0, np.inf, np.inf],
                        [np.inf, np.inf, 0, 1],
                        [np.inf, np.inf, 1, 0],
                    ],
                ]
            )
        )

    def test_similarity_measure(self):
        """Test similarity measure."""
        x = np.arange(10)
        y = np.arange(10)[::-1]
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 0)

        x = np.arange(10)
        y = np.arange(10)
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 1)

        x = np.arange(10)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 0.50)

        x = np.arange(10)
        y = np.arange(5)

        self.assertRaises(AssertionError, similarity_measure, vector_1=x, vector_2=y)

    def test_similarity_matrix(self):
        """Test similarity matrix."""
        weights = np.array([[1, 0, 1, 1, 1], [0, 1, 0, 0, 0], [1, 1, 0, 1, 1]])
        sim_mat = similarity_matrix(
            weights=weights, similarity_function=similarity_measure
        )
        target_mat = np.array([[1, 0, 3 / 5], [0, 1, 2 / 5], [3 / 5, 2 / 5, 1]])
        np.testing.assert_array_almost_equal(sim_mat, target_mat)

    def test_stgkm_next_assignment(self):
        """Test stgkm assignment of points at current time step."""
        pass

    def run_tests(self):
        """Run all tests."""
        self.test_temporal_graph_distance()
        self.test_similarity_measure()
        self.test_similarity_matrix()


tests = Tests()
tests.run_tests()
