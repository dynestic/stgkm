""" Graph distance functions. """
import numpy as np


def s_journey(connectivity_matrix: np.ndarray):
    """Calculate the s-journey distance between every pair of nodes at every time step.

    Args:
        connectivity_matrix: Connectivity matrix of dynamic graph
    """
    distance_matrix = np.copy(connectivity_matrix)
    timesteps, num_nodes, _ = distance_matrix.shape

    # Replace all off-diagonal zeros with infinity
    diagonal_mask = np.array([np.eye(num_nodes, dtype=bool)] * timesteps)
    distance_matrix = np.where(
        (distance_matrix == 0) & (~diagonal_mask), np.inf, distance_matrix
    )

    # iterate backwards through each time slice
    for time in np.arange(timesteps - 1)[::-1]:
        current_slice = distance_matrix[time, :, :]
        next_slice = distance_matrix[time + 1, :, :]

        # iterate through each row
        for row in range(num_nodes):
            connections = np.where(current_slice[row] == 1)[0]
            ind_to_update = np.where(current_slice[row] > 1)[0]

            # If there are no connections, there's nothing to update
            # If there are no np.infs, there's nothing to update
            if (len(connections) > 0) & (len(ind_to_update) > 0):
                if len(connections) > 1:
                    connections_distance = (
                        np.min(next_slice[connections, :][:, ind_to_update], axis=0) + 1
                    )
                else:
                    connections_distance = (
                        next_slice[connections, :][:, ind_to_update] + 1
                    )
                current_slice[row, ind_to_update] = np.minimum(
                    current_slice[row, ind_to_update], connections_distance
                )
            else:
                continue
    # Replace any nonzero entries on diagonals with zero
    distance_matrix = np.where(diagonal_mask, 0, distance_matrix)
    return distance_matrix


def s_journey_online(prev_t_journies: np.ndarray, new_connectivity: np.ndarray):
    """
    Calculate the s_journey for new connectivity information forward in time, in an online fashion.

    prev_t_journies (np.ndarray): Array storing the s-journies for the previous drift_time_window time steps

    new_connectivity (np.ndarray): Array storing the connectivity matrices for the new time steps.
    """

    new_s_journies = s_journey(connectivity_matrix=new_connectivity)
    new_timesteps, num_nodes, _ = new_s_journies.shape
    prev_timesteps, _, _ = prev_t_journies.shape

    total_distance = np.hstack((prev_t_journies, new_s_journies))

    for time in range(prev_timesteps)[::-1]:
        current_slice = total_distance[time, :, :]
        next_slice = total_distance[time + 1, :, :]

        # iterate through each row
        for row in range(num_nodes):
            connections = np.where(current_slice[row] == 1)[0]
            ind_to_update = np.where(current_slice[row] == np.inf)[0]

            # If there are no connections, there's nothing to update
            # If there are no np.infs, there's nothing to update
            if (len(connections) > 0) & (len(ind_to_update) > 0):
                if len(connections) > 1:
                    connections_distance = (
                        np.min(next_slice[connections, :][:, ind_to_update], axis=0) + 1
                    )
                else:
                    connections_distance = (
                        next_slice[connections, :][:, ind_to_update] + 1
                    )
                current_slice[row, ind_to_update] = np.minimum(
                    current_slice[row, ind_to_update], connections_distance
                )
            else:
                continue
    # Replace any nonzero entries on diagonals with zero
    diagonal_mask = np.array(
        [np.eye(num_nodes, dtype=bool)] * prev_timesteps + new_timesteps
    )
    distance_matrix = np.where(diagonal_mask, 0, total_distance)

    return distance_matrix
