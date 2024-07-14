import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from stgkm.distance_functions import s_journey


class CliqueCrossClique:
    """Class for Clique cross Clique."""

    def __init__(self, num_clusters: int, num_members: int, num_timesteps: int):
        self.num_clusters = num_clusters
        self.num_members = num_members
        self.num_timesteps = num_timesteps
        # self.p_intra = p_intra
        # self.p_inter = p_inter
        self.connectivity_matrix = None

    def create_clique_cross_clique(self):
        """
        Create a clique cross clique.
        """

        time_slices = []

        for time in range(self.num_timesteps):
            cluster = np.ones((self.num_members, self.num_members))

            block_matrix = np.zeros(
                (
                    self.num_members * self.num_clusters,
                    self.num_members * self.num_clusters,
                )
            )

            connection = time % self.num_members

            for i in range(self.num_clusters):
                block_matrix[
                    i * self.num_members : (i + 1) * self.num_members,
                    i * self.num_members : (i + 1) * self.num_members,
                ] = cluster

                for j in range(self.num_clusters):
                    # prob = np.random.uniform()
                    # if prob <= self.p_inter:
                    block_matrix[
                        i * self.num_members + connection,
                        j * self.num_members + connection,
                    ] = 1

            time_slices.append(block_matrix)

        connectivity_matrix = np.stack(time_slices)

        if self.connectivity_matrix is None:
            self.connectivity_matrix = connectivity_matrix

        return connectivity_matrix

    def plot_clique_cross_clique(self, filepath: Optional[str] = None):
        """
        Plot 3 time steps of clique cross clique.
        """

        assert (
            self.connectivity_matrix is not None
        ), "Must create connectivity matrix first"

        labels = [[i] * self.num_members for i in range(self.num_clusters)]
        labels = np.concatenate(labels)
        color_dict = {0: "dodgerblue", 1: "red", 2: "limegreen", 3: "green", -1: "cyan"}

        _, axs = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
        for time in range(3):
            graph = nx.from_numpy_array(self.connectivity_matrix[time])

            # Remove self loops from the figure for improved readability
            graph.remove_edges_from(nx.selfloop_edges(graph))

            if time == 0:
                pos = nx.spring_layout(graph)

            nx.draw(
                graph,
                with_labels=True,
                node_color=[color_dict[label] for label in labels],
                font_size=40,
                node_size=np.ones(self.num_members * self.num_clusters) * 2000,
                pos=pos,
                ax=axs[time % 3],
            )

            axs[time % 3].set_title("Timestep %i" % time, size=100)

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, format="eps")
        plt.show()

        return None


class RandomCliqueCrossClique:
    def __init__(
        self,
        num_clusters: int,
        num_members: int,
        num_timesteps: int,
        p_inter: float,
        p_intra: float,
    ):
        assert 1 > p_intra > p_inter > 0, "1 > p_intra > p_inter > 0 must be satisfied."

        self.num_clusters = num_clusters
        self.num_members = num_members
        self.num_timesteps = num_timesteps
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.connectivity_matrix = None

    def create_clique_cross_clique(self):
        """
        Create a clique cross clique.
        """

        time_slices = []

        for _ in range(self.num_timesteps):
            block_matrix = np.zeros(
                (
                    self.num_members * self.num_clusters,
                    self.num_members * self.num_clusters,
                )
            )

            for i in range(self.num_clusters):
                rand_cluster = np.random.uniform(
                    size=(self.num_members, self.num_members)
                )
                cluster = np.where(rand_cluster < self.p_intra, 1, 0)
                np.fill_diagonal(cluster, np.ones(self.num_members))

                block_matrix[
                    i * self.num_members : (i + 1) * self.num_members,
                    i * self.num_members : (i + 1) * self.num_members,
                ] = cluster

                for j in range(self.num_clusters):
                    for k in range(self.num_members):
                        prob = np.random.uniform()
                        if prob < self.p_inter:
                            block_matrix[
                                i * self.num_members + k,
                                j * self.num_members + k,
                            ] = 1

            time_slices.append(block_matrix)

        connectivity_matrix = np.stack(time_slices)

        if self.connectivity_matrix is None:
            self.connectivity_matrix = connectivity_matrix

        return connectivity_matrix

    def plot_clique_cross_clique(self, filepath: Optional[str] = None):
        """
        Plot 3 time steps of clique cross clique.
        """

        assert (
            self.connectivity_matrix is not None
        ), "Must create connectivity matrix first"

        labels = [[i] * self.num_members for i in range(self.num_clusters)]
        labels = np.concatenate(labels)
        color_dict = {0: "dodgerblue", 1: "red", 2: "limegreen", 3: "green", -1: "cyan"}

        _, axs = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
        for time in range(3):
            graph = nx.from_numpy_array(self.connectivity_matrix[time])
            # Remove self loops from the figure for improved readability
            graph.remove_edges_from(nx.selfloop_edges(graph))

            if time == 0:
                pos = nx.spring_layout(graph)

            nx.draw(
                graph,
                with_labels=True,
                node_color=[color_dict[label] for label in labels],
                font_size=40,
                node_size=np.ones(self.num_members * self.num_clusters) * 2000,
                pos=pos,
                ax=axs[time % 3],
            )
            axs[time % 3].set_title("Timestep %i" % time, size=100)

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, format="eps")
        plt.show()

        return None

    def clique_simulation(self, num_simulations, penalty):
        """
        Simulate clique cross clique.
        """

        assert (
            1 > self.p_intra > self.p_inter > 0
        ), "Simulation results are for 1 > self.intra > self.inter > 0"

        simulation_results = np.zeros(
            (
                self.num_timesteps,
                self.num_members * self.num_clusters,
                self.num_members * self.num_clusters,
                num_simulations,
            )
        )

        for sim in range(num_simulations):
            connectivity_matrix = self.create_clique_cross_clique()
            distance_matrix = s_journey(connectivity_matrix)
            penalized_distance = np.where(
                distance_matrix == np.inf, penalty, distance_matrix
            )
            simulation_results[:, :, :, sim] = penalized_distance
        return simulation_results

    def get_expectation(
        self,
        simulation_results: np.ndarray,
        node_u: int,
        node_v: int,
        time: int,
    ):

        total_vertices = self.num_members * self.num_clusters
        assert 0 <= node_u < total_vertices, "Invalid node."
        assert 0 <= node_v < total_vertices, "Invalid node."
        assert 0 <= time < self.num_timesteps, "Invalid time."

        # vals, counts = np.unique(
        #     simulation_results[time, node_u, node_v, :], return_counts=True
        # )

        # expectation = np.sum(vals * counts / num_simulations)

        expectation = np.average(simulation_results[time, node_u, node_v, :])

        return expectation

    def compare_expectations(
        self,
        node_u: int,
        node_v: int,
        time: int,
        verbose: bool = False,
        simulation_results: Optional[np.ndarray] = None,
        num_simulations: Optional[int] = None,
        penalty: Optional[int] = None,
    ):

        assert (
            node_u // self.num_members == node_v // self.num_members
        ), "Nodes u and v must be from the same cluster."

        if simulation_results is None:
            assert (num_simulations is not None) and (
                penalty is not None
            ), "If the simulation has not already been carried out, please provide number of simulations to run and a distance penalty for disconnected vertices."

            print("Running ", num_simulations, " simulations.")

            simulation_results = self.clique_simulation(
                num_simulations=num_simulations, penalty=penalty
            )
        else:
            num_simulations = np.shape(simulation_results)[-1]

        if verbose:
            print(
                "Calculating expectation of distance between vertex ",
                node_u,
                " and vertex ",
                node_v,
                ".",
            )

        in_exp = self.get_expectation(
            simulation_results=simulation_results,
            node_u=node_u,
            node_v=node_v,
            time=time,
        )

        current_cluster = node_u // self.num_members

        if verbose:
            print(
                "Calculating expectations of distances between vertex ",
                node_u,
                " and all vertices in different clusters.",
            )

        results = []
        for index in range(self.num_clusters * self.num_members):
            if not index // self.num_members == current_cluster:
                out_exp = self.get_expectation(
                    simulation_results=simulation_results,
                    node_u=node_u,
                    node_v=index,
                    time=time,
                )

                if in_exp > out_exp:
                    print(
                        "Intra-cluster distance expectation between vertex ",
                        node_u,
                        " and vertex ",
                        node_v,
                        " at time ",
                        time,
                        " is not less than inter-cluster distance expectation between vertex ",
                        node_u,
                        " and vertex ",
                        index,
                    )
                    results.append(0)
                else:
                    results.append(1)

        if verbose:
            print("Finished.")
        return results

    def compare_expectations_all_vertices(self, num_simulations, penalty):
        violations = 0
        sim_mat = self.clique_simulation(
            num_simulations=num_simulations, penalty=penalty
        )
        for time in range(self.num_timesteps):
            print("Comparing across time", time)
            for node_1 in range(self.num_members * self.num_clusters):
                for node_2 in range(self.num_members * self.num_clusters):
                    if node_1 // self.num_members == node_2 // self.num_members:
                        results = self.compare_expectations(
                            node_u=node_1,
                            node_v=node_2,
                            time=time,
                            simulation_results=sim_mat,
                        )

                        if not np.all(np.array(results) == 1):
                            violations += 1
                            print(
                                "Expectation condition not satisfied for vertex ",
                                node_1,
                                ", vertex ",
                                node_2,
                                ", and time ",
                                time,
                                ".",
                            )
        if violations == 0:
            print("Expectation criteria satisfied.")
        else:
            print("Expectation criteria violated ", violations, " times.")

        return None

    def expectation_heatmap(self, num_simulations, penalty):
        sim_mat = self.clique_simulation(
            num_simulations=num_simulations, penalty=penalty
        )

        expectations = np.zeros(
            (
                self.num_timesteps,
                self.num_members * self.num_clusters,
                self.num_members * self.num_clusters,
            )
        )

        for time in range(self.num_timesteps):
            for node_1 in range(self.num_members * self.num_clusters):
                for node_2 in range(self.num_members * self.num_clusters):
                    exp = self.get_expectation(
                        simulation_results=sim_mat,
                        node_u=node_1,
                        node_v=node_2,
                        time=time,
                    )

                    expectations[time, node_1, node_2] = exp
        return expectations


class TheseusClique:
    def __init__(self, num_members: int, num_timesteps):
        self.num_members = num_members
        self.num_timesteps = num_timesteps
        self.connectivity_matrix = None

    def create_theseus_clique(self):
        """
        Create a Theseus clique, as described in the paper.
        """
        connectivity_matrix = np.zeros(
            (self.num_timesteps, self.num_members * 2, self.num_members * 2)
        )

        cluster = np.ones((self.num_members, self.num_members))
        block_matrix = np.zeros((self.num_members * 2, self.num_members * 2))

        for k in range(2):
            block_matrix[
                k * self.num_members : (k + 1) * self.num_members,
                k * self.num_members : (k + 1) * self.num_members,
            ] = cluster

        connectivity_matrix[0] = block_matrix

        for r in range(self.num_timesteps // self.num_members):
            for i in range(self.num_members):
                time = r * self.num_members + i % self.num_members + 1

                if time < self.num_timesteps:
                    prev_connections = connectivity_matrix[time - 1]
                    curr_connections = prev_connections.copy()

                    node_1 = i
                    node_2 = (r + i) % self.num_members + self.num_members

                    # print(node_1, node_2)

                    node_1_connections = np.where(curr_connections[node_1, :] == 1)[0]
                    node_2_connections = np.where(curr_connections[node_2, :] == 1)[0]

                    # Drop previous connections
                    # return the unique values in set 1 that are not in set 2
                    node_1_drops = np.setdiff1d(node_1_connections, node_2_connections)
                    node_2_drops = np.setdiff1d(node_2_connections, node_1_connections)

                    curr_connections[node_1, node_1_drops] = 0
                    curr_connections[node_1_drops, node_1] = 0
                    curr_connections[node_2, node_2_drops] = 0
                    curr_connections[node_2_drops, node_2] = 0

                    # node_1 adds anything node_2 is connected to
                    # node_2 adds anything node_1 is connected to

                    curr_connections[node_2, node_1_connections] = 1
                    curr_connections[node_1_connections, node_2] = 1
                    curr_connections[node_1, node_2_connections] = 1
                    curr_connections[node_2_connections, node_1] = 1

                    # If the nodes weren't previously connected, make sure they don't connect now
                    if not node_2 in node_1_connections:
                        curr_connections[node_1, node_2] = 0
                    if not node_1 in node_2_connections:
                        curr_connections[node_2, node_1] = 0

                    # reset diagonal
                    np.fill_diagonal(curr_connections, np.ones(self.num_members * 2))

                    connectivity_matrix[time] = curr_connections

        if self.connectivity_matrix is None:
            self.connectivity_matrix = connectivity_matrix
        return connectivity_matrix


class ThreeClusterConnectivity:
    def __init__(self, pop_size: int, num_changes: int):
        """
        Create three cluster connectivity matrix.

        Intracluster nodes are fully connected to begin. At each time step, up to
        num_changes edges are dropped within clusters and added between clusters.

        Args:
            pop_size (int): Size of population of each of three clusters
            changes (int): Max number of edges to drop or add at each time step
        Returns:
            cluster_connectivity_matrix (np.ndarray) Three cluster connectivity matrix.
        """
        self.pop_size = pop_size
        self.num_changes = num_changes

    def create_three_cluster_connectivity_matrix(self) -> np.ndarray:

        cluster = np.ones((self.pop_size, self.pop_size))
        zeros = np.zeros((self.pop_size, self.pop_size))
        three_clusters = np.block(
            [[cluster, zeros, zeros], [zeros, cluster, zeros], [zeros, zeros, cluster]]
        )
        cluster_connectivity_matrix = np.repeat([three_clusters], 20, axis=0)

        for time_slice in range(20):
            connectivity_slice = cluster_connectivity_matrix[time_slice]
            ones_x, ones_y = np.where(connectivity_slice == 1)
            zeros_x, zeros_y = np.where(connectivity_slice == 0)

            num_ones = np.arange(len(ones_x))
            num_zeros = np.arange(len(zeros_x))

            indices_to_delete = np.random.choice(
                num_ones, random.randint(0, self.num_changes)
            )
            indices_to_add = np.random.choice(
                num_zeros, random.randint(0, self.num_changes)
            )

            for indices_y in ones_y[indices_to_delete]:
                for indices_x in ones_x[indices_to_delete]:
                    connectivity_slice[indices_x, indices_y] = 0
            for indices_y in zeros_y[indices_to_add]:
                for indices_x in zeros_x[indices_to_add]:
                    connectivity_slice[indices_x, indices_y] = 1

        return cluster_connectivity_matrix
