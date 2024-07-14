"""Functions for running different algorithms and comparing their performance."""

from typing import List, Mapping, Tuple, Optional
import os
from sklearn.metrics.cluster import adjusted_mutual_info_score
import numpy as np
import networkx as nx
import kmedoids
import matplotlib.pyplot as plt
from DCDID1.run_DCDID import run_dcdid, convert_labels
from stgkm.helper_functions import run_stgkm
from stgkm.helper_functions import agglomerative_clustering


def static_kmedoids(connectivity_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Run k-medoids on a collapsed, static graph

    Inputs:
        connectivity_matrix (np.ndarray): Connectivity matrix txNxN
        num_clusters (int): Number of clusters
    Outputs:
        (np.ndarray) 1xn array containing a label for each node in the graph.
    """

    sum_mat = 1 / (np.sum(connectivity_matrix, axis=0) + 0.0001)
    km = kmedoids.KMedoids(n_clusters=num_clusters, method="fasterpam")
    c = km.fit(sum_mat)
    return c.labels_


def run_connected_components(connectivity_matrix: np.ndarray) -> List[Mapping]:
    """
    Run connected components on every time slice of a connectivity matrix.

    Inputs:
        connectivity_matrix (np.ndarray): Connectivity matrix txNxN
    Outputs:
        (List[Mapping]): List of length t containing a dictionary of labels for each time step. The dictionary has the form {cluster: [node]}.
    """
    all_labels = []
    for time, _ in enumerate(connectivity_matrix):
        graph_slice = nx.from_numpy_array(connectivity_matrix[time])
        clusters = [
            (index, list(c))
            for index, c in enumerate(nx.connected_components(graph_slice))
        ]
        all_labels.append(dict(clusters))
    return all_labels


def compare_performance(
    connectivity_matrix: np.ndarray,
    true_labels: np.ndarray,
    stgkm_args: Optional[Mapping] = {},
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare performance (AMI score) of STGkM, DCDID, connected components, and static k-medoids on a connectivity matrix.

    Inputs:
        connectivity_matrix (np.ndarray): Connectivity matrix txNxN
        true_labels (np.ndarray): 1xn array of true long-term clusters
        stgkm_args (Mapping): Dictionary containing arguments for STGkM.
    Returns:
       stgkm_score (float): AMI score of STGkM performance.
       dcdid_score (float): AMI score of DCDID performance.
       cc_score (float): AMI score of connected components performance
       static_km_score (float): AMI score of static k-medoids performance
       stgkm_label_mat (np.ndarray): txnxk STGkM binary assignment matrix
       dcdid_label_mat (np.ndarray): txnxk DCDID binary assignment matrix
       cc_label_mat (np.ndarray): txnxk connected components binary assignment matrix
    """

    num_clusters = len(np.unique(true_labels))

    _, stgkm_label_mat, stgkm_ltc = run_stgkm(
        connectivity_matrix=connectivity_matrix,
        penalty=stgkm_args.get("penalty", 5),
        num_clusters=num_clusters,
        max_drift=stgkm_args.get("max_drift", 1),
        drift_time_window=stgkm_args.get("drift_time_window", 1),
        max_iter=stgkm_args.get("max_iter", 100),
        random_state=stgkm_args.get("random_state", 1),
    )

    stgkm_score = adjusted_mutual_info_score(
        labels_true=true_labels, labels_pred=stgkm_ltc
    )

    dcdid_labels = run_dcdid(connectivity_matrix=connectivity_matrix)
    dcdid_label_mat = convert_labels(
        connectivity_matrix=connectivity_matrix, all_labels=dcdid_labels
    )
    dcdid_ltc = agglomerative_clustering(
        weights=dcdid_label_mat.T, num_clusters=num_clusters
    )
    dcdid_score = adjusted_mutual_info_score(
        labels_true=true_labels, labels_pred=dcdid_ltc
    )

    cc_labels = run_connected_components(connectivity_matrix=connectivity_matrix)
    cc_label_mat = convert_labels(
        connectivity_matrix=connectivity_matrix, all_labels=cc_labels
    )
    cc_ltc = agglomerative_clustering(weights=cc_label_mat.T, num_clusters=num_clusters)
    cc_score = adjusted_mutual_info_score(labels_true=true_labels, labels_pred=cc_ltc)

    static_km_labels = static_kmedoids(
        connectivity_matrix=connectivity_matrix, num_clusters=num_clusters
    )
    static_km_score = adjusted_mutual_info_score(
        labels_true=true_labels, labels_pred=static_km_labels
    )

    return (
        stgkm_score,
        dcdid_score,
        cc_score,
        static_km_score,
        stgkm_label_mat,
        dcdid_label_mat,
        cc_label_mat,
    )
