"""Synthetic Three Cluster Graph Script"""

import numpy as np
import kmedoids
from stgkm.distance_functions import s_journey
from stgkm.stgkm_figures import (
    three_snapshots_dynamic_clustering,
    choosing_num_clusters_plot,
)
from stgkm.helper_functions import agglomerative_clustering
from stgkm.synthetic_graphs import ThreeClusterConnectivity

from stgkm.helper_functions import choose_num_clusters

### Create Connectivity Matrix ###
TCC = ThreeClusterConnectivity(num_changes=30, pop_size=10)
three_cluster_connectivity_matrix = TCC.create_three_cluster_connectivity_matrix()

### Calculate s-journey distance ###
distance_matrix = s_journey(three_cluster_connectivity_matrix)


TIME, NUM_VERTICES, _ = distance_matrix.shape
MIN_CLUSTERS = 2
MAX_CLUSTERS = 11
START = 0
END = -10
random_state = np.random.choice(100, 1)[0]

MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
PENALTY = 8

penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)

obj_values, opt_k, label_history, medoid_history = choose_num_clusters(
    min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    penalized_distance=penalized_distance[START:END],
    connectivity_matrix=three_cluster_connectivity_matrix[START:],
    random_state=random_state,
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    max_iterations=100,
    medoid_selection="num",
)

km = kmedoids.KMedoids(
    opt_k,
    method="fasterpam_time",
    max_drift=1,
    drift_time_window=1,
    max_iter=1000,
    random_state=1,
    online=False,
)
c = km.fit(penalized_distance)
opt_labels = c.labels_
opt_medoids = c.medoid_indices_

opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=opt_k)
print("k: ", opt_k)

## Visualize three snapshots of the dynamic graph ###
three_snapshots_dynamic_clustering(
    timesteps=[0, 1, 2],
    membership=opt_labels,
    num_clusters=3,
    connectivity_matrix=three_cluster_connectivity_matrix,
    centers=opt_medoids,
    fig_title="Cluster Evolution",
    snapshot_title="Timestep ",
    filepath="Synthetic_cluster_evolution.pdf",
)

choosing_num_clusters_plot(
    min_num_clusters=MIN_CLUSTERS,
    max_num_clusters=MAX_CLUSTERS,
    sum_distance_from_centers=obj_values,
    fig_title="Avg. Silhouette Score vs. Number of Clusters",
    ylabel="Avg. Silhouette Score",
    filepath="Synthetic_opt_num_clusters.pdf",
)
