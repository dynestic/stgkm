"""Functions and script for running clique-cross-clique experiments."""

import pickle
from typing import List
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from stgkm.helper_functions import run_stgkm
from stgkm.synthetic_graphs import CliqueCrossClique, RandomCliqueCrossClique
from stgkm.stgkm_figures import plot_expectation_heatmap

NUM_TIMESTEPS = 10
NUM_CLUSTERS = 3
NUM_MEMBERS = 5
P_INTRA = 0.3
P_INTER = 0.2

CXC1 = CliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
)

CXC2 = RandomCliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
    p_intra=P_INTRA,
    p_inter=P_INTER,
)

CXC1.create_clique_cross_clique()
CXC2.create_clique_cross_clique()

FILEPATH_1 = "STGKM_Figures/clique_cross_clique.eps"
FILEPATH_2 = "STGKM_Figures/random_clique_cross_clique.eps"
CXC1.plot_clique_cross_clique(filepath=FILEPATH_1)
CXC2.plot_clique_cross_clique(filepath=FILEPATH_2)


## Compare expectations for a pair of nodes at a specific time
results = CXC2.compare_expectations(
    num_simulations=100, penalty=5, node_u=0, node_v=1, time=5, verbose=True
)

#### Look at all distances across all times
print(
    "Check that E[d^t(u,v)] < E[d^t(u, w)] for all u,v in the same cluster and u,w in different clusters across all times t."
)
CXC2.compare_expectations_all_vertices(num_simulations=1000, penalty=5)
expectations = CXC2.expectation_heatmap(num_simulations=1000, penalty=5)

plot_expectation_heatmap("cxc_heatmap.pdf", expectations)

#### Run STGkM on both connectivity matrices
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
PENALTY = 5
#####

for connectivity_matrix in [CXC1.connectivity_matrix, CXC2.connectivity_matrix]:
    c, opt_labels, opt_ltc = run_stgkm(
        connectivity_matrix=connectivity_matrix,
        penalty=PENALTY,
        num_clusters=3,
        max_drift=1,
        drift_time_window=1,
        max_iter=100,
        random_state=1,
    )

    print(opt_ltc)


#### Carry out sensitivity experiment
def test_sensitivity(
    num_members_list: List[int], num_clusters_list: List[int], p_intra, p_inter
) -> dict:
    """
    Test sensitivity of STGkM performance across a CXC with a specified p_intra and p_inter across various temporal and spatial complexities.

    Inputs:
        num_members_list (List[int]): Number of members in each cluster of the CXC.
        num_clusters_list (List[int]): Number of clusters in the CXC.
        p_intra: Intra-cluster connectivity probability of the CXC.
        p_inter: Inter-cluster connectivity probability of the CXC.
    Returns:
        data (dict): Dictionary storing results of experiments.
    """

    data = {"num_members": [], "num_clusters": [], "num_timesteps": [], "ami_score": []}

    for num_members in num_members_list:
        for num_clusters in num_clusters_list:
            print("Processing ", num_clusters, " clusters with ", num_members, " each.")

            if (p_intra is None) and (p_inter is None):
                cxc = CliqueCrossClique(
                    num_clusters=num_clusters,
                    num_members=num_members,
                    num_timesteps=100,
                )
            else:
                cxc = RandomCliqueCrossClique(
                    num_clusters=num_clusters,
                    num_members=num_members,
                    num_timesteps=100,
                    p_intra=p_intra,
                    p_inter=p_inter,
                )
            connectivity_matrix = cxc.create_clique_cross_clique()

            true_labels = np.concatenate(
                [[i] * num_members for i in range(num_clusters)]
            )

            for time in range(1, 50):
                _, _, opt_ltc = run_stgkm(
                    connectivity_matrix=connectivity_matrix[:time],
                    penalty=10,
                    num_clusters=num_clusters,
                    max_drift=1,
                    drift_time_window=1,
                    max_iter=100,
                    random_state=1,
                )

                score = adjusted_mutual_info_score(
                    labels_true=true_labels, labels_pred=opt_ltc
                )

                data["num_members"].append(num_members)
                data["num_clusters"].append(num_clusters)
                data["num_timesteps"].append(time)
                data["ami_score"].append(score)
    return data


NUM_MEMBERS_LIST = [5, 10, 25, 50, 100]
NUM_CLUSTERS_LIST = [3, 5, 10, 20]
P_INTRA_LIST = [None, 0.5, 0.5, 0.1]
P_INTER_LIST = [None, 0.3, 0.05, 0.01]
FILEPATHS = [
    "sensitivity_intraNone_interNone.pkl",
    "sensitivity_intra5_inter3.pkl",
    "sensitivity_intra5_inter05.pkl",
    "sensitivity_intra1_inter01.pkl",
]

for P_INTRA, P_INTER, FILEPATH in zip(P_INTRA_LIST, P_INTER_LIST, FILEPATHS):
    experiment_output = test_sensitivity(
        num_clusters_list=NUM_CLUSTERS_LIST,
        num_members_list=NUM_MEMBERS_LIST,
        p_intra=P_INTRA,
        p_inter=P_INTER,
    )

    with open(FILEPATH, "wb") as file:
        pickle.dump(experiment_output, file)
