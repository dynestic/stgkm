"""Script for generating Table 1 and Figures 5-10 in the paper."""

import numpy as np
from stgkm.synthetic_graphs import (
    CliqueCrossClique,
    RandomCliqueCrossClique,
    ThreeClusterConnectivity,
    TheseusClique,
)
from stgkm.stgkm_figures import plot_cluster_history
from compare_performance import compare_performance

PLOT = True

#### CXC experiment
NUM_TIMESTEPS = 20
NUM_CLUSTERS = 3
NUM_MEMBERS = 5

CXC1 = CliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
)

CXC2 = RandomCliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
    p_intra=0.50,
    p_inter=0.30,
)

CXC3 = RandomCliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
    p_intra=0.50,
    p_inter=0.05,
)

CXC4 = RandomCliqueCrossClique(
    num_clusters=NUM_CLUSTERS,
    num_members=NUM_MEMBERS,
    num_timesteps=NUM_TIMESTEPS,
    p_intra=0.10,
    p_inter=0.01,
)

true_labels = np.concatenate([[i] * NUM_MEMBERS for i in range(NUM_CLUSTERS)])

for CXC_index, CXC in enumerate([CXC1, CXC2, CXC3, CXC4]):
    stgkm_scores = []
    dcdid_scores = []
    cc_scores = []
    sum_scores = []

    for _ in range(1):
        connectivity_matrix = CXC.create_clique_cross_clique()

        (
            stgkm_score,
            dcdid_score,
            cc_score,
            sum_score,
            stgkm_label_mat,
            dcdid_label_mat,
            cc_label_mat,
        ) = compare_performance(
            connectivity_matrix=connectivity_matrix,
            true_labels=true_labels,
            stgkm_args={"penalty": 10, "max_drift": 1, "drift_time_window": 1},
        )
        stgkm_scores.append(stgkm_score)
        cc_scores.append(cc_score)
        sum_scores.append(sum_score)
        dcdid_scores.append(dcdid_score)

    subtitles = [
        "Clique-cross-Clique",
        "Strong Random Clique-cross-Clique",
        "Mixed Random Clique-cross-Clique",
        "Weak Random Clique-cross-Clique",
    ]

    if PLOT:
        plot_cluster_history(
            weight_matrices={
                "STGkM": stgkm_label_mat,
                "CC": cc_label_mat,
                "DCDID": dcdid_label_mat,
            },
            subtitle=subtitles[CXC_index],
            filepath="CXC" + str(CXC_index),
        )

    print(
        "CXC",
        np.average(stgkm_scores),
        np.average(cc_scores),
        np.average(sum_scores),
        np.average(dcdid_scores),
    )

### Three Cluster Connectivity Matrix Experiment
TCC = ThreeClusterConnectivity(num_changes=30, pop_size=10)
true_labels = np.concatenate([[i] * 10 for i in range(3)])

stgkm_scores = []
dcdid_scores = []
cc_scores = []
sum_scores = []

for i in range(100):
    three_cluster_connectivity_matrix = TCC.create_three_cluster_connectivity_matrix()

    (
        stgkm_score,
        dcdid_score,
        cc_score,
        sum_score,
        stgkm_label_mat,
        dcdid_label_mat,
        cc_label_mat,
    ) = compare_performance(
        connectivity_matrix=three_cluster_connectivity_matrix,
        true_labels=true_labels,
        stgkm_args={"penalty": 5, "max_drift": 1, "drift_time_window": 1},
    )

    stgkm_scores.append(stgkm_score)
    dcdid_scores.append(dcdid_score)
    cc_scores.append(cc_score)
    sum_scores.append(sum_score)

if PLOT:
    plot_cluster_history(
        weight_matrices={
            "STGkM": stgkm_label_mat,
            "CC": cc_label_mat,
            "DCDID": dcdid_label_mat,
        },
        subtitle="Three Clusters",
        filepath="3Cluster",
        figsize=(23, 10),
        fig_text_x=0.8,
        fig_text_y=0.95,
    )

print(
    "3 cluster",
    np.average(stgkm_scores),
    np.average(cc_scores),
    np.average(sum_scores),
    np.average(dcdid_scores),
)

### Theseus Clique Experiment
num_members = 10
TC = TheseusClique(num_members=num_members, num_timesteps=50)
true_labels = np.concatenate([[i] * num_members for i in range(2)])

stgkm_scores = []
dcdid_scores = []
cc_scores = []
sum_scores = []

for i in range(100):
    theseus_connectivity_matrix = TC.create_theseus_clique()

    (
        stgkm_score,
        dcdid_score,
        cc_score,
        sum_score,
        stgkm_label_mat,
        dcdid_label_mat,
        cc_label_mat,
    ) = compare_performance(
        connectivity_matrix=theseus_connectivity_matrix,
        true_labels=true_labels,
        stgkm_args={"penalty": 5, "max_drift": 1, "drift_time_window": 1},
    )
    stgkm_scores.append(stgkm_score)
    dcdid_scores.append(dcdid_score)
    cc_scores.append(cc_score)
    sum_scores.append(sum_score)

if PLOT:
    plot_cluster_history(
        weight_matrices={
            "STGkM": stgkm_label_mat,
            "CC": cc_label_mat,
            "DCDID": dcdid_label_mat,
        },
        subtitle="Theseus Clique",
        filepath="Theseus",
        figsize=(30, 8),
    )
print(
    "Theseus",
    np.average(stgkm_scores),
    np.average(cc_scores),
    np.average(sum_scores),
    np.average(dcdid_scores),
)
