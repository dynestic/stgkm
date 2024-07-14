"""Generate figures for STGKM experiments."""

import pickle
import os
from typing import List, Optional, Mapping, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from stgkm.helper_functions import similarity_matrix, similarity_measure


def visualize_graph(
    connectivity_matrix: np.ndarray,
    labels: Optional[List] = None,
    centers: Optional[List] = None,
    color_dict: Optional[Mapping] = None,
    figsize: Tuple[int] = (10, 10),
):
    """
    Visualize the dynamic graph at each time step.

    Inputs:
        connectivity_matrix (np.ndarray): txnxn connectivity matrix
        labels (Optional[List]): Optional List containing either short-term or long-term cluster labels
        centers (Optional[List]): Optional List containing cluster centers
        color_dict (Optional[Mapping]): Optional dictionary for node colors of form {label: color}
        figsize (Tuple[int]): Optional tuple for figure size.
    """
    timesteps, num_vertices, _ = connectivity_matrix.shape

    if labels is None:
        labels = []
    if centers is None:
        centers = []
    if color_dict is None:
        color_dict = {
            0: "tab:blue",
            1: "tab:orange",
            2: "tab:green",
            3: "tab:red",
            4: "tab:purple",
            5: "tab:brown",
            6: "tab:pink",
            7: "tab:gray",
            8: "tab:olive",
            9: "tab:cyan",
        }

    if len(np.unique(labels)) > len(color_dict):
        raise Exception("Color dictionary requires more than 10 keys/values")

    # Set layout for figures
    g_0 = nx.Graph(connectivity_matrix[0])
    g_0.remove_edges_from(nx.selfloop_edges(g_0))
    pos = nx.spring_layout(g_0)

    for time in range(timesteps):
        plt.figure(figsize=figsize)
        # No labels
        if len(labels) == 0:
            nx.draw(nx.Graph(connectivity_matrix[time]), pos=pos, with_labels=True)
        # Static long term labels
        elif len(labels) == num_vertices:
            graph = nx.Graph(connectivity_matrix[time])
            graph.remove_edges_from(nx.selfloop_edges(graph))
            nx.draw(
                graph,
                pos=pos,
                node_color=[color_dict[label] for label in labels],
                with_labels=True,
            )
        # Changing labels at each time step
        elif len(labels) == timesteps:
            if len(centers) != 0:
                center_size = np.ones(num_vertices) * 300
                center_size[centers[time].astype(int)] = 500
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    node_size=center_size,
                    with_labels=True,
                )
            else:
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    with_labels=True,
                )

        plt.show()

def choosing_num_clusters_plot(
    min_num_clusters: int,
    max_num_clusters: int,
    sum_distance_from_centers: List[float],
    fig_title: str,
    ylabel: str,
    filepath: str,
):
    """
    Create figure tracking objective value vs number of clusters.

    Args:
        min_num_clusters (int): Minimum number of clusters
        max_num_clusters (int): Maximum number of clusters
        sum_distance_from_centers (List[float]): Sum of the vertex distance from centers
    """

    plt.figure(figsize=(10, 5))
    plt.plot(range(min_num_clusters, max_num_clusters), sum_distance_from_centers)
    plt.xlabel("Number of Clusters k", size=20)
    plt.tick_params(labelsize=16)
    plt.ylabel(ylabel, size=20)
    plt.title(fig_title, size=20)

    plt.scatter(
        np.argmax(sum_distance_from_centers) + min_num_clusters,
        np.max(sum_distance_from_centers),
        marker="o",
        facecolors="none",
        edgecolors="r",
        s=500,
        linewidths=2,
        label="Optimal Value",
    )
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath, format="pdf")

    return None


def three_snapshots_dynamic_clustering(
    connectivity_matrix: np.ndarray,
    timesteps: List[int],
    num_clusters: int,
    membership: np.ndarray,
    centers: np.ndarray,
    fig_title: str,
    snapshot_title: str,
    filepath: str,
    color_dict: Optional[dict] = None,
    pkl_path: Optional[str] = None,
):
    """
    Show three snapshots of dynamic clusters in a dynamic graph.

    Args:
        connectivity_matrix (np.ndarray): Dynamic graph connectivity matrix
        timestpes (List[int]): Three timestpes to visualize
        membership (np.ndarray): Cluster membership history
        centers (np.ndarray): Cluster center history
        fig_title (str): Title for figure
        snapshot_title (str): Title for subfigure
        filepath (str): Filepath at which to save the figure
        color_dict (Optional[dict]): Color dictionary to use in the figures
        pos (Optional[str]) : Path to pkl file of saved positions for nodes in dynamic graph.
            If not provided, positions are generated from the random locations at time zero.
    """

    if color_dict is None:
        color_dict = {0: "dodgerblue", 1: "red", 2: "limegreen", 3: "green", -1: "cyan"}

    assert len(timesteps) == 3, "Can only visualize three time steps."

    _, _, num_points = connectivity_matrix.shape
    num_timesteps, _ = centers.shape
    # num_timesteps, num_clusters = centers.shape

    # Set layout for graph in visualization
    init_graph = nx.Graph(connectivity_matrix[0])
    init_graph.remove_edges_from(nx.selfloop_edges(init_graph))
    if pkl_path is None:
        pos = nx.spring_layout(init_graph)
    else:
        with open(pkl_path, "rb") as f:
            pos = pickle.load(f)

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
    fig.suptitle(fig_title, fontsize=100)

    for index, time in enumerate(timesteps):
        if membership.shape == (num_timesteps * num_clusters, num_points):
            time_labels = np.argmax(
                membership[time * num_clusters : time * (num_clusters) + num_clusters],
                axis=0,
            )
        elif membership.shape == (num_timesteps, num_points):
            time_labels = membership[time]
        center_size = np.ones(num_points) * 200
        center_size[centers[time].astype(int)] = 1000
        final_graph = nx.Graph(connectivity_matrix[time])
        final_graph.remove_edges_from(nx.selfloop_edges(final_graph))
        nx.draw(
            final_graph,
            with_labels=False,
            node_size=center_size,
            node_color=[color_dict[label] for label in time_labels],
            pos=pos,
            ax=axs[index % 3],
        )
        axs[index % 3].set_title("%s %i" % (snapshot_title, timesteps[index]), size=100)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, format="pdf")
    plt.show()

    return fig


def community_matrix_figure(
    clusters: np.ndarray,
    matrix: np.ndarray,
):
    """Return matrix segmented by communities."""

    communities = []
    for unique_label in np.unique(clusters):
        communities.append(np.where(clusters == unique_label)[0])

    sorted_communities = sorted(communities, key=lambda x: len(x), reverse=True)
    cols = []
    for community_1 in sorted_communities:
        col = []
        for community_2 in sorted_communities:
            col_entry = matrix[community_1, :][:, community_2]
            col.append(col_entry)
        cols.append(np.hstack(col))

    reordered_mat = np.vstack(cols)
    return reordered_mat


def similarity_matrix_figure(
    full_assignments: np.ndarray,
    long_term_clusters: np.ndarray,
    fig_title: str,
    filepath: str,
):
    """Make similarity matrix figure."""

    sim_mat = similarity_matrix(
        weights=full_assignments.T, similarity_function=similarity_measure
    )

    reordered_mat = community_matrix_figure(clusters=long_term_clusters, matrix=sim_mat)

    _, axs = plt.subplots(figsize=(20, 20))
    axs.axis("off")
    plt.imshow(reordered_mat, cmap="viridis")
    cbar = plt.colorbar(pad=0.01, ticks=np.arange(0.2, 1.1, 0.1))
    cbar.ax.tick_params(labelsize=30)
    plt.title(fig_title, size=40, y=1.05)
    plt.savefig(filepath, format="pdf")
    plt.show()

    return None


def plot_cluster_history(
    weight_matrices: Mapping,
    subtitle: str,
    filepath: os.PathLike,
    figsize: Optional[Tuple[int]] = (30, 10),
    fig_text_x: Optional[float] = 0.85,
    fig_text_y: Optional[float] = 0.9,
):
    """
    Plot cluster assignment histories for STGkM, CC, and DCDID.

    Inputs:
        weight_matrices (Mapping): Dictionary containing binary weight assignment matrices for various algorithms.
        subtitle (str): Subtitle for each subfigure title.
        filepath (os.PathLike): Path to which to save final figure
        figsize (Optional[Tuple[int]]): Figure size.
        fig_text_x (Optional[float]): x location of legend
        fig_text_y (Optional[float]): y location of legend
    """
    titles = weight_matrices.keys()
    _, axs = plt.subplots(1, 3, figsize=figsize)
    axs = axs.flatten()
    for index, title in enumerate(titles):
        axs[index].matshow(weight_matrices[title].T)
        axs[index].set_ylabel("Node ID", fontsize=40)
        axs[index].set_xlabel("Time", fontsize=40)
        axs[index].set_title(title, fontsize=40)
        axs[index].xaxis.set_ticks_position("bottom")
        axs[index].tick_params(axis="both", which="major", labelsize=30)
    plt.suptitle(
        subtitle + "\n Cluster Assignment Histories",
        fontsize=40,
    )
    plt.figtext(
        fig_text_x,
        fig_text_y,
        "*Colors correspond to different clusters",
        ha="center",
        fontsize=30,
        bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout()
    plt.savefig("SyntheticEvolution/" + filepath + ".pdf", format="pdf")
    plt.show()

    return None


def plot_expectation_heatmap(filepath: os.PathLike, expectations: np.ndarray):
    """
    Plot heatmap of expectations of distances between nodes in a clique-cross-clique.

    Inputs:
        filepath (os.PathLike): Filepath to which to save the figure.
        expectations (np.ndarray): txnxn array containing expectations.
    """
    fig, axs = plt.subplots(1, 3, figsize=(100, 30))
    axs = axs.flatten()
    plot_num = [0, 3, 6]
    for i, ax in enumerate(axs):
        sns.heatmap(
            expectations[plot_num[i]],
            annot=False,
            fmt=".2f",
            cmap="viridis",
            ax=ax,
            cbar=True,
            vmin=0,
            vmax=5,
        )
        ax.set_xlabel("Node 1", fontsize=80)
        ax.set_ylabel("Node 2", fontsize=80)
        ax.set_title(f"Time = {plot_num[i]}", fontsize=80)
        ax.tick_params(axis="both", which="major", labelsize=80)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=80)

    plt.suptitle(
        "Expectation of s-journey distances between every pair of vertices over 1000 simulations",
        fontsize=100,
    )
    fig.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filepath, format="pdf")
    plt.show()
    return None


def plot_sensitivity_figure(filepath: os.PathLike):
    """
    Plot sensitivity of STGkM for finding ground truth clusters on clique-cross-cliques with various temporal and spatial complexities and structures.

    Inputs:
        filepath: (os.PathLike) Path to which save figure
    """
    _, axes = plt.subplots(1, 4, figsize=(16, 12), sharey=True)
    min_val = 30
    max_val = 200
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Create scatter plots for each subplot
    filepaths = [
        "sensitivity_intra" + str(p[0]) + "_inter" + str(p[1]) + ".pkl"
        for p in [[None, None], [5, 3], [5, "05"], [1, "01"]]
    ]
    p_list = [None, 0.50, 0.50, 0.10]
    p_prime_list = [None, 0.30, 0.05, 0.01]

    titles = [
        "Strong Random \n Clique-cross-Clique",
        "Mixed Random \n Clique-cross-Clique",
        "Weak Random \n Clique-cross-Clique",
    ]

    for i, ax in enumerate(axes):
        with open(filepaths[i], "rb") as file:
            loaded_data = pickle.load(file)

        df = pd.DataFrame.from_dict(data=loaded_data)
        df = df[df["num_timesteps"] < 20]
        df.rename(
            {
                "num_timesteps": "Number of time steps",
                "ami_score": "AMI score",
                "num_clusters": "Number of clusters",
                "num_members": "Cluster size",
            },
            axis=1,
            inplace=True,
        )

        if i == 3:
            legend_cond = True
        else:
            legend_cond = False
        sns.scatterplot(
            data=df,
            x="Number of time steps",
            y="AMI score",
            size="Cluster size",
            hue="Number of clusters",
            alpha=0.5,
            sizes=(min_val, max_val),
            palette="colorblind",
            ax=ax,
            legend=legend_cond,
        )
        if legend_cond:
            ax.legend(fontsize=12)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        if i == 0:
            ax.set_title("Clique-cross-Clique", fontsize=16)
        else:
            ax.set_title(
                titles[i - 1] + f"\n p = {p_list[i]}, p' = {p_prime_list[i]}",
                fontsize=16,
            )
        ax.tick_params(axis="both", which="major", labelsize=12)  # Major ticks

        if (i != 0) and (i != 4):
            ax.set_ylabel(None)
        else:
            ax.set_ylabel("AMI score", fontsize=16)

        ax.set_xlabel("Number of time steps", fontsize=16)

    plt.suptitle(
        "Sensitivity of AMI Score for STGkM run on Standard and Random Clique Cross Cliques",
        fontsize=20,
    )  # Adjust layout to prevent overlapping
    plt.tight_layout()

    plt.savefig(filepath, format="pdf")
    plt.show()

    return None
