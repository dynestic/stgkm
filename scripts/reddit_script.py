"""Script for running experiments on the Reddit dataset."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from stgkm.distance_functions import s_journey
from stgkm.helper_functions import choose_num_clusters
from stgkm.helper_functions import agglomerative_clustering
from stgkm.stgkm_figures import (
    similarity_matrix_figure,
    choosing_num_clusters_plot,
)


def get_reddit_df(
    out_filepath: os.PathLike, sentiment: str, in_filepath: Optional[os.PathLike] = None
):
    """
    Get the reddit dataframe. Create it if it doesn't exist. Load it if it does. Output the desired sentiment dataframe.
    """
    assert sentiment in [
        "positive",
        "negative",
    ], "Sentiment can only be positive or negative."

    if not os.path.isfile(out_filepath):
        assert (
            in_filepath is not None
        ), "If out_filepath does not exist, in_filepath must be provided."
        df = pd.read_csv(in_filepath, sep="\t")
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(lambda x: pd.to_datetime(str(x)))
        df["DATE"] = df["TIMESTAMP"].dt.date
        unique_dates = np.sort(df["DATE"].unique())
        date_dict = dict(zip(unique_dates, np.arange(len(unique_dates))))
        df["TIMESTAMP_INT"] = df["DATE"].apply(lambda x: date_dict[x])
        df.to_pickle(out_filepath)
    else:
        df = pd.read_pickle(out_filepath)

    if sentiment == "positive":
        fin_df = df[df["LINK_SENTIMENT"] == 1]
    elif sentiment == "negative":
        fin_df = df[df["LINK_SENTIMENT"] == -1]
    fin_df["DATE"] = fin_df["TIMESTAMP"].dt.to_period("M")
    return fin_df


def load_reddit_connectivity_matrix(out_folder: str, sentiment: str):
    """
    Load reddit connectivity matrix and connection weights.
    """
    if sentiment == "positive":
        connectivity_matrix = np.load(out_folder + "/reddit_connectivity_positive.npy")
        weight_matrix = np.load(out_folder + "/reddit_weight_positive.npy")
    elif sentiment == "negative":
        connectivity_matrix = np.load(out_folder + "/reddit_connectivity_negative.npy")
        weight_matrix = np.load(out_folder + "/reddit_weight_negative.npy")
    return connectivity_matrix, weight_matrix


def create_reddit_connectivity_matrix(
    reddit_df, subreddits, sentiment: str, out_folder: str
):
    """
    Create the connectivity matrix and the weight matrix.
    """
    unique_dates = np.sort(reddit_df["DATE"].unique())
    date_dict = dict(zip(unique_dates, np.arange(len(unique_dates))))
    timestamps = len(unique_dates)
    subreddit_dict = dict(zip(subreddits, np.arange(len(subreddits))))

    connectivity_matrix = np.zeros((timestamps, len(subreddits), len(subreddits)))
    weight_matrix = np.ones((timestamps, len(subreddits), len(subreddits)))

    for _, row in reddit_df.iterrows():
        source = row["SOURCE_SUBREDDIT"]
        target = row["TARGET_SUBREDDIT"]

        if (source in subreddits) & (target in subreddits):
            time = date_dict[row["DATE"]]
            Va = subreddit_dict[source]
            Vb = subreddit_dict[target]
            connectivity_matrix[time, Va, Vb] = 1
            connectivity_matrix[time, Vb, Va] = 1
            weight_matrix[time, Va, Vb] += 1
            weight_matrix[time, Vb, Va] += 1
    ##########################
    np.save(
        out_folder + "/reddit_connectivity_" + sentiment + ".npy", connectivity_matrix
    )
    np.save(out_folder + "/reddit_weight_" + sentiment + ".npy", weight_matrix)
    print("matrices created")
    return connectivity_matrix, weight_matrix


def get_reddit_connectivity_matrix(sentiment: str, fin_df, out_folder):
    "Get the reddit connectivity_matrix. Create it if it doesn't exist. Load it if it does."

    ########### EXTRACT SOURCE TARGET PAIRS ####################
    source_vc = fin_df["SOURCE_SUBREDDIT"].value_counts()
    source_reddits = source_vc[source_vc.values > 20].index.values

    source_target_pairs = fin_df[fin_df["SOURCE_SUBREDDIT"].isin(source_reddits)][
        "TARGET_SUBREDDIT"
    ]

    target_vc = fin_df["TARGET_SUBREDDIT"].value_counts()
    target_reddits = target_vc[target_vc.values > 20].index.values

    source_target_threshold = np.intersect1d(source_target_pairs, target_reddits)

    subreddits = np.intersect1d(source_reddits, source_target_threshold)

    index_to_subreddit = dict(zip(np.arange(len(subreddits)), subreddits))

    ##########################################################
    if not os.path.isfile(
        out_folder + "/reddit_connectivity_" + sentiment + ".npy",
    ):
        connectivity_matrix, weight_matrix = create_reddit_connectivity_matrix(
            sentiment=sentiment,
            reddit_df=fin_df,
            subreddits=subreddits,
            out_folder="Reddit_loaded_data",
        )
    else:
        connectivity_matrix, weight_matrix = load_reddit_connectivity_matrix(
            out_folder, sentiment=sentiment
        )
    return connectivity_matrix, weight_matrix, index_to_subreddit


# in_filepath = "data/soc-redditHyperlinks-title.tsv"
SENTIMENT = "positive"
OUT_FILEPATH = "loaded_data/Reddit_loaded_data/reddit_df.pkl"
OUT_FOLDER = "loaded_data/Reddit_loaded_data"

FIN_DF = get_reddit_df(out_filepath=OUT_FILEPATH, sentiment=SENTIMENT)
CONNECTIVITY_MATRIX, WEIGHT_MATRIX, INDEX_TO_SUBREDDIT = get_reddit_connectivity_matrix(
    sentiment=SENTIMENT, fin_df=FIN_DF, out_folder=OUT_FOLDER
)

# # # calculate s_journey
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
START = 0
END = -5
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
STR_ID = (
    "_"
    + SENTIMENT
    + "_"
    + str(MAX_DRIFT)
    + str(DRIFT_TIME_WINDOW)
    + str(START)
    + str(-END)
    + "_TEMP"
)

t, num_vertices, _ = CONNECTIVITY_MATRIX.shape
PENALTY = t + END

distance_matrix = s_journey(CONNECTIVITY_MATRIX)
penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)
penalized_distance = 1 / WEIGHT_MATRIX * penalized_distance

random_state = np.random.choice(100, 1)[0]

obj_values, opt_k, label_history, medoid_history = choose_num_clusters(
    min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    penalized_distance=penalized_distance[START:END],
    connectivity_matrix=CONNECTIVITY_MATRIX[START:],
    random_state=random_state,
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    max_iterations=100,
)

print(opt_k)
opt_labels = label_history[np.argmax(obj_values)]
opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=opt_k)

similarity_matrix_figure(
    full_assignments=opt_labels,
    long_term_clusters=opt_ltc,
    fig_title="Negative Sentiment Reddit Data \n Short Term Cluster Similarity Scores k=%d"
    % opt_k,
    filepath="reddit_sentiment" + STR_ID + ".pdf",
)

choosing_num_clusters_plot(
    min_num_clusters=MIN_CLUSTERS,
    max_num_clusters=MAX_CLUSTERS,
    sum_distance_from_centers=obj_values,
    fig_title="Avg. Silhouette Score vs. Number of Clusters",
    ylabel="Avg. Silhouette Score",
    filepath="STGKM_Figures/Reddit_opt_num_clusters" + STR_ID + ".pdf",
)

########### PURITY HISTOGRAM
purity = []
for col in range(41):
    num_classifications = len(np.unique(opt_labels[:, col]))
    purity.append(num_classifications)

_, axs = plt.subplots(figsize=(20, 20))
axs.hist(purity)
axs.set_title(
    "Number of Clusters to which \n a Negative Subreddit Belongs over Time", fontsize=50
)
axs.set_ylabel("Count", fontsize=50)
axs.axvline(np.average(purity), color="r", linestyle="dashed", linewidth=4)
axs.set_xlabel("Number of Clusters", fontsize=50)
axs.tick_params(labelsize=30)
plt.savefig(
    "STGKM_Figures/Reddit_cluster_membership_hist" + STR_ID + ".pdf", format="pdf"
)

###############################################
# Long term cluster membership
for i in range(opt_k):
    mems = np.where(opt_ltc == i)[0]
    print([INDEX_TO_SUBREDDIT[index] for index in mems])

# Most popular (how often they appear over time) topics in each short term cluster
for i in range(opt_k):
    member_ind = np.where(opt_labels == i)[1]
    num_clustered = len(member_ind)
    print(num_clustered)

    vals, counts = np.unique(member_ind, return_counts=True)
    arg_sorted_counts = np.argsort(counts)[::-1]
    sorted_counts = np.sort(counts)[::-1]
    print(sorted_counts)
    print([INDEX_TO_SUBREDDIT[index] for index in vals[arg_sorted_counts]])

saved_labels = np.save(
    "loaded_data/Reddit_loaded_data/Reddit_opt_labels_" + STR_ID, opt_labels
)
saved_medoids = np.save(
    "loaded_data/Reddit_loaded_data/Reddit_opt_medoids" + STR_ID,
    medoid_history[np.argmin(obj_values)],
)
saved_ltc = np.save("loaded_data/Reddit_loaded_data/Reddit_opt_ltc_" + STR_ID, opt_ltc)
