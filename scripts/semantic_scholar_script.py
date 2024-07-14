"""Run STGkM on Semantic Scholar data."""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stgkm.distance_functions import s_journey
from stgkm.helper_functions import agglomerative_clustering
from stgkm.stgkm_figures import (
    similarity_matrix_figure,
    choosing_num_clusters_plot,
)
from stgkm.helper_functions import choose_num_clusters


def retrieve_papers(year: int):
    """Retreive 1000 most highly cited papers for a given year."""

    # Define the paper search endpoint URL
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    frames = []
    for index in range(5):
        # Define the required query parameter and its value (in this case, the keyword we want to search for)
        query_params = {
            "query": "dynamic network",
            "limit": 100,
            "offset": index * 100,
            # "publicationDateOrYear": "2022-01-01:2022-01-31",
            "year": year,
            # "minCitationCount": 10,
            "sort": "citationCount:desc",
            "publicationTypes": "JournalArticle",
            "fields": "venue,publicationDate,authors,references.authors,references.venue",
        }

        # Directly define the API key (Reminder: Securely handle API keys in production environments)
        # api_key = "your api key goes here"  # Replace with the actual API key

        # Define headers with API key
        headers = {"x-api-key": "nRELYbPhEK8AYSDFRCZud1aJyjG3r904alKwt0e2"}

        # Make the GET request with the URL and query parameters
        searchResponse = requests.get(url, params=query_params, headers=headers)
        if searchResponse.status_code == 200:
            json_data = searchResponse.json()
            df = pd.DataFrame.from_dict(json_data["data"])
            frames.append(df)
        else:
            print(searchResponse.json())
            break
    if len(frames) > 0:
        fin_df = pd.concat(frames)

        fin_df["publicationDate"] = fin_df["publicationDate"].apply(
            lambda x: pd.to_datetime(x)
        )
        fin_df["month"] = fin_df["publicationDate"].dt.month
        fin_df["year"] = fin_df["publicationDate"].dt.year
        return fin_df
    else:
        print("no data generated for ", year)
        return None


def create_SS_dataframe():
    fin_dfs = []
    year_df = None
    count = 0
    for year in range(2000, 2024):
        while (year_df is None) and (count < 3):
            year_df = retrieve_papers(year=year)
            count += 1

            if year_df is None:
                print("Year ", year, "failed ", count, " times")

        if year_df is not None:
            fin_dfs.append(year_df)
            year_df = None
            count = 0

    final_df = pd.concat(fin_dfs)
    print('Dataframe saved to "semantic_scholar_df.pkl".')
    final_df.to_pickle(path="semantic_scholar_df.pkl")
    return final_df


def get_venues(df: pd.DataFrame):
    """
    Return venues that have at least 10 citations and are both cited and referenced.
    """
    venue_list = []
    ref_venue_list = []
    for _, row in df.iterrows():
        venue = row["venue"]
        if len(venue) > 0:
            for paper in row["references"]:
                ref_venue = paper["venue"]
                if len(ref_venue) > 0:
                    venue_list.append(venue)
                    ref_venue_list.append(ref_venue)

    vals, counts = np.unique(ref_venue_list, return_counts=True)
    ten_citations_ind = np.where(counts >= 50)[0]
    ten_citations = vals[ten_citations_ind]
    final_venues = np.intersect1d(venue_list, ten_citations)
    return final_venues


def create_SS_connectivity_matrix(final_df: pd.DataFrame):
    final_venues = get_venues(final_df)
    unique_years = sorted(final_df["year"].dropna().unique())

    connectivity_matrix = np.zeros(
        (len(unique_years), len(final_venues), len(final_venues))
    )
    weight_matrix = np.ones((len(unique_years), len(final_venues), len(final_venues)))

    venue_dict = dict(zip(final_venues, np.arange(len(final_venues))))
    venue_dict_reverse = dict(zip(np.arange(len(final_venues)), final_venues))
    year_dict = dict(zip(unique_years, np.arange(len(unique_years))))

    for _, row in final_df.iterrows():
        year = row["year"]
        if ~np.isnan(year):
            time = year_dict[year]
            venue = row["venue"]
            if venue in final_venues:
                for paper in row["references"]:
                    ref_venue = paper["venue"]
                    if ref_venue in final_venues:
                        venue_index = venue_dict[venue]
                        ref_venue_index = venue_dict[ref_venue]

                        connectivity_matrix[
                            int(time) - 1, venue_index, ref_venue_index
                        ] = 1
                        connectivity_matrix[
                            int(time) - 1, ref_venue_index, venue_index
                        ] = 1
                        weight_matrix[int(time) - 1, venue_index, ref_venue_index] += 1
                        weight_matrix[int(time) - 1, ref_venue_index, venue_index] += 1
    return connectivity_matrix, weight_matrix, venue_dict, venue_dict_reverse, year_dict


FINAL_DF = pd.read_pickle("loaded_data/SS_loaded_data/semantic_scholar_df.pkl")
CONNECTIVITY_MATRIX, WEIGHT_MATRIX, VENUE_DICT, VENUE_DICT_REVERSE, YEAR_DICT = (
    create_SS_connectivity_matrix(final_df=FINAL_DF)
)

RANDOM_STATE = np.random.choice(200, 1)[0]
MAX_DRIFT = 1
DRIFT_TIME_WINDOW = 1
START = 0
END = -10
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
PENALTY = len(YEAR_DICT) + END
STR_ID = str(MAX_DRIFT) + str(DRIFT_TIME_WINDOW) + str(START) + str(-END) + "_temp"

distance_matrix = s_journey(connectivity_matrix=CONNECTIVITY_MATRIX)
weighted_distance = 1 / WEIGHT_MATRIX * distance_matrix
penalized_distance = np.where(distance_matrix == np.inf, PENALTY, distance_matrix)
penalized_distance = 1 / WEIGHT_MATRIX * penalized_distance

obj_values, opt_k, label_history, medoid_history = choose_num_clusters(
    min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    penalized_distance=penalized_distance[START:END],
    connectivity_matrix=CONNECTIVITY_MATRIX[START:],
    random_state=RANDOM_STATE,
    max_drift=MAX_DRIFT,
    drift_time_window=DRIFT_TIME_WINDOW,
    max_iterations=100,
)

print("opt k:", opt_k)
opt_labels = label_history[np.argmax(obj_values)]
opt_ltc = agglomerative_clustering(weights=opt_labels.T, num_clusters=opt_k)
for i in range(opt_k):
    ind = np.where(opt_ltc == i)[0]
    print([VENUE_DICT_REVERSE[index] for index in ind], "\n\n")

similarity_matrix_figure(
    full_assignments=opt_labels,
    long_term_clusters=opt_ltc,
    fig_title="Semantic Scholar Data \n Short Term Cluster Similarity Scores k=%d"
    % opt_k,
    filepath="STGKM_Figures/SS_figures/SS_short_term_cluster_similarity_"
    + STR_ID
    + ".pdf",
)

choosing_num_clusters_plot(
    min_num_clusters=MIN_CLUSTERS,
    max_num_clusters=MAX_CLUSTERS,
    sum_distance_from_centers=obj_values,
    fig_title="Avg. Silhouette Score vs. Number of Clusters",
    ylabel="Avg. Silhouette Score",
    filepath="STGKM_Figures/SS_figures/SS_opt_num_clusters_" + STR_ID + ".pdf",
)
saved_labels = np.save("loaded_data/SS_loaded_data/opt_labels_" + STR_ID, opt_labels)
saved_medoids = np.save(
    "loaded_data/SS_loaded_data/opt_medoids_" + STR_ID,
    medoid_history[np.argmin(obj_values)],
)
saved_ltc = np.save("loaded_data/SS_loaded_data/opt_ltc_" + STR_ID, opt_ltc)


##### Plot "Purity"
print("\n\n\n")
purity = []
for col in range(199):
    num_classifications = len(np.unique(opt_labels[:, col]))
    purity.append(num_classifications)

_, axs = plt.subplots(figsize=(20, 20))
axs.hist(purity)
axs.set_title("Number of Clusters to which a Journal Belongs over Time", fontsize=40)
axs.set_ylabel("Count", fontsize=100)
axs.axvline(np.average(purity), color="r", linestyle="dashed", linewidth=4)
axs.set_xlabel("Number of Clusters", fontsize=50)
axs.tick_params(labelsize=30)
plt.savefig(
    "STGKM_Figures/SS_figures/cluster_membership_hist" + STR_ID + ".pdf", format="pdf"
)
