"""Script to get roll call votes from the House of Representatives website."""
import requests
import xmltodict
import pandas as pd
import numpy as np

final_data = {"time": [], "legislator": [], "party": [], "vote": [], "last_name": []}
for roll_call_id in range(100, 383):
    URL = "https://clerk.house.gov/evs/2023/roll" + str(roll_call_id) + ".xml"

    response = requests.get(URL, timeout=30)
    vote_data = xmltodict.parse(response.content)

    for voter in vote_data["rollcall-vote"]["vote-data"]["recorded-vote"]:
        final_data["time"].append(roll_call_id - 100)
        final_data["legislator"].append(voter["legislator"]["@name-id"])
        final_data["last_name"].append(voter["legislator"]["@sort-field"])
        final_data["party"].append(voter["legislator"]["@party"])
        final_data["vote"].append(voter["vote"])

voter_data = pd.DataFrame(data=final_data)
value_counts = (
    voter_data["legislator"]
    .value_counts()
    .rename_axis("legislator")
    .reset_index(name="counts")
)
relevant_voters = value_counts[value_counts["counts"] == 283]["legislator"]
final_voter_data = voter_data[voter_data["legislator"].isin(relevant_voters)]

voter_dictionary = dict(zip(final_voter_data["legislator"], np.arange(432)))

fvd_0 = final_voter_data[final_voter_data["time"] == 0]
fvd_0["legislator_id"] = [
    voter_dictionary[legislator] for legislator in fvd_0["legislator"]
]

connectivity_matrix = np.zeros((283, 432, 432))
for time in range(283):
    print("Processing vote #", time)

    curr_timeslice = final_voter_data[final_voter_data["time"] == time]

    yes_voters = curr_timeslice[curr_timeslice["vote"].isin(["Yea", "Aye"])][
        "legislator"
    ]
    no_voters = curr_timeslice[curr_timeslice["vote"].isin(["No", "Nay"])]["legislator"]
    present_voters = curr_timeslice[curr_timeslice["vote"] == "Present"]["legislator"]
    not_voting = curr_timeslice[curr_timeslice["vote"] == "Not Voting"]["legislator"]

    # If they vote the same, they are connected
    for voter_1 in yes_voters:
        for voter_2 in yes_voters:
            connectivity_matrix[
                time, voter_dictionary[voter_1], voter_dictionary[voter_2]
            ] = 1
            connectivity_matrix[
                time, voter_dictionary[voter_2], voter_dictionary[voter_1]
            ] = 1

    for voter_1 in no_voters:
        for voter_2 in no_voters:
            connectivity_matrix[
                time, voter_dictionary[voter_1], voter_dictionary[voter_2]
            ] = 1
            connectivity_matrix[
                time, voter_dictionary[voter_2], voter_dictionary[voter_1]
            ] = 1

    for voter_1 in present_voters:
        for voter_2 in present_voters:
            connectivity_matrix[
                time, voter_dictionary[voter_1], voter_dictionary[voter_2]
            ] = 1
            connectivity_matrix[
                time, voter_dictionary[voter_2], voter_dictionary[voter_1]
            ] = 1

    for voter_1 in not_voting:
        for voter_2 in not_voting:
            connectivity_matrix[
                time, voter_dictionary[voter_1], voter_dictionary[voter_2]
            ] = 1
            connectivity_matrix[
                time, voter_dictionary[voter_2], voter_dictionary[voter_1]
            ] = 1


final_voter_data.to_csv("final_voter_data_temp.csv")
fvd_0.to_csv("fvd_0_temp")
np.save("roll_call_connectivity_temp.npy", connectivity_matrix)
