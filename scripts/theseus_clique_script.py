"""Visualize theseus clique."""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stgkm.synthetic_graphs import TheseusClique

NUM_MEMBERS = 5
TC = TheseusClique(num_members=NUM_MEMBERS, num_timesteps=100)
_ = TC.create_theseus_clique()

#### CREATE FIGURE
labels = [[i] * NUM_MEMBERS for i in range(2)]
labels = np.concatenate(labels)
color_dict = {0: "dodgerblue", 1: "red"}
fig, axs = plt.subplots(2, 5, figsize=(40, 15))
axs = axs.flatten()
for time, ax in enumerate(axs):
    graph = nx.from_numpy_array(TC.connectivity_matrix[time])
    pos = nx.spring_layout(graph, k=0.9)
    nx.draw(
        graph,
        nodelist=np.arange(NUM_MEMBERS * 2),
        node_color=[color_dict[label] for label in labels],
        node_size=2000,
        pos=pos,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=dict(zip(np.arange(10), np.arange(10))),
        font_size=40,
        ax=ax,
    )
    ax.set_title("Timestep %d" % time, fontsize=40)

for i in range(1, 5):
    plt.plot(
        [i / 5, i / 5],
        [0, 1],
        color="k",
        lw=5,
        transform=plt.gcf().transFigure,
        clip_on=False,
    )
plt.plot(
    [0, 1],
    [0.5, 0.5],
    color="k",
    lw=5,
    transform=plt.gcf().transFigure,
    clip_on=False,
)
fig.tight_layout()
plt.savefig("theseus_clique_pattern.pdf", format="pdf")
plt.show()
