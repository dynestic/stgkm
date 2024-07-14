import networkx as nx
import numpy as np
from DCDID1.DCDID import (
    node_addition,
    node_deletion,
    edge_addition,
    edge_deletion,
    getchangegraph,
    CDID,
    conver_comm_to_lab,
)
from typing import List, Mapping


def run_dcdid(connectivity_matrix: np.ndarray, verbose: bool = False) -> List[Mapping]:
    """
    Run DCDID.

    Inputs:
        connectivity_matrix (np.ndarray): Connectivity matrix txNxN
        verbose (bool): Whether or not to print messages for each time step processed and program finish.
    Returns:
        (List[Mapping]): List of length t containing a dictionary of labels for each time step. The dictionary has the form {cluster: [node]}.
    """
    all_labels = []
    edges_added = set()
    edges_removed = set()
    nodes_added = set()
    nodes_removed = set()

    G = nx.from_numpy_array(connectivity_matrix[0])

    # nx.draw_networkx(G)

    comm = {}
    comm = CDID(G, 0)

    initcomm = conver_comm_to_lab(comm)
    all_labels.append(initcomm)

    # getscore(comm_va,comm_list)
    # start=time.time()
    G1 = nx.Graph()
    G1 = G

    for i in range(1, len(connectivity_matrix)):
        if verbose is True:
            print("index", i)
        G2 = nx.from_numpy_array(connectivity_matrix[i])

        total_nodes = set(G1.nodes()) | set(G2.nodes())
        #    current_nodes.add(1002)
        #    previous_nodes.add(1001)

        nodes_added = set(G2.nodes()) - set(G1.nodes())
        nodes_removed = set(G1.nodes()) - set(G2.nodes())
        edges_added = set(G2.edges()) - set(G1.edges())
        edges_removed = set(G1.edges()) - set(G2.edges())
        all_change_comm = set()
        addn_ch_comm, addn_pro_edges, addn_commu = node_addition(G2, nodes_added, comm)
        edges_added = edges_added - addn_pro_edges  # 去掉已处理的边
        all_change_comm = all_change_comm | addn_ch_comm
        deln_ch_comm, deln_pro_edges, deln_commu = node_deletion(
            G1, nodes_removed, addn_commu
        )
        all_change_comm = all_change_comm | deln_ch_comm
        edges_removed = edges_removed - deln_pro_edges

        adde_ch_comm = edge_addition(edges_added, deln_commu)
        all_change_comm = all_change_comm | adde_ch_comm

        dele_ch_comm = edge_deletion(edges_removed, deln_commu)
        all_change_comm = all_change_comm | dele_ch_comm
        #    print('all_change_comm',all_change_comm)
        unchangecomm = ()  # 未改变的社区标签
        newcomm = {}  # 格式为｛节点：社区｝
        newcomm = deln_commu  # 添加边和删除边，只是在现有节点上处理，不会新增节点，删除节点（前面已处理）
        unchangecomm = set(newcomm.values()) - all_change_comm
        unchcommunity = {
            key: value for key, value in newcomm.items() if value in unchangecomm
        }  # 未改变的社区 ：标签和结点
        # 找出变化社区所对应的子图，然后对子图运用信息动力学找出新的社区结构，加上未改变的社区结构，得到新的社区结构。
        #    print('change community:',all_change_comm)
        Gtemp = nx.Graph()
        Gtemp = getchangegraph(all_change_comm, newcomm, G2)

        unchagecom_maxlabe = 0
        if len(unchangecomm) > 0:
            unchagecom_maxlabe = max(unchangecomm)
        #    print('subG',Gtemp.edges())
        if Gtemp.number_of_edges() < 1:  # 社区未发生变化
            comm = newcomm
        else:
            getnewcomm = CDID(Gtemp, unchagecom_maxlabe)
            #
            d = dict(unchcommunity)
            d.update(getnewcomm)

            comm = dict(d)  # 把当前获得的社区结构，作为下一次的社区输入

        fin_communities = conver_comm_to_lab(comm)
        all_labels.append(fin_communities)
        # print('getcommunity:',fin_communities.keys())
        # getscore(list(conver_comm_to_lab(comm).values()),comm_new)
        G1.clear()
        G1.add_edges_from(G2.edges())
        G2.clear()
    if verbose:
        print("all done")
    return all_labels


def convert_labels(
    connectivity_matrix: np.ndarray, all_labels: List[Mapping]
) -> np.ndarray:
    """
    Convert dictionary labels to a binary label matrix.

    Inputs:
        connectivity_matrix (np.ndarray): Connectivity matrix txNxN
        all_labels (List[Mapping]): List of length t containing a dictionary of labels for each time step. The dictionary has the form {cluster: [node]}.
    Returns:
        (np.ndarray): txnxk binary assignment matrix.
    """
    num_nodes = connectivity_matrix.shape[1]
    label_mat = np.zeros((connectivity_matrix.shape[0], num_nodes))
    for time in range(connectivity_matrix.shape[0]):
        curr_labels = all_labels[time]

        for key in curr_labels.keys():
            for rep in curr_labels[key]:
                label_mat[time, rep] = key

    return label_mat
