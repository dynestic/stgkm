# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:02:42 2018

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:41:26 2018

@author: Administrator
"""
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
import time
import datetime
from itertools import count
from sklearn import metrics
import math
import matplotlib.pyplot as plt  # 画图用


def str_to_int(x):
    return [[int(v) for v in line.split()] for line in x]


def node_addition(
    G, addnodes, communitys
):  # 输入的communitys社区格式为｛节点：社区名称｝
    change_comm = set()  # 存放结构可能发现改变的社区标签
    processed_edges = set()  # 已处理过的边，需要从增加边中删除已处理过的边

    for u in addnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # 邻居所在社区标签
        pc = set()
        for v in neighbors_u:
            neig_comm.add(communitys[v])
            pc.add((u, v))
            pc.add((v, u))  # 无向图中都 是一条边，加两次方便操作
        if len(neig_comm) > 1:  # 说明此加入结点不在社区内部
            change_comm = change_comm | neig_comm
            lab = max(communitys.values()) + 1
            communitys.setdefault(u, lab)  # 为v分配一个社区标签
            change_comm.add(lab)
        else:
            if len(neig_comm) == 1:  # 说明结点在社区内部，或只与一个社区连接
                communitys.setdefault(v, neig_comm[0])  # 将结点加入到本社区
                processed_edges = processed_edges | pc
            else:
                communitys.setdefault(
                    v, max(communitys.values()) + 1
                )  # 新加结点未和其它结点有连接，分配新的社区标签

    return (
        change_comm,
        processed_edges,
        communitys,
    )  # 返回可能发生变化的社区，处理过的边和最新社区结构。


def node_deletion(G, delnodes, communitys):  # tested, correct
    change_comm = set()  # 存放结构可能发现改变的社区标签
    processed_edges = set()  # 已处理过的边，需要从增加边中删除已处理过的边
    for u in delnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # 邻居所在社区标签
        for v in neighbors_u:
            neig_comm.add(communitys[v])
            processed_edges.add((u, v))
            processed_edges.add((v, u))
        del communitys[u]  # 删除结点和社区
        change_comm = change_comm | neig_comm
    return (
        change_comm,
        processed_edges,
        communitys,
    )  # 返回可能发生变化的社区，处理过的边和最新社区结构。


def edge_addition(
    addedges, communitys
):  # 如果加入边在社区内部，不会引起社区变化则不做处理，否则标记
    change_comm = set()  # 存放结构可能发现改变的社区标签
    #    print addedges
    #    print communitys
    for item in addedges:
        neig_comm = set()  # 邻居所在社区标签
        neig_comm.add(communitys[item[0]])  # 判断一边两端的节点所在社区
        neig_comm.add(communitys[item[1]])
        if len(neig_comm) > 1:  # 说明此加入边不在社区内部
            change_comm = change_comm | neig_comm
    return change_comm  # 返回可能发生变化的社区，


def edge_deletion(
    deledges, communitys
):  # 如果删除边在社区内部可能引起社区变化，在社区外部则不会变化
    change_comm = set()  # 存放结构可能发现改变的社区标签
    for item in deledges:
        neig_comm = set()  # 邻居所在社区标签
        neig_comm.add(communitys[item[0]])  # 判断一边两端的节点所在社区
        neig_comm.add(communitys[item[1]])
        if len(neig_comm) == 1:  # 说明此加入边不在社区内部
            change_comm = change_comm | neig_comm
    return change_comm  # 返回可能发生变化的社区


def getchangegraph(all_change_comm, newcomm, Gt):
    Gte = nx.Graph()
    com_key = newcomm.keys()
    for v in Gt.nodes():
        if v not in com_key or newcomm[v] in all_change_comm:
            Gte.add_node(v)
            neig_v = Gt.neighbors(v)
            for u in neig_v:
                if u not in com_key or newcomm[u] in all_change_comm:
                    Gte.add_edge(v, u)
                    Gte.add_node(u)

    return Gte


def CDID(
    Gsub, maxlabel
):  # G_sub为子图,对可能改变结构的子图运行信息动力学,maxlabel为未改变社区结构的最大标签

    # initial information
    Neigb = {}
    info = 0
    # 平均度、最大度
    avg_d = 0
    max_deg = 0
    N = Gsub.number_of_nodes()
    deg = dict(Gsub.degree())
    max_deg = max(deg.values())
    avg_d = sum(deg.values()) * 1.0 / N

    ti = 1
    list_I = {}  # 存放各节点信息，初始为各节点度，每次迭代不断动态变化
    maxinfo = 0
    starttime = datetime.datetime.now()
    for v in Gsub.nodes():
        if deg[v] == max_deg:
            info_t = 1 + ti * 0
            ti = ti + 1
            #            print v,max_deg,info_t
            maxinfo = info_t
        else:
            info_t = deg[v] * 1.0 / max_deg
            # info_t=round(random.uniform(0,1),3)
        #    info_t=deg[v]*1.0/max_deg
        list_I.setdefault(v, info_t)
        Neigb.setdefault(v, [n for n in Gsub.neighbors(v)])  # 节点v的邻居结点
        info += info_t
    node_order = sorted(list_I.items(), key=lambda t: t[1], reverse=True)
    node_order_list = list(zip(*node_order))[0]

    # 计算节点间相似度， 杰卡德系数
    def sim_jkd(u, v):
        list_v = [n for n in Gsub.neighbors(v)]
        list_v.append(v)
        list_u = [n for n in Gsub.neighbors(u)]
        list_u.append(u)
        t = set(list_v)
        s = set(list_u)

        return len(s & t) * 1.0 / len(s | t)

    # 计算节点间hop2数
    def hop2(u, v):
        list_v = Neigb[v]
        list_u = Neigb[u]
        t = set(list_v)
        s = set(list_u)
        return len(s & t)

    st = {}  # 存放相似度
    hops = {}  # 存放hop2数
    hop2v = {}  # 存放hop2数比值
    sum_s = {}  # 存放各节点邻居相似度之和
    avg_sn = {}  # 存放各节点的局部平均相似度，局部指的是在邻居节点
    avg_dn = {}  # 存放各节点的局部平均度

    for v, Iv in list_I.items():
        sum_v = 0
        sum_deg = 0
        tri = nx.triangles(Gsub, v) * 1.0
        listv = Neigb[v]
        num_v = len(listv)
        sum_deg += deg[v]

        for u in listv:
            keys = str(v) + "_" + str(u)
            p = st.setdefault(keys, sim_jkd(v, u))
            h2 = hop2(v, u)
            hops.setdefault(keys, h2)
            if tri == 0:
                if deg[v] == 1:
                    hop2v.setdefault(keys, 1)
                else:
                    hop2v.setdefault(keys, 0)
            else:
                hop2v.setdefault(keys, h2 / tri)

            sum_v += p
            sum_deg += deg[u]

        sum_s.setdefault(v, sum_v)
        avg_sn.setdefault(v, sum_v * 1.0 / num_v)
        avg_dn.setdefault(v, sum_deg * 1.0 / (num_v + 1))
    #    print('begin loop')

    #    oldinfo = 0
    info = 0
    t = 0
    while 1:
        info = 0
        t = t + 1
        Imax = 0

        for i in range(len(node_order_list)):
            v = node_order_list[i]
            Iv = list_I[v]
            for u in Neigb[v]:
                # p=sim_jkd(v,u)
                keys = str(v) + "_" + str(u)

                Iu = list_I[u]
                if Iu - Iv < 0:
                    #                           It=It*1.0/E
                    It = 0
                else:
                    It = math.exp(Iu - Iv) - 1
                # It=It*1.0*deg[u]/(deg[v]+deg[u])
                if It < 0.0001:
                    It = 0  #
                fuv = It
                #                       print(fuv)
                p = st[keys]
                p1 = p * hop2v[keys]
                Iin = p1 * fuv  #
                Icost = avg_sn[v] * fuv * (1 - p) / avg_dn[v]
                #                Icost=avg_s*fuv*avg_c/avg_d
                #                Icost=(avg_sn[v])*fuv/avg_dn[v]

                Iin = Iin - Icost
                if Iin < 0:
                    Iin = 0
                Iv = Iv + Iin
                #                       print(v,u,Iin,Icost,Iv,Iu,It)
                if Iin > Imax:
                    Imax = Iin

            if Iv > maxinfo:
                Iv = maxinfo
            list_I[v] = Iv
            # print(v,u,Iin,Iv,Iu,tempu[0],pu,tempu[1],fuv)
            info += list_I[v]
        # if v==3:
        #                print(v,Iv)

        if Imax < 0.0001:
            break

    endtime = datetime.datetime.now()
    #    print ('time:', (endtime - starttime).seconds)
    # 社团划分**************************************************************

    queue = []
    order = []
    community = {}
    lab = maxlabel
    number = 0
    for v, Info in list_I.items():
        if v not in community.keys():
            lab = lab + 1
            queue.append(v)
            order.append(v)
            community.setdefault(v, lab)
            number = number + 1
            while len(queue) > 0:
                node = queue.pop(0)
                for n1 in Neigb[node]:
                    if (not n1 in community.keys()) and (not n1 in queue):
                        if abs(list_I[n1] - list_I[node]) < 0.001:
                            queue.append(n1)
                            order.append(n1)
                            community.setdefault(n1, lab)
                            number = number + 1
        if number == N:
            break

            #    print (order)
            #    print(community)
    order_value = [community[k] for k in sorted(community.keys())]
    commu_num = len(set(order_value))  # 社团数量
    endtime1 = datetime.datetime.now()
    # print('社团划分结束')
    # print(list_I)
    # print('community number:', commu_num)
    # print('alltime:', (endtime1 - starttime).seconds)
    return community


def conver_comm_to_lab(comm1):  # 转换社区格式为，标签为主键，节点为value
    overl_community = {}
    for node_v, com_lab in comm1.items():
        if com_lab in overl_community.keys():
            overl_community[com_lab].append(node_v)
        else:
            overl_community.update({com_lab: [node_v]})
    return overl_community


def getscore(comm_va, comm_list):
    actual = []
    baseline = []
    for j in range(len(comm_va)):  # groundtruth，j代表每个社区,j为社区名称
        for c in comm_va[j]:  # 社区中的每个节点，代表各节点
            flag = False
            for k in range(len(comm_list)):  # 检测到的社区，k为社区名称
                if c in comm_list[k] and flag == False:
                    flag = True
                    actual.append(j)
                    baseline.append(k)
                    break
    print("nmi", metrics.normalized_mutual_info_score(actual, baseline))
    print("ari", metrics.adjusted_rand_score(actual, baseline))


def drawcommunity(g, partition, filepath):
    pos = nx.spring_layout(g)
    count1 = 0
    t = 0
    node_color = [
        "#66CCCC",
        "#FFCC00",
        "#99CC33",
        "#CC6600",
        "#CCCC66",
        "#FF99CC",
        "#66FFFF",
        "#66CC66",
        "#CCFFFF",
        "#CCCC00",
        "#CC99CC",
        "#FFFFCC",
    ]
    #    print(node_color[1])

    for com in set(partition.values()):
        count1 = count1 + 1.0
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]

        nx.draw_networkx_nodes(
            g, pos, list_nodes, node_size=220, node_color=node_color[t]
        )
        nx.draw_networkx_labels(g, pos)
        t = t + 1

    nx.draw_networkx_edges(g, pos, alpha=0.5)
    plt.savefig(filepath)
    plt.show()


############################################################
# ----------main-----------------
# edges_added = set()
# edges_removed = set()
# nodes_added = set()
# nodes_removed = set()
# G=nx.Graph()
# #edge_file='switch.t01.edges'
# edge_file='15node_t01.txt'
# #path='./LFR/muw=0.1/'
# path='./data/test1/'
# with open(path+edge_file,'r') as f:

#     edge_list=f.readlines()
#     for edge in edge_list:
#         edge=edge.split()
#         G.add_node(int(edge[0]))
#         G.add_node(int(edge[1]))
#         G.add_edge(int(edge[0]),int(edge[1]))
# G=G.to_undirected()
# #初始图
# print('T0时间片的网络G0*********************************************')
# nx.draw_networkx(G)
# fpath='./data/pic/G_0.png'
# plt.savefig(fpath)           #输出方式1: 将图像存为一个png格式的图片文件
# plt.show()
# #print G.edges()
# #comm_file='switch.t01.comm'
# comm_file='15node_comm_t01.txt'
# with open(path+comm_file,'r') as f:
#     comm_list=f.readlines()
#     comm_list=str_to_int(comm_list)
# comm={}#用来存放所检测到的社区结构，格式｛节点：社区标签｝
# comm=CDID(G,0)   #初始社区
# #画社区
# print('T0时间片的社区C0*********************************************')
# drawcommunity(G,comm,'./data/pic/community_0.png')
# initcomm=conver_comm_to_lab(comm)
# comm_va=list(initcomm.values())
# getscore(comm_va,comm_list)
# start=time.time()
# G1=nx.Graph()
# G2=nx.Graph()
# G1=G
# #filename='switch.t0'
# filename='15node_'
# for i in range(2,5):
#     print('begin loop:', i-1)
#     #comm_new_file=open(path+'output_new_'+str(i)+'.txt','r')
# #    comm_new_file=open(path+filename+str(i)+'.comm','r')
#     comm_new_file=open(path+filename+'comm_t0'+str(i)+'.txt','r')
#     if i<10:
# #        edge_list_old_file=open(path+'switch.t0'+str(i-1)+'.edges','r')
# #        edge_list_old=edge_list_old_file.readlines()
# #        edge_list_new_file=open(path+filename+str(i)+'.edges','r')
#         edge_list_new_file=open(path+filename+'t0'+str(i)+'.txt','r')
#         edge_list_new=edge_list_new_file.readlines()
#         comm_new=comm_new_file.readlines()
#     elif i==10:
# #        edge_list_old_file=open(path+'switch.t09.edges','r')
# #        edge_list_old=edge_list_old_file.readlines()
#         edge_list_new_file=open(path+'switch.t10.edges','r')
#         edge_list_new=edge_list_new_file.readlines()
#         comm_new=comm_new_file.readlines()
#     else:
# #        edge_list_old_file=open(path+'switch.t'+str(i-1)+'.edges','r')
# #        edge_list_old=edge_list_old_file.readlines()
#         edge_list_new_file=open(path+'switch.t'+str(i)+'.edges','r')
#         edge_list_new=edge_list_new_file.readlines()
#         comm_new=comm_new_file.readlines()
#     comm_new=str_to_int(comm_new)

# #    for line in edge_list_old:
# #         temp = line.strip().split()
# #
# #         G1.add_edge(int(temp[0]),int(temp[1]))
#     for line in edge_list_new:
#          temp = line.strip().split()
#          G2.add_edge(int(temp[0]),int(temp[1]))
#     print('T'+str(i-1)+'时间片的网络G'+str(i-1)+'*********************************************')
#     nx.draw_networkx(G2)
#     fpath='./data/pic/G_'+str(i-1)+'.png'
#     plt.savefig(fpath)           #输出方式1: 将图像存为一个png格式的图片文件
#     plt.show()
# #    total_nodes = previous_nodes.union(current_nodes)#当前时间片和上一时间片总节点数，两集合相关
#     total_nodes =set(G1.nodes())| set(G2.nodes())
# #    current_nodes.add(1002)
# #    previous_nodes.add(1001)

#     nodes_added=set(G2.nodes())-set(G1.nodes())
#     print ('增加节点集为：',nodes_added)
#     nodes_removed=set(G1.nodes())-set(G2.nodes())
#     print ('删除节点集为：',nodes_removed)
# #    print ('G2',G2.nodes())
# #    print ('G1',G1.nodes())
# #    print ('add node',nodes_added)
# #    print ('remove node',nodes_removed)
#     edges_added = set(G2.edges())-set(G1.edges())
#     print ('增加边集为：',edges_added)
#     edges_removed = set(G1.edges())-set(G2.edges())
#     print ('删除边集为：',edges_removed)
# #    print ('add edges',edges_added)
# #    print ('remove edges',edges_removed)
# #    print len(G1.edges())
# #    print len(edges_added),len(edges_removed)
#     all_change_comm=set()
#     #添加结点处理#############################################################
#     addn_ch_comm,addn_pro_edges,addn_commu = node_addition(G2,nodes_added,comm)
# #    print ('addnode_community',addn_commu)
# #    print edges_added
# #    print addn_pro_edges
#     edges_added=edges_added-addn_pro_edges#去掉已处理的边
# #    print edges_added
#     all_change_comm=all_change_comm | addn_ch_comm
# #    print('addn_ch_comm',addn_ch_comm)

#     #删除结点处理#############################################################
# #    print('nodes_removed',nodes_removed)
#     deln_ch_comm,deln_pro_edges,deln_commu  = node_deletion(G1,nodes_removed,addn_commu)
#     all_change_comm=all_change_comm | deln_ch_comm
#     edges_removed=edges_removed-deln_pro_edges
# #    print('deln_ch_comm',deln_ch_comm)
# #    print ('delnode_community',deln_commu)
#     #添加边处理#############################################################
# #    print('edges_added',edges_added)
#     adde_ch_comm= edge_addition(edges_added,deln_commu)
#     all_change_comm=all_change_comm | adde_ch_comm
# #    print('all_change_comm',all_change_comm)
#     #删除边处理#############################################################
#     dele_ch_comm= edge_deletion(edges_removed,deln_commu)
#     all_change_comm=all_change_comm | dele_ch_comm
# #    print('all_change_comm',all_change_comm)
#     unchangecomm=()#未改变的社区标签
#     newcomm={}#格式为｛节点：社区｝
#     newcomm=deln_commu# 添加边和删除边，只是在现有节点上处理，不会新增节点，删除节点（前面已处理）
#     unchangecomm=set(newcomm.values())-all_change_comm
#     unchcommunity={ key:value for key,value in newcomm.items() if value in unchangecomm}#未改变的社区 ：标签和结点
#     #找出变化社区所对应的子图，然后对子图运用信息动力学找出新的社区结构，加上未改变的社区结构，得到新的社区结构。
# #    print('change community:',all_change_comm)
#     Gtemp=nx.Graph()
#     Gtemp=getchangegraph(all_change_comm,newcomm,G2)
#     unchagecom_maxlabe=0
#     if len(unchangecomm)>0:
#         unchagecom_maxlabe=max(unchangecomm)
# #    print('subG',Gtemp.edges())
#     if Gtemp.number_of_edges()<1:#社区未发生变化
#         comm=newcomm
#     else:
#         getnewcomm=CDID(Gtemp,unchagecom_maxlabe)
#         print('T'+str(i-1)+'时间片子图delta_g'+str(i-1)+'*********************************************')
#         nx.draw_networkx(Gtemp)
#         fpath='./data/pic/delta_g'+str(i-1)+'.png'
#         plt.savefig(fpath)
#         plt.show()

# #        print('newcomm',getnewcomm)
#         #合并社区结构，未改的加上新获得的
# #        mergecomm=dict(unchcommunity, **getnewcomm )#格式为｛节点：社区｝
#         d=dict(unchcommunity)
#         d.update(getnewcomm)
#         comm=dict(d) #把当前获得的社区结构，作为下一次的社区输入
#         print('T'+str(i-1)+'时间片的网络社团结构C'+str(i-1)+'*********************************************')
#         drawcommunity(G2,comm,'./data/pic/community_'+str(i-1)+'.png')
# #    print ('getcommunity:',conver_comm_to_lab(comm))
#     getscore(list(conver_comm_to_lab(comm).values()),comm_new)
#     print('community number:', len(set(comm.values())))
#     print(comm)
#     G1.clear()
#     G1.add_edges_from(G2.edges())
#     G2.clear()
# print ('all done')
