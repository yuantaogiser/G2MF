''' 
    代码目的：生成图的A_matrix,graph_indicator,graph_labels,node_labels
    根据POI数据及其生成的delaunay边，生成可用于图计算的属性。
    在实际应用中，需要最少2个图
'''
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np

from utiles_global_variables import *


def generate_adjacency_matrix(edges, num_nodes,sum_point_index):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj_matrix[edge[0]-sum_point_index, edge[1]-sum_point_index] = 1
        adj_matrix[edge[1]-sum_point_index, edge[0]-sum_point_index] = 1  # 由于是无向图，需要设置为1
    return adj_matrix

def generate_graph_attributes(gpd_node,
                              gpd_edge,
                              gpd_block,
                              graph_index = 0, 
                              sum_point_index = 0, 
                              MST=False,
                              save_graph_label=True):
    '''
    根据单组图的点和边，生成图的邻接矩阵、图引索、图标签、点标签
    参数解释：
    graph_index    ：输入的block是第几个图--将图的顺序表示出来
    sum_point_index：累计节点的引索--让所有输入的节点的引索均是不同的
    gpd_node       ：输入的节点
    gpd_edge       ：输入的边
    gpd_block      ：输入的面或者街区
    '''

    if isinstance(gpd_node, str):
        gpd_node = gpd.read_file(gpd_node).set_crs('4326')
    if isinstance(gpd_edge, str):
        gpd_edge = gpd.read_file(gpd_edge).set_crs('4326')        

    # 1. 给poi点位赋予id信息，即当前图的节点引索
    id_list = gpd_node.index.to_list()
    gpd_node['id_list'] = [tmp_id + sum_point_index for tmp_id in id_list]

    # 2. 把每一条边相交的两个节点的id记录下来
    length_weight_list = []
    edge_id_list = []
    for i in range(len(gpd_edge)):
        target_edge = gpd_edge.iloc[i]
        target_edge = gpd.GeoDataFrame(pd.DataFrame(target_edge).T,geometry=[target_edge.geometry],crs='epsg:4326')
        points_intersect_line = gpd.sjoin(gpd_node, target_edge, predicate='intersects')
        tmp_id_list = points_intersect_line['id_list'].to_list()
        edge_id_list.append(tmp_id_list)
        '''2.1 计算点位之间的距离，进行反距离加权
        这里可以改进，但是否一定存在空间扰动？'''
        length_weight_list.append(target_edge['length'].to_list()[0])
    edge_id_list_copy = edge_id_list.copy()

    # ----------------------- 最小生成树 ----------------------------------
    if MST:
        edges_nx_list = []
        for i in range(len(edge_id_list)):
            edge_id = edge_id_list[i]
            tmp_tuple = (edge_id[0], edge_id[1], {"weight":length_weight_list[i]})
            edges_nx_list.append(tmp_tuple)
        G = nx.Graph()
        G.add_edges_from(edges_nx_list)
        T = nx.minimum_spanning_tree(G, algorithm='kruskal') # algorithm='kruskal', 'prim', or 'boruvka'
        
        edge_id_list = []
        for edge in T.edges(data=True):
            edge_id_list.append([min(edge[0:2]),max(edge[0:2])])

        if False:
            edge_index_list = []
            for j in range(len(edge_id_list)):
                tmp_index = edge_id_list_copy.index(edge_id_list[j])
                edge_index_list.append(tmp_index)
            MST_edge = gpd_edge.iloc[edge_index_list]
            MST_edge.to_file(r'D:\000 博士生涯\000 Achivement\2023 城市功能区识别\src\1 图\dataset\graph_data\MST_graph.shp',driver='ESRI Shapefile',encoding='utf-8')
            gpd_edge.to_file(r'D:\000 博士生涯\000 Achivement\2023 城市功能区识别\src\1 图\dataset\graph_data\gpd_graph.shp',driver='ESRI Shapefile',encoding='utf-8')
    # --------------------------------------------------------------------
        
    # 3. e.g. node1->node2（1指向了2）同时（2也指向了1）node1<-node2，详情查看DS_A.txt
    reversed_list = [tmp_list[::-1] for tmp_list in edge_id_list]
    for tmp_list in reversed_list:
        edge_id_list.append(tmp_list)
    sorted_list = sorted(edge_id_list)
    edge_tuple_list = [tuple(edge) for edge in sorted_list]

    # 4. 主要部分：生成 A.txt
    '''将邻接矩阵的下标转换为适用于PyG的DS_A.txt格式, https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip
        1. 先将其转换成字典表的形式
        2. 遍历该表，等价于读取每一个矩阵引索'''

    edges = edge_tuple_list
    num_nodes = len(gpd_node)  # 节点的数量
    adj_matrix = generate_adjacency_matrix(edges, num_nodes,sum_point_index)

    # 为了与 DS_A.txt 保持一致，下方的 i 和 row 都进行了 +1。
    # 这里加上了sum_point_index，使得每个节点都有唯一index
    A_dict={i+1+sum_point_index: (np.nonzero(row)[0]+1+sum_point_index).tolist() for i, row in enumerate(adj_matrix)}
    # print(A_dict)

    A_matrix = []
    A_dict_head = list(A_dict)
    for i in range(len(A_dict_head)):
        tmp_A_dict_head = A_dict_head[i]
        if i != 0:
            for j in A_dict[tmp_A_dict_head]:
                if j < tmp_A_dict_head:
                    continue
                else:
                    tmp1 = [A_dict_head[i],j]
                    tmp2 = [j,A_dict_head[i]]
                    A_matrix.append(tmp1)
                    A_matrix.append(tmp2)
        else:
            for j in A_dict[tmp_A_dict_head]:
                tmp1 = [A_dict_head[i],j]
                tmp2 = [j,A_dict_head[i]]
                A_matrix.append(tmp1)
                A_matrix.append(tmp2)

    # 堆叠邻接矩阵
    A_matrix = np.array(A_matrix)

    # 标记图的引索，即给给每个节点赋予图的值，告诉网络哪些节点是哪个图里的
    graph_indicator = np.repeat([graph_index],len(gpd_node))

    if save_graph_label:
        # 标记图的标签。这里根据block的shape文件中的值得到的。
        columns_list = gpd_block.columns.tolist()
        if 'change_to' in columns_list:
            tmp_POI_TYPE_CODE = gpd_block['change_to'].to_list()[0]
            if np.isnan(tmp_POI_TYPE_CODE) or gpd_block['change_to'].tolist()[0] == 0:
                graph_labels = LU_TYPE_CODE[str(int(gpd_block['CODE'].to_list()[0]))]
            else:
                tmp = list(RECLASSIFY_TYPE_CODE_DICT.keys())[list(RECLASSIFY_TYPE_CODE_DICT.values()).index(tmp_POI_TYPE_CODE)]
                graph_labels = POI_TYPE_CODE[tmp]
        else:
            graph_labels = LU_TYPE_CODE[str(int(gpd_block['CODE'].to_list()[0]))]
    else:
        graph_labels = 100
    # 标记节点的标签。这里根据poi的重分类shape文件得到的。
    node_labels = []
    for i in range(sum_point_index, len(gpd_node) + sum_point_index):
        assert len(gpd_node[gpd_node['id_list']==i]['land_use'].to_list())==1
        tmp_label = POI_TYPE_CODE[gpd_node[gpd_node['id_list']==i]['land_use'].to_list()[0]]
        node_labels.append(tmp_label)

    sum_point_index += len(gpd_node)
    return A_matrix,graph_indicator,graph_labels,node_labels,sum_point_index

