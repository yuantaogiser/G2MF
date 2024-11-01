import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry

import networkx as nx

from utiles_global_variables import *

import warnings
warnings.filterwarnings('ignore')


def decision_tree_for_graph_label():
    return

def series2gdf(series):
    gdf = gpd.GeoDataFrame(pd.DataFrame(series).T, geometry=[series.geometry], crs='epsg:4326')
    return gdf

def generate_sample_lines(target_inst):
    # center_point = target_inst.centroid
    center_x = target_inst.centroid.x.to_list()[0]
    center_y = target_inst.centroid.y.to_list()[0]

    R = 0.02
    piece = 90 # 将一个圆等分成90份

    th = np.arange(0, 2 * math.pi, 2 * math.pi / piece)
    xList = R * np.cos(th) + center_x
    yList = R * np.sin(th) + center_y
    pointsList = np.transpose(np.vstack((xList, yList)))

    latSampleList = pointsList[:,0]
    lonSampleList = pointsList[:,1]
    points_list = pd.DataFrame({'Latitude': latSampleList, 'Longitude': lonSampleList})
    gpd_points_list = gpd.GeoDataFrame(points_list, geometry = gpd.points_from_xy(points_list.Latitude, points_list.Longitude), crs='epsg:4326')

    sample_lines = gpd.GeoDataFrame([])
    for i in range(len(gpd_points_list)):
        target_point = gpd_points_list.iloc[i]
        
        tmp_line = gpd.GeoSeries(geometry.LineString([(target_point.geometry.x, target_point.geometry.y), 
                                            (center_x, center_y)]), crs='EPSG:4326', index=['1'])
        tmp_line = gpd.GeoDataFrame(geometry=[tmp_line[0]],crs='epsg:4326')
        sample_lines = pd.concat([sample_lines,tmp_line],axis=0)
    
    return sample_lines

def calculate_distance(intersection,target_inst):
    # print(intersection)
    coord_x = intersection.x
    coord_y = intersection.y
    
    center_x = target_inst.centroid.x.to_list()[0]
    center_y = target_inst.centroid.y.to_list()[0]
    
    distance = np.sqrt( (coord_x-center_x)**2 + (coord_y-center_y)**2 )
    return distance

def formula_RSI(distance_list):
    tmp_list = []
    for target_distance in distance_list:
        # 除以100是为了让其变成浮点数
        tmp = abs(100 * target_distance / np.sum(distance_list) - 100/len(distance_list)) / 100 
        tmp_list.append(tmp)
    rsi = np.sum(tmp_list)
    return rsi

def calculate_RSI(target_inst):

    target_inst = series2gdf(target_inst)
    sample_lines = generate_sample_lines(target_inst)
    target_inst = series2gdf(target_inst.iloc[0]).reset_index(drop=True).to_crs('epsg:3857')
    target_inst_boundary = target_inst.boundary

    distance_list = []
    for j in range(len(sample_lines)):
        '''计算主体统一转换成epsg:3857'''
        target_sample_line = sample_lines.iloc[j]
        # print(target_sample_line)

        target_sample_line = series2gdf(target_sample_line).reset_index(drop=True).to_crs('epsg:3857')
        # target_inst = series2gdf(target_inst)

        intersections = target_inst_boundary.intersection(target_sample_line)

        if not intersections.is_empty[0]:

            intersections = intersections.explode(index_parts=True)[0]
            
            tmp_distance_list = np.array([])
            for k in range(len(intersections)):
                intersection = intersections.iloc[k]

                if not intersection.geom_type == 'Point':
                    continue

                distance = calculate_distance(intersection,target_inst)
                tmp_distance_list = np.append(tmp_distance_list,distance)
            # if len(intersections)>=3:
            #     tmp_distance_list = np.array([np.max(tmp_distance_list), np.min(tmp_distance_list)])
        else:
            continue
        for tmp in tmp_distance_list:
            distance_list.append(tmp)

    rsi = formula_RSI(distance_list)
    return rsi

def calculate_compactness(area, perimeter):
    compactness = np.sqrt(4 * math.pi * area / perimeter ** 2)
    return compactness

def calculate_area(target_inst_3857):
    area = target_inst_3857.area.tolist()[0]
    return area

def calculate_perimeter(target_inst_3857):
    perimeter = target_inst_3857.length.tolist()[0]
    return perimeter

def calculate_regularity(area, target_inst_3857):
    regularity = area / target_inst_3857.minimum_rotated_rectangle().area.tolist()[0]
    return regularity

def calculate_aspect_ratio(target_inst_3857):
    minimum_rotated_rectangle = target_inst_3857.minimum_rotated_rectangle()
    minimum_rotated_rectangle_coord = minimum_rotated_rectangle.boundary.get_coordinates().drop_duplicates()
    assert len(minimum_rotated_rectangle_coord) == 4

    x0 = minimum_rotated_rectangle_coord.iloc[0].x
    y0 = minimum_rotated_rectangle_coord.iloc[0].y

    length_list = []
    for i in range(1,4):
        xi = minimum_rotated_rectangle_coord.iloc[i].x
        yi = minimum_rotated_rectangle_coord.iloc[i].y

        tmp_length = np.sqrt( (xi-x0)**2 + (yi-y0)**2 )
        length_list.append(tmp_length)
    length_long = np.median(length_list)
    length_short = np.min(length_list)
    aspect_ratio = length_short / length_long

    index_long = length_list.index(length_long) + 1
    #  计算出弧度，如果是角度需要乘以(180 / math.pi)
    direction = math.acos(abs(minimum_rotated_rectangle_coord.iloc[index_long].x - x0) / length_long)
    return aspect_ratio, direction

def generate_sematic_node_with_feature_in_single_instance(target_inst,sum_area):

    target_inst_3857 = series2gdf(target_inst).to_crs('epsg:3857')
    
    # 计算polygon的面积
    area = calculate_area(target_inst_3857)

    if target_inst_3857['value'].to_list()[0] == 1:
        # 计算半径形状指数
        rsi = calculate_RSI(target_inst)
        # 计算polygon的周长
        perimeter = calculate_perimeter(target_inst_3857)
        # 计算紧凑度
        compactness = calculate_compactness(area, perimeter)
        # 计算规整度
        regularity = calculate_regularity(area, target_inst_3857)
        # 计算纵横比,方向
        aspect_ratio,direction = calculate_aspect_ratio(target_inst_3857)

        center_point_element = {
            'feat_A'   : round(area, 6),
            'feat_P'   : round(perimeter, 6),
            'feat_C'   : round(compactness, 6),
            'feat_RSI' : round(rsi, 6),
            'feat_R'   : round(regularity, 6),
            'feat_AR'  : round(aspect_ratio, 6),
            'feat_D'   : round(direction, 6),
            'feat_areaR':round(area/sum_area, 6),
            'node_type': int(target_inst['value'])
                                }
    else:
        center_point_element = {
            'feat_A'   : round(area, 6),
            'feat_P'   : 0.,
            'feat_C'   : 0.,
            'feat_RSI' : 0.,
            'feat_R'   : 0.,
            'feat_AR'  : 0.,
            'feat_D'   : 0.,
            'feat_areaR':round(area/sum_area, 6),
            'node_type': int(target_inst['value'])
                                }

    center_point_geometry = series2gdf(target_inst).centroid.reset_index(drop=True)

    center_point = gpd.GeoDataFrame(pd.DataFrame.from_dict(center_point_element,orient='index').T,
                                    geometry = center_point_geometry,
                                    crs = 'epsg:4326')
    return center_point

def generate_sematic_node_with_feature(gpd_inst):

    sum_area = gpd_inst.to_crs('epsg:3857').area.sum()

    center_point_list = gpd.GeoDataFrame([])
    for i in range(len(gpd_inst)):

        target_inst = gpd_inst.iloc[i]

        center_point = generate_sematic_node_with_feature_in_single_instance(target_inst,sum_area)
        center_point_list = pd.concat([center_point_list,center_point])

    center_point_list = center_point_list.reset_index(drop=True)

    return center_point_list

def generate_adjacency_matrix(edges, num_nodes,sum_point_index):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj_matrix[edge[0]-sum_point_index, edge[1]-sum_point_index] = 1
        adj_matrix[edge[1]-sum_point_index, edge[0]-sum_point_index] = 1  # 由于是无向图，需要设置为1
    return adj_matrix

def generate_sematic_graph_attributes(gpd_node,
                                      gpd_edge,
                                      gpd_block,
                                      graph_index = 0, 
                                      sum_point_index = 0, 
                                      MST=False,
                                      save_graph_label = True):
    '''
    根据单组图的点和边，生成图的邻接矩阵、图引索、图标签、点标签
    参数解释：
    graph_index    ：输入的block是第几个图--将图的顺序表示出来
    sum_point_index：累计节点的引索--让所有输入的节点的引索均是不同的
    gpd_node        ：输入的节点
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
    # 此外！！！这里是将sematic map的语义属性赋给了节点！！
    node_labels = []
    node_attributes = []
    for i in range(sum_point_index, len(gpd_node) + sum_point_index):
        assert len(gpd_node[gpd_node['id_list']==i]['node_type'].to_list())==1
        tmp_label = gpd_node[gpd_node['id_list']==i]['node_type'].to_list()[0]
        node_labels.append(tmp_label)

        tmp_feature_A   = gpd_node[gpd_node['id_list']==i]['feat_A'].to_list()[0]
        tmp_feature_P   = gpd_node[gpd_node['id_list']==i]['feat_P'].to_list()[0]
        tmp_feature_C   = gpd_node[gpd_node['id_list']==i]['feat_C'].to_list()[0]
        tmp_feature_RSI = gpd_node[gpd_node['id_list']==i]['feat_RSI'].to_list()[0]
        tmp_feature_R   = gpd_node[gpd_node['id_list']==i]['feat_R'].to_list()[0]
        tmp_feature_AR  = gpd_node[gpd_node['id_list']==i]['feat_AR'].to_list()[0]
        tmp_feature_D  = gpd_node[gpd_node['id_list']==i]['feat_D'].to_list()[0]
        tmp_feature_areaR  = gpd_node[gpd_node['id_list']==i]['feat_areaR'].to_list()[0]
        tmp_feat_geometric = [tmp_feature_A, tmp_feature_P, tmp_feature_C, tmp_feature_RSI, tmp_feature_R, tmp_feature_AR, tmp_feature_D, tmp_feature_areaR]
        node_attributes.append(tmp_feat_geometric)

    sum_point_index += len(gpd_node)
    return A_matrix,graph_indicator,graph_labels,node_labels,node_attributes, sum_point_index
