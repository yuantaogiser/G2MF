'''
根据shp文件，构建图结构
'''

import pandas as pd
import geopandas as gpd
import os
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from time import time

from utiles_generate_block_graph import generate_sematic_node_with_feature
from utiles_generate_block_graph import generate_sematic_graph_attributes
from utiles_generate_poi_graph import generate_graph_attributes
from shapely.geometry import Polygon

def calculate_area_ratio(target_block, target_sematic_map_tmp):
    target_block_area = target_block.area.to_list()[0]
    target_sematic_map_tmp_area = target_sematic_map_tmp.dissolve().area.to_list()[0]

    area_ratio = target_sematic_map_tmp_area / target_block_area
    return area_ratio

def get_edges_in_polygon(gpd_poi_delaunay_lines,target_polygon):
    target_polygon_buffer = gpd.GeoDataFrame(pd.DataFrame(target_polygon).drop('geometry').T,geometry=[target_polygon['geometry']],crs='epsg:4326').buffer(0.00000001) #buffer 1cm
    target_polygon_buffer = gpd.GeoDataFrame(geometry=target_polygon_buffer,crs='epsg:4326')
    edges_within_polygon = gpd.sjoin(gpd_poi_delaunay_lines, target_polygon_buffer, predicate='within')
    return edges_within_polygon

def calculate_internal_angles(poly):
    if isinstance(poly, Polygon):
        coords = list(poly.exterior.coords)[:-1]  # 去掉重复的最后一个点
    else:
        raise ValueError("Input must be a shapely Polygon")
    
    angles = []
    n = len(coords)
    for i in range(n):
        # 取前一个点、当前点和后一个点
        pt1 = np.array(coords[(i - 1) % n])
        pt2 = np.array(coords[i % n])
        pt3 = np.array(coords[(i + 1) % n])
        
        # 计算向量
        vec1 = pt1 - pt2
        vec2 = pt3 - pt2
        # 计算点积和向量长度
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # 计算角度并转换为度
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        angle_deg = np.degrees(angle)
        angles.append(angle_deg)
    
    return angles

def find_min_dist_two_points(points_intersect_polygon, gpd_lines):
    # 思路：通过将各个子图dissolve再buffer，这样就能有多个独立的面，然后用面去与子图相交
    #      再将相交的点集，找出剩下点集的全局最短距离的点，连接之后，迭代。
    gpd_lines_dissolve = gpd_lines.dissolve()
    gpd_lines_buffer = gpd_lines_dissolve.buffer(0.00001)
    gpd_lines_buffer_singlepart = gpd_lines_buffer.explode(index_parts=True)
    if len(gpd_lines_buffer_singlepart)==1:
        return gpd_lines, len(gpd_lines_buffer_singlepart)
    
    tmp_buffer = gpd_lines_buffer_singlepart.iloc[0]
    # tmp_buffer = gpd.GeoDataFrame(pd.DataFrame(tmp_buffer).T, geometry=[tmp_buffer.geometry], crs='epsg:4326')
    tmp_buffer = gpd.GeoDataFrame(geometry=[tmp_buffer], crs='epsg:4326')
    if 'index_right' in points_intersect_polygon.columns:
        points_intersect_polygon = points_intersect_polygon.drop(['index_right'], axis=1) # 多次sjoin需要删除这个字段
    if 'index_left' in points_intersect_polygon.columns:
        points_intersect_polygon = points_intersect_polygon.drop(['index_left'], axis=1) # 多次sjoin需要删除这个字段
    points_intersect_buffer = gpd.sjoin(points_intersect_polygon, tmp_buffer, predicate='intersects')

    points_leftover = gpd.overlay(points_intersect_polygon, points_intersect_buffer, how='difference')

    dist_matrix = np.zeros((len(points_leftover),len(points_intersect_buffer)))
    for i in range(len(points_leftover)):
        for j in range(len(points_intersect_buffer)):
            tmp_point_leftover = points_leftover.iloc[i]
            tmp_points_intersect_buffer = points_intersect_buffer.iloc[j]

            x1 = tmp_point_leftover.geometry.x
            y1 = tmp_point_leftover.geometry.y
            x2 = tmp_points_intersect_buffer.geometry.x
            y2 = tmp_points_intersect_buffer.geometry.y
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

            dist_matrix[i,j] = dist

    # min_value = dist_matrix.min()
    min_index = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
    
    tmp_line = gpd.GeoSeries(geometry.LineString([(points_leftover.iloc[min_index[0]].geometry.x, points_leftover.iloc[min_index[0]].geometry.y), 
                                            (points_intersect_buffer.iloc[min_index[1]].geometry.x, points_intersect_buffer.iloc[min_index[1]].geometry.y)]),
                         crs='EPSG:4326',
                         index=['1'])
    tmp_line = gpd.GeoDataFrame(geometry=[tmp_line[0]],crs='epsg:4326')
    gpd_lines = pd.concat([gpd_lines,tmp_line],axis=0)
    
    return gpd_lines,len(gpd_lines_buffer_singlepart)


def build_graph_by_points(path_poi):

    if isinstance(path_poi,str):
        gpd_poi = gpd.read_file(path_poi)
        gpd_poi = gpd.GeoDataFrame(gpd_poi).drop_duplicates(subset=['geometry'])
    else:
        gpd_poi = path_poi

    # 1. 生成delaunay_lines，这里的线型是multi的
    gpd_poi_dissolve = gpd_poi.dissolve()
    geom_poi_delaunay_lines = gpd_poi_dissolve.delaunay_triangles(only_edges=True)
    gpd_poi_delaunay_lines = gpd.GeoDataFrame(geometry=geom_poi_delaunay_lines)

    # 2. 生成delaunay_polygons，这里的面型是collection
    geom_poi_delaunay_polygons = gpd_poi_dissolve.delaunay_triangles(only_edges=False)
    gpd_poi_delaunay_polygons = gpd.GeoDataFrame(geometry=geom_poi_delaunay_polygons)
    geom_poi_delaunaygeometry = gpd_poi_delaunay_polygons.geometry.explode()[0]
    gpd_poi_delaunay_polygons = gpd.GeoDataFrame(geometry=geom_poi_delaunaygeometry)

    # 3. multipart转换为singlepart
    multiline_list = gpd_poi_delaunay_lines[gpd_poi_delaunay_lines.geometry.type=='MultiLineString']
    multiline_list["singleparts"] = multiline_list.apply(lambda x: [p for p in x.geometry.geoms], axis=1)
    exploded = multiline_list.explode("singleparts") #List to rows
    exploded = exploded.set_geometry("singleparts")
    del(exploded["geometry"])
    gpd_poi_delaunay_lines = exploded.rename_geometry("geometry")
    gpd_poi_delaunay_lines = gpd.GeoDataFrame(gpd_poi_delaunay_lines).set_crs(crs='epsg:4326').reset_index(drop=True)


    # gpd_poi_delaunay_polygons.to_file('/path/to/folder/',driver='ESRI Shapefile',encoding='utf-8')
    # gpd_poi_delaunay_lines.to_file('/path/to/folder/',driver='ESRI Shapefile',encoding='utf-8')
    gpd_poi_delaunay_lines['length'] = gpd_poi_delaunay_lines.to_crs('epsg:3857').length
    return gpd_poi_delaunay_lines

def build_graph_by_points_IJGIS(path_poi):
    '''
    reference: 
        Su, Y., Zhong, Y., Liu, Y., & Zheng, Z. 
        2023
        A graph-based framework to integrate semantic object/land-use relationships for urban land-use mapping with case studies of Chinese cities. 
        International Journal of Geographical Information Science, 37(7), 1582-1614. 
        https://www.tandfonline.com/doi/abs/10.1080/13658816.2023.2203199
    '''

    if isinstance(path_poi,str):
        gpd_poi = gpd.read_file(path_poi)
        gpd_poi = gpd.GeoDataFrame(gpd_poi).drop_duplicates(subset=['geometry'])
    else:
        gpd_poi = path_poi

    gpd_lines = gpd.GeoDataFrame([])
    for target_poi_index in range(len(gpd_poi)-1):
        target_poi = gpd_poi.iloc[target_poi_index]
        
        for next_poi_index in range(target_poi_index+1,len(gpd_poi)):
            next_poi = gpd_poi.iloc[next_poi_index]
            
            tmp_line = gpd.GeoSeries(geometry.LineString([(target_poi.geometry.x, target_poi.geometry.y), 
                                            (next_poi.geometry.x, next_poi.geometry.y)]),
                        crs='EPSG:4326',
                        index=['1'])
            tmp_line = gpd.GeoDataFrame(geometry=[tmp_line[0]],crs='epsg:4326')
            
            gpd_lines = pd.concat([gpd_lines,tmp_line],axis=0)
    gpd_lines = gpd_lines.reset_index(drop=True)

    gpd_lines['length'] = gpd_lines.to_crs('epsg:3857').length.to_list()
    gpd_lines = gpd_lines[gpd_lines['length']<=1000]
    gpd_lines = gpd_lines.reset_index(drop=True)
    return gpd_lines

def parallel_body(data_list):
    poi_in_UFZ               = data_list[0]
    target_instance          = data_list[1]
    target_polygon           = data_list[2]
    graph_index              = data_list[3]
    poi_sum_point_index      = data_list[4]
    block_sum_point_index    = data_list[5]
    sort_index               = data_list[6]

    MST = False
    save_graph_label = True
    # -----------------------------
    # ------ 生成POI图 -------
    # -----------------------------
    poi_DT_lines = build_graph_by_points(poi_in_UFZ)
    # poi_DT_lines = build_graph_by_points_IJGIS(poi_in_UFZ)

    poi_DT_lines = poi_DT_lines.reset_index(drop=True)
    poi_in_UFZ = poi_in_UFZ.reset_index(drop=True)
    poi_A_matrix,poi_graph_indicator,poi_graph_labels,poi_node_labels,poi_sum_point_index = \
        generate_graph_attributes(poi_in_UFZ, poi_DT_lines, target_polygon, graph_index, poi_sum_point_index, 
                                  MST, save_graph_label)

    poi_graph = [poi_A_matrix,poi_graph_indicator,poi_graph_labels,poi_node_labels,sort_index]
    # -----------------------------
    # ------ 生成block图 -------
    # -----------------------------
    block_point_list = generate_sematic_node_with_feature(target_instance)
    block_DT_lines = build_graph_by_points(block_point_list)
    # block_DT_lines = build_graph_by_points_IJGIS(block_point_list)

    block_DT_lines = block_DT_lines.reset_index(drop=True)
    tmp_points_intersect_polygon = block_point_list.reset_index(drop=True)
    block_A_matrix,block_graph_indicator,block_graph_labels,block_node_labels,block_node_attributes, block_sum_point_index = \
        generate_sematic_graph_attributes(tmp_points_intersect_polygon,block_DT_lines,target_polygon,graph_index,block_sum_point_index, 
                                          MST, save_graph_label)

    block_graph = [block_A_matrix,block_graph_indicator,block_graph_labels,block_node_labels,block_node_attributes,sort_index]
    return [poi_graph, block_graph]

if __name__=="__main__":
    multiprocessing.freeze_support()

    '''加载所需的所有数据，对齐两种数据源，然后共同过滤，以保证图标签一致。'''
    path_poi_tmp = '/path/to/folder/poi'
    path_block_tmp = '/path/to/folder/mapping_unit'
    path_sematic_map_tmp = '/path/to/folder/sematic_map'

    # 预定义graph index、sum_point_index，方便并行操作
    graph_index = 1
    poi_sum_point_index = 0
    block_sum_point_index = 0
    poi_A_matrix_list, poi_graph_indicator_list, poi_graph_labels_list, poi_node_labels_list, poi_sort_index_list = [], [], [], [], []
    block_A_matrix_list, block_graph_indicator_list, block_graph_labels_list, block_node_labels_list, block_node_attributes_list, block_sort_index_list = [], [], [], [], [], []
    index_list = []

    city_name = 'bjsh'
    filename_list = os.listdir(path_poi_tmp)
    for filename in filename_list:
        if city_name == 'bjsh':
            if filename in ['hefei', 'nanjing']:
                continue

        elif city_name == 'hfnj':
            if filename not in ['hefei', 'nanjing']:
                continue
        else:
            raise NameError('输入错误城市')

        path_poi = os.path.join(path_poi_tmp, filename, filename+".shp")
        path_block = os.path.join(path_block_tmp, filename, filename + ".shp")
        path_sematic_map = os.path.join(path_sematic_map_tmp, filename + ".shp")

        gpd_poi = gpd.read_file(path_poi)
        gpd_block = gpd.read_file(path_block)
        gpd_block['sort_index'] = np.zeros(len(gpd_block)).tolist()
        gpd_sematic_map = gpd.read_file(path_sematic_map)

        for sort_index in tqdm(range(len(gpd_block))):
            target_block = gpd_block.iloc[sort_index]
            target_block = gpd.GeoDataFrame(pd.DataFrame(target_block).T, geometry=[target_block.geometry], crs='epsg:4326')
            
            if not target_block['CODE'].to_list()[0] > 0:
                continue

            columns_list = target_block.columns.tolist()
            # 过滤block中，被标记为‘is_label’要素值为非1的。
            if 'is_label' in columns_list:
                if target_block['is_label'].tolist()[0] != 1:
                    continue
            # 过滤block中，在后处理阶段被判定为需要删除的block
            if 'change_to' in columns_list:
                if target_block['change_to'].tolist()[0] == -1:
                    continue

            '''筛选block list的可用block'''
            target_sematic_map_tmp = gpd.clip(gpd_sematic_map, target_block)
            target_sematic_map = target_sematic_map_tmp[target_sematic_map_tmp['value']==1]
            target_instance = target_sematic_map.explode(index_parts=True).reset_index(drop=True)

            # 过滤掉面积小于15平米的实例
            target_instance_3857 = target_instance.to_crs('epsg:3857')
            target_instance_3857['area'] = target_instance_3857.area.tolist()
            target_instance = target_instance[target_instance_3857['area']>15].dropna()
            target_instance = target_instance.reset_index(drop=True)

            # 当instance的数量小于3的时候，不再构建图。
            if len(target_instance) < 3:
                continue

            '''筛选poi list的可用block'''
            points_intersect_polygon = gpd.sjoin(gpd_poi, target_block, predicate='intersects')
            # 删除poi中字段为 “删除” 的字段
            if '删除' in points_intersect_polygon['land_use'].to_list():
                points_intersect_polygon = points_intersect_polygon[points_intersect_polygon['land_use']!='删除']

            # 当前block存在poi点数少于5个，则不计算该图
            if len(points_intersect_polygon) < 5:
                continue

            points_intersect_polygon = points_intersect_polygon.drop(['address', 'adname', 'page_publi', 'adcode', 'pname', 'cityname', 'name', 'location', '_id', 'Unnamed_ 1', 'index_right'],axis=1)
            
            index_list.append([points_intersect_polygon,
                            target_instance,
                            target_block,
                            graph_index,
                            poi_sum_point_index,
                            block_sum_point_index,
                            sort_index])
            poi_sum_point_index += len(points_intersect_polygon)
            block_sum_point_index += len(target_instance)
            gpd_block['sort_index'].iloc[sort_index] = sort_index    
            graph_index += 1

    # parallel body
    start_time = time()
    core_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=core_num)
    result_list = pool.map_async(parallel_body, index_list).get()
    pool.close()
    pool.join()
    print('cost time: ',round(time()-start_time,2))

    for result in result_list:
        poi_result = result[0]
        block_result = result[1]

        poi_A_matrix, poi_graph_indicator, poi_graph_labels, poi_node_labels, poi_sort_index = poi_result
        poi_A_matrix = poi_A_matrix.tolist()
        poi_graph_indicator = poi_graph_indicator.tolist()
        poi_sort_index_list.append(poi_sort_index)
        poi_graph_labels_list.append(poi_graph_labels)
        for tmp in poi_A_matrix:
            poi_A_matrix_list.append(tmp)
        for tmp in poi_graph_indicator:
            poi_graph_indicator_list.append(tmp)
        for tmp in poi_node_labels:
            poi_node_labels_list.append(tmp)

        if not isinstance(block_result[0], np.ndarray):
            continue
        elif not isinstance(block_result[-2], list):
            continue
        block_A_matrix, block_graph_indicator, block_graph_labels, block_node_labels, block_node_attributes, block_sort_index = block_result
        block_A_matrix = block_A_matrix.tolist()
        block_graph_indicator = block_graph_indicator.tolist()
        block_sort_index_list.append(block_sort_index)
        block_graph_labels_list.append(block_graph_labels)
        for tmp in block_A_matrix:
            block_A_matrix_list.append(tmp)
        for tmp in block_graph_indicator:
            block_graph_indicator_list.append(tmp)
        for tmp in block_node_labels:
            block_node_labels_list.append(tmp)
        for tmp in block_node_attributes:
            block_node_attributes_list.append(tmp)

    name = f'{city_name}output'
    os.makedirs(f'dataset/graph_data/mydataset/poi{name}/raw', exist_ok=True)
    np.savetxt(f'dataset/graph_data/mydataset/poi{name}/raw/poi{name}_A.txt', poi_A_matrix_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/poi{name}/raw/poi{name}_graph_indicator.txt', poi_graph_indicator_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/poi{name}/raw/poi{name}_graph_labels.txt', poi_graph_labels_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/poi{name}/raw/poi{name}_node_labels.txt', poi_node_labels_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/poi{name}/raw/poi{name}_sort_index.txt', poi_sort_index_list, delimiter=',',fmt='%d')

    os.makedirs(f'dataset/graph_data/mydataset/block{name}/raw', exist_ok=True)
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_A.txt', block_A_matrix_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_graph_indicator.txt', block_graph_indicator_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_graph_labels.txt', block_graph_labels_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_node_labels.txt', block_node_labels_list, delimiter=',',fmt='%d')
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_node_attributes.txt', block_node_attributes_list, delimiter=',',fmt='%f')
    np.savetxt(f'dataset/graph_data/mydataset/block{name}/raw/block{name}_sort_index.txt', block_sort_index_list, delimiter=',',fmt='%d')
