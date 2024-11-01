import os
import arcpy
from config import *
import geopandas as gpd

import pandas as pd

def Clip_Road(boundary, road):
    """
    利用边界数据对道路进行裁剪
    """
    # 2024年4月18日 为了保留区划边界部分的block，对boundary进行buffer，这样可以切出来更多的road
    threshold = 2000
    arcpy.Buffer_analysis(boundary, 'boundary_buffered', f'{threshold} meters')

    if arcpy.Exists(road):
        arcpy.Clip_analysis(road, 'boundary_buffered', "road_clip")
        return f"road_clip"
    else:
        raise ValueError(f"DATA {road} NOT EXIST!")


def Select_Road(road_path):
    """
    根据属性对道路进行筛选
    """
    if arcpy.Exists(road_path):
        arcpy.FeatureClassToFeatureClass_conversion(road_path, out_name=f'road_select', where_clause=ROAD_SELECT_EXPRESSION)
    else:
        raise ValueError(f"DATA {road_path} NOT EXIST!")
    return f'road_select'


def Convert_Centerline(osm_path, threshold):
    """
    提取中心线
    """
    # 投影到EPSG:3857(后续提取centerline的时候使用EPSG:4326会报错)
    arcpy.Project_management(osm_path, 'road_prj', arcpy.SpatialReference(PROJ_CRS))

    # 1. 缓冲区构建
    arcpy.Buffer_analysis('road_prj', 'road_buffered', f'{threshold} meters')

    # 2. 融合缓冲区(注意，不能在上一步通过dissolve_option="ALL"进行融合，原因未知)
    # 注意 dissolve会导致路网属性丢失
    # 后续会SpatialJoin回去
    arcpy.Dissolve_management('road_buffered', 'road_buffered_dissolved')

    # 3. 提取中心线
    arcpy.PolygonToCenterline_topographic('road_buffered_dissolved', 'road_centerline')
    arcpy.Delete_management(['road', 'road_buffered', 'road_buffered_dissolved'])

    return 'road_centerline'


def Extend_Road(roads, distance):
    """
    延伸线
    """
    arcpy.FeatureToLine_management(roads, 'extend_road')
    arcpy.ExtendLine_edit('extend_road', f'{distance} meters', 'EXTENSION')
    # arcpy.Delete_management(roads)
    return 'extend_road'


def Check_Topo(roads):
    """
    检查拓扑
    """
    dataset_name = 'topo'
    topology_name = 'topology'
    topo_data_name = 'road_data'
    # work tree : topo\topology
    # 1. 生成要素集
    arcpy.CreateFeatureDataset_management(out_name=dataset_name, spatial_reference=arcpy.SpatialReference(PROJ_CRS))
    # 2. 复制道路到要素集中
    arcpy.FeatureClassToFeatureClass_conversion(roads, dataset_name, topo_data_name)
    # 3. 生成拓扑并，向拓扑中添加要素类，添加拓扑规则
    arcpy.CreateTopology_management(dataset_name, topology_name)
    arcpy.AddFeatureClassToTopology_management(f"{dataset_name}/{topology_name}", f"{dataset_name}/{topo_data_name}")
    arcpy.AddRuleToTopology_management(f"{dataset_name}/{topology_name}", "Must Not Have Dangles (Line)",
                                       f"{dataset_name}/{topo_data_name}")

    # 4. 拓扑检查
    try:
        arcpy.ValidateTopology_management("{0}/{1}".format(dataset_name, topology_name))
    except:
        print('错误数量过大，但是并不影响进程，拓朴验证已修复路网')

    # 5. 输出Topo监测出的断头点
    arcpy.ExportTopologyErrors_management("{0}/{1}".format(dataset_name, topology_name), out_basename='err')
    arcpy.Delete_management('err_line')
    arcpy.Delete_management('err_poly')
    return "err_point"


def Clean_Spike(roads, pts, threshold, path_output, keep_spike=False):
    """
    清除悬挂线段(断头路)
    """
    # 1. 复制一份道路数据
    arcpy.MakeFeatureLayer_management(roads, 'roads_copy')

    # 2. 找出悬挂线
    arcpy.SelectLayerByLocation_management('roads_copy', 'INTERSECT', pts)
    arcpy.CopyFeatures_management('roads_copy', "spike_roads")
    if keep_spike:
        arcpy.FeatureClassToFeatureClass_conversion(pts, path_output, f"err_points.shp")
        arcpy.FeatureClassToFeatureClass_conversion("spike_roads", path_output, f"err_lines.shp")
    spike = "spike_roads"

    # 3. 计算悬挂线的长度
    arcpy.AddGeometryAttributes_management(spike, 'LENGTH_GEODESIC', 'METERS')

    # 4. 筛除过短的悬挂线
    expression = f"LENGTH_GEO < {threshold}"
    arcpy.FeatureClassToFeatureClass_conversion(spike, out_name="to_be_cut", where_clause=expression)
    arcpy.Erase_analysis(roads, 'to_be_cut', 'road_master')

    # 5. 删除过程中文件
    return 'road_master'


def Join_Attributes(origin, master):
    """
    利用空间连接找回道路原来的属性
    """
    # 1. 先构建处理好后道路的缓冲区
    arcpy.Buffer_analysis(master, 'road_master_buffered', "10 meters")
    field_mappings = arcpy.FieldMappings()
    for field_name in FIELD_TO_KEEP:
        # 创建 FieldMap 对象并添加输入字段和输出字段
        field_map = arcpy.FieldMap()
        field_map.addInputField(origin, field_name)
        out_field = field_map.outputField
        out_field.name = field_name
        out_field.aliasName = field_name
        field_map.outputField = out_field
        # 将 FieldMap 对象添加到 FieldMappings 对象中
        field_mappings.addFieldMap(field_map)

    arcpy.SpatialJoin_analysis("road_master_buffered",
                               origin,
                               "road_master_buffered_with_attr",
                               join_operation="JOIN_ONE_TO_ONE",
                               match_option="LARGEST_OVERLAP",
                               join_type='KEEP_ALL',
                               field_mapping=field_mappings)

    arcpy.SpatialJoin_analysis("road_master",
                               "road_master_buffered_with_attr",
                               "road_master_with_attr",
                               "JOIN_ONE_TO_ONE",
                               match_option="WITHIN",
                               join_type='KEEP_ALL',
                               field_mapping=field_mappings)

    return "road_master_with_attr"


def Delete_Fields(roads):
    """
    删除一些不必要的字段
    """
    fields_to_delete = [f for f in ["Join_Count", "TARGET_FID"]]
    arcpy.DeleteField_management(roads, fields_to_delete)
    arcpy.CopyFeatures_management(roads, "road_results")
    # arcpy.Delete_management(roads)
    arcpy.AddGeometryAttributes_management(roads, "LENGTH_GEODESIC", "METERS")
    # arcpy.Delete_management(['topo', 'extend_road', 'road_selected'])
    return "road_results"


def Create_mapping_unit(road, path_administrative, path_output, out_name):
    """
    利用线生成parcel
    """
    # # 1. 先生成缓冲距离字段
    arcpy.CalculateField_management(
        road, "bufferDis", "getDis(!highway!, !railway!)", "PYTHON3", CALCULATE_FIEDLD_EXPRESSION
        # road, "bufferDis", "getDis(!fclass!)", "PYTHON3", CALCULATE_FIEDLD_EXPRESSION
    )

    # 2. 生成缓冲区(所有都合并为一个面即可)
    arcpy.Buffer_analysis(road, f"{road}_buffer", "bufferDis", dissolve_option="ALL")

    # 3. 要素转面
    arcpy.FeatureToPolygon_management(f"{road}_buffer", "parcel")

    # 4. 筛除最大面积的元素(道路本体，通过SHAPE_Length筛除)和过小的元素
    
    arcpy.FeatureClassToFeatureClass_conversion("parcel",
                                                out_path = path_output,
                                                out_name = out_name,
                                                # where_clause = "SHAPE_Length <= 100000"
                                                where_clause="SHAPE_Area >= 5000 and SHAPE_Length <= 100000"
                                                )
    retrans_proj(os.path.join(path_output, out_name + '.shp'), path_administrative)
    return out_name

def retrans_proj(path_data, path_administrative):
    gpd_data = gpd.read_file(path_data)
    gpd_data = gpd_data.to_crs('epsg:4326')

    # gpd_layer = gpd.read_file(os.path.join(FILE_ROOT, DATABASE_NAME),layer='boundary_buffered')
    gpd_layer = gpd.read_file(path_administrative)
    cliped = gpd.clip(gpd_layer,gpd_data)
    cliped = cliped.explode(index_parts=True)

    '''这里需要根据block的面积和周长来剔除一些夸张的、不可保留的block
        突然想到，我们做的是城市研究，直接用城市边界进行裁剪即可！！！
    ''' 
    # final_block = merge_remine_block(cliped,gpd_layer)
    cliped.to_file(path_data,driver='ESRI Shapefile',encoding='utf-8')

def gpd_clip(gpd_layer,gpd_cliped):
    cliped = gpd.clip(gpd_layer,gpd_cliped)
    return cliped
    
def merge_remine_block(gpd_block,gpd_bound):

    gpd_block_dissolve = gpd_block.dissolve()
    gpd_block_buffer_series = gpd_block_dissolve.buffer(0.001)
    gpd_block_buffer = gpd.GeoDataFrame(pd.DataFrame(gpd_block_buffer_series).T, geometry=gpd_block_buffer_series, crs='epsg:4326')

    gpd_block_buffer_clipped = gpd.clip(gpd_bound,gpd_block_buffer)
    gpd_block_left = gpd_block_buffer_clipped.symmetric_difference(gpd_bound)
    # gpd_block_left = gpd.GeoDataFrame(pd.DataFrame(gpd_block_left).T, geometry=gpd_block_left, crs='epsg:4326')
    gpd_block_left = gpd.GeoDataFrame(geometry=gpd_block_left, crs='epsg:4326')
    gpd_block_merged = pd.concat([gpd_block_left, gpd_block])
    gpd_block_merged = gpd_block_merged.reset_index(drop=True)

    return gpd_block_merged