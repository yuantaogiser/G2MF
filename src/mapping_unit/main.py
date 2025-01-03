# -*- coding: utf-8 -*-
from utils import *
from os.path import join

import multiprocessing
import warnings
warnings.filterwarnings('ignore')

def process_body(path_workspace, path_database_name, path_osm_road,
                 path_administrative_filename, path_output, filename_output,
                 smooth_level: int = 30, extend_distance: int = 100, spike_keep: int = 500):
    """
    处理的主题函数
    """
    os.makedirs(path_workspace, exist_ok=True)
    try:
        arcpy.CreateFileGDB_management(path_workspace, path_database_name)
    except:
        print("gdb already exist")
    arcpy.env.workspace = join(path_workspace, path_database_name)
    arcpy.env.outputMFlag = "DISABLE_M_VALUE" 
    arcpy.env.overwriteOutput = True
    print(f"config workspace : {arcpy.env.workspace}")

    # 1. Clip & Select
    osm_clip = Clip_Road(path_administrative_filename, path_osm_road)
    select_osm = Select_Road(osm_clip)
    raw_road = select_osm

    # 2. Convert_Centerline
    #    It is the most core operation.
    centerline = Convert_Centerline(select_osm, smooth_level)

    # 3. Extend road
    extendedlines = Extend_Road(centerline, extend_distance)

    # 4. Check topo
    err_point = Check_Topo(extendedlines)

    # 5. Clean spike
    master_road = Clean_Spike(extendedlines, err_point, spike_keep, path_output, keep_spike=False)

    # 6. Join attributes
    master_road = Join_Attributes(raw_road, master_road)
    road_result = Delete_Fields(master_road)

    # 7. Create mapping unit
    road_result = "road_results"
    mapping_unit = Create_mapping_unit(road_result, path_administrative_filename, path_output, filename_output)

    return mapping_unit


def parallel_body(data):
    path_workspace, path_database_name, path_osm_road, path_administrative, path_output_folder, shpfile = data

    filename = shpfile.split('.')[0]
    filename_output = filename.lower()

    path_output = os.path.join(path_output_folder, filename_output)
    os.makedirs(path_output, exist_ok=True)
    path_administrative_filename = os.path.join(path_administrative, shpfile)

    output_path = process_body(path_workspace, path_database_name, path_osm_road,
                                path_administrative_filename, path_output, filename_output,
                                smooth_level=50, extend_distance=200, spike_keep=2000)
    print(output_path,"--> Done!")

if __name__ == '__main__':

    # gbd文件根目录
    path_workspace = '/path/to/folder/root'

    # OSM文件路径
    path_osm_road = '/path/to/folder/osm_road'

    # 边界数据
    path_administrative = '/path/to/folder/administrative_boundary'

    # 输出路径
    path_output_folder = '/path/to/folder/output'
    

    tmp_filename_list = os.listdir(path_administrative)
    shpfile_list = []
    for tmp in tmp_filename_list:
        if '.shp' in tmp:
            shpfile_list.append(tmp)

    data_list = []
    for shpfile in shpfile_list:
        path_database_name = shpfile.split('.')[0].lower() + '.gdb'
        data_list.append([path_workspace, path_database_name, path_osm_road, path_administrative, path_output_folder, shpfile])

    # parallel body
    core_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=core_num)
    pool.map_async(parallel_body, data_list).get()
    pool.close()
    pool.join()
