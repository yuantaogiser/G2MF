import os
import sys
import tempfile
import subprocess
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
from shapely.geometry import mapping

import matplotlib.pyplot as plt
import matplotlib as mpl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import folium

from utils_graph_construction import generate_graph_attributes_poi,generate_graph_attributes_sem, build_graph_by_poi, build_graph_by_sem, build_IDT_graph_by_poi, build_IDT_graph_by_sem
from utils_graph_calculate import LoadModel, PredictGraph
from utils_help import *
from utils_model import *
from utils_labels import *

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QAction, QInputDialog, QMessageBox, QPushButton, QLabel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal, QObject

# 全局变量
gdf_aoi, gdf_poi, gdf_sem = None, None, None
DT_Graph_poi, IDT_Graph_poi, MST_Graph_poi  = None, None, None
DT_Graph_sem, IDT_Graph_sem, MST_Graph_sem  = None, None, None
attribute_for_poi_count, attribute_for_sem_count = None, None
BOOL_AOI, BOOL_POI, BOOL_SEM = False, False, False
BOOL_SEMGRAPH_MST, BOOL_SEMGRAPH_DT, BOOL_SEMGRAPH_IDT, BOOL_PHYGRAPH_MST, BOOL_PHYGRAPH_DT, BOOL_PHYGRAPH_IDT = False, False, False, False, False, False
TYPE_GT, TYPE_G2MF_MST, TYPE_G2MF_DT, TYPE_G2MF_IDT, TYPE_SEM_MST, TYPE_SEM_DT, TYPE_SEM_IDT, TYPE_PHY_MST, TYPE_PHY_DT, TYPE_PHY_IDT  = '?','?','?','?','?','?','?','?','?','?'


def create_main_window():
    global LABEL1, LABEL2, LABEL3, LABEL4

    window = QMainWindow()
    window.setWindowTitle('An Intelligent Toolbox for Identifying Urban Functional Zone')
    window.setGeometry(100, 100, 1200, 800)

    main_widget = QWidget()
    window.setCentralWidget(main_widget)

    layout = QVBoxLayout(main_widget)

    # 创建 2x2 布局
    top_layout = QHBoxLayout()
    bottom_layout = QHBoxLayout()

    layout.addLayout(top_layout)
    layout.addLayout(bottom_layout)

    # 左上角 - 状态栏 - 再次切割成上下布局
    top_left_layout = QHBoxLayout()
    top_layout.addLayout(top_left_layout)

    # 右上角 - OSM 底图
    osm_view = QWebEngineView()
    top_layout.addWidget(osm_view)
    show_osm_map(osm_view)

    # 左上角上部 - 示例组件1（QLabel）
    LABEL1 = QLabel(StrLabel.str_label_1())
    LABEL1.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    LABEL1.setStyleSheet("background-color: white; padding: 10px; font-size: 30px;")
    top_left_layout.addWidget(LABEL1)

    # 左上角中部 - 示例组件2（QLabel）
    LABEL2 = QLabel(StrLabel.str_label_2())
    LABEL2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    LABEL2.setStyleSheet("background-color: white; padding: 10px; font-size: 30px;")
    top_left_layout.addWidget(LABEL2)

    # 左上角下部 - 示例组件3（QLabel）
    LABEL3 = QLabel(StrLabel.str_label_3())
    LABEL3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    LABEL3.setStyleSheet("background-color: white; padding: 10px; font-size: 30px;")
    top_left_layout.addWidget(LABEL3)

    # 左上角下部 - 示例组件4（QLabel）
    LABEL4 = QLabel(StrLabel.str_label_4())
    LABEL4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    LABEL4.setStyleSheet("background-color: white; padding: 10px; font-size: 30px;")
    top_left_layout.addWidget(LABEL4)

    # 左下角 - 可视化POI和Sem数据的饼状图
    poi_sem_count_view = QWebEngineView()
    bottom_layout.addWidget(poi_sem_count_view)

    # 右下角 - 卫星影像
    satellite_view = QWebEngineView()
    bottom_layout.addWidget(satellite_view)
    show_satellite_map(satellite_view)

    create_menu(window, osm_view, poi_sem_count_view, satellite_view)

    return window

def create_menu(window, osm_view, poi_sem_count_view, satellite_view):
    menubar = window.menuBar()

    file_menu = menubar.addMenu('File')
    open_aoi_action = QAction('Open AOI Shapefile', window)
    open_aoi_action.triggered.connect(lambda: open_aoi_shapefile(osm_view,satellite_view))
    open_poi_action = QAction('Open POI Shapefile', window)
    open_poi_action.triggered.connect(lambda: open_poi_shapefile(osm_view))
    open_sem_action = QAction('Open LandCover Shapefile', window)
    open_sem_action.triggered.connect(lambda: open_sem_shapefile(satellite_view))
    clear_poi_action = QAction('Clear OSM Screen', window)
    clear_poi_action.triggered.connect(lambda: clear_poi_shapefile(osm_view))
    clear_sem_action = QAction('Clear Satellite Screen', window)
    clear_sem_action.triggered.connect(lambda: clear_sem_shapefile(satellite_view))
    restart_action = QAction('Restart', window)
    restart_action.triggered.connect(lambda: restart_program())
    exit_action = QAction('EXIT', window)
    exit_action.triggered.connect(lambda: sys.exit())
    file_menu.addAction(open_aoi_action)
    file_menu.addAction(open_poi_action)
    file_menu.addAction(open_sem_action)
    file_menu.addAction(clear_poi_action)
    file_menu.addAction(clear_sem_action)
    file_menu.addAction(restart_action)
    file_menu.addAction(exit_action)

    view_menu = menubar.addMenu('View')
    colorPOI_action = QAction('Color POI by Attribute', window)
    colorSem_action = QAction('Color LC by Attribute', window)
    visual_poi_graph_action = QAction('Visualization for Semantic Graph', window)
    visual_LC_graph_action = QAction('Visualization for Physical Graph', window)
    visual_poi_count_action = QAction('Visualization for Semantic Count', window)
    visual_LC_count_action = QAction('Visualization for Physical Count', window)
    colorPOI_action.triggered.connect(lambda: color_poi_by_attribute(osm_view))
    colorSem_action.triggered.connect(lambda: color_sem_by_attribute(satellite_view))
    visual_poi_graph_action.triggered.connect(lambda: visual_poi_graph(osm_view))
    visual_LC_graph_action.triggered.connect(lambda: visual_sem_graph(satellite_view))
    visual_poi_count_action.triggered.connect(lambda: count_poi_by_attribute(poi_sem_count_view))
    visual_LC_count_action.triggered.connect(lambda: count_sem_by_attribute(poi_sem_count_view))
    view_menu.addAction(colorPOI_action)
    view_menu.addAction(colorSem_action)
    view_menu.addAction(visual_poi_graph_action)
    view_menu.addAction(visual_LC_graph_action)
    view_menu.addAction(visual_poi_count_action)
    view_menu.addAction(visual_LC_count_action)

    graph_construct_menu = menubar.addMenu('GraphConstruct')
    MST4S_action = QAction('Minimum Spanning Tree for Semantic', window)
    DT4S_action = QAction('Delaunay Triangulation for Semantic', window)
    IDT4S_action = QAction('Improved Delaunay Triangulation for Semantic', window)
    MST4P_action = QAction('Minimum Spanning Tree for Physical', window)
    DT4P_action = QAction('Delaunay Triangulation for Physical', window)
    IDT4P_action = QAction('Improved Delaunay Triangulation for Physical', window)
    MST4S_action.triggered.connect(lambda: semantic_graph_construction_MST(MST4S_action))
    DT4S_action.triggered.connect(lambda: semantic_graph_construction_DT(DT4S_action))
    IDT4S_action.triggered.connect(lambda: semantic_graph_construction_IDT(IDT4S_action))
    MST4P_action.triggered.connect(lambda: physical_graph_construction_MST(MST4P_action))
    DT4P_action.triggered.connect(lambda: physical_graph_construction_DT(DT4P_action))
    IDT4P_action.triggered.connect(lambda: physical_graph_construction_IDT(IDT4P_action))
    graph_construct_menu.addAction(MST4S_action)
    graph_construct_menu.addAction(DT4S_action)
    graph_construct_menu.addAction(IDT4S_action)
    graph_construct_menu.addAction(MST4P_action)
    graph_construct_menu.addAction(DT4P_action)
    graph_construct_menu.addAction(IDT4P_action)

    loadmodel_menu = menubar.addMenu('LoadModel')
    MM_action = QAction('Multimodal Model', window)
    SSM_action = QAction('Single Semantic Model', window)
    SPM_action = QAction('Single Physical Model', window)
    LCM_action = QAction('Load a Custom Model', window)
    MM_action.triggered.connect(lambda: LoadModel.load_mm_model(LABEL3))
    SSM_action.triggered.connect(lambda: LoadModel.load_ssm_model(LABEL3))
    SPM_action.triggered.connect(lambda: LoadModel.load_spm_model(LABEL3))
    LCM_action.triggered.connect(lambda: LoadModel.load_custom_model(LABEL3))
    loadmodel_menu.addAction(MM_action)
    loadmodel_menu.addAction(SSM_action)
    loadmodel_menu.addAction(SPM_action)
    loadmodel_menu.addAction(LCM_action)

    inference_menu = menubar.addMenu('Inference')
    G2MF_action = QAction('G2MF', window)
    SO_action = QAction('SemanticObject', window)
    PO_action = QAction('PhysicalObject', window)
    G2MF_action.triggered.connect(lambda: inference_multimodal(G2MF_action))
    SO_action.triggered.connect(lambda: inference_semantic_object(SO_action))
    PO_action.triggered.connect(lambda: inference_physical_object(PO_action))
    inference_menu.addAction(G2MF_action)
    inference_menu.addAction(SO_action)
    inference_menu.addAction(PO_action)

    batch_graph_construct_menu = menubar.addMenu('BatchGraphConstruct')
    batch_MST4S_action = QAction('Minimum Spanning Tree for Semantic', window)
    batch_DT4S_action = QAction('Delaunay Triangulation for Semantic', window)
    batch_IDT4S_action = QAction('Improved Delaunay Triangulation for Semantic', window)
    batch_MST4P_action = QAction('Minimum Spanning Tree for Physical', window)
    batch_DT4P_action = QAction('Delaunay Triangulation for Physical', window)
    batch_IDT4P_action = QAction('Improved Delaunay Triangulation for Physical', window)
    batch_MST4S_action.triggered.connect(lambda: batch_semantic_graph_construction_MST4S())
    batch_DT4S_action.triggered.connect(lambda: batch_semantic_graph_construction_DT4S())
    batch_IDT4S_action.triggered.connect(lambda: batch_semantic_graph_construction_IDT4S())
    batch_MST4P_action.triggered.connect(lambda: batch_physical_graph_construction_MST4S())
    batch_DT4P_action.triggered.connect(lambda: batch_physical_graph_construction_DT4S())
    batch_IDT4P_action.triggered.connect(lambda: batch_physical_graph_construction_IDT4S())
    batch_graph_construct_menu.addAction(batch_MST4S_action)
    batch_graph_construct_menu.addAction(batch_DT4S_action)
    batch_graph_construct_menu.addAction(batch_IDT4S_action)
    batch_graph_construct_menu.addAction(batch_MST4P_action)
    batch_graph_construct_menu.addAction(batch_DT4P_action)
    batch_graph_construct_menu.addAction(batch_IDT4P_action)

    batch_inference_menu = menubar.addMenu('BatchInference')
    batch_G2MF_action = QAction('G2MF', window)
    batch_SO_action = QAction('SemanticObject', window)
    batch_PO_action = QAction('PhysicalObject', window)
    batch_G2MF_action.triggered.connect(lambda: batch_inference_multimodal())
    batch_SO_action.triggered.connect(lambda: batch_inference_semantic_object())
    batch_PO_action.triggered.connect(lambda: batch_inference_physical_object())
    batch_inference_menu.addAction(batch_G2MF_action)
    batch_inference_menu.addAction(batch_SO_action)
    batch_inference_menu.addAction(batch_PO_action)

    help_menu = menubar.addMenu('Help')
    guide_action = QAction('Guideline', window)
    about_action = QAction('About', window)
    guide_action.triggered.connect(lambda: guideline())
    about_action.triggered.connect(lambda: about())
    help_menu.addAction(guide_action)
    help_menu.addAction(about_action)

def show_osm_map(web_view):
    folium_map = folium.Map(location=[0, 0], zoom_start=2)
    display_map(web_view, folium_map)

def show_satellite_map(web_view):
    folium_map = folium.Map(location=[0, 0], zoom_start=2)

    # 添加卫星图层
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(folium_map)

    display_map(web_view, folium_map)

def open_aoi_shapefile(osm_view, satellite_view):
    global gdf_aoi, BOOL_AOI
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Open AOI Shapefile", "", "Shapefiles (*.shp)", options=options)
    if file_path:
        gdf_aoi = gpd.read_file(file_path)
        if not any(x == 'Polygon' for x in gdf_aoi.geom_type):
            QMessageBox.warning(None, "Warning", "Please load a correct shapefile.")
        display_aoi_shapefile(osm_view, satellite_view, gdf_aoi)
        BOOL_AOI = True
        LABEL1.setText(StrLabel.str_label_1(BOOL_AOI=BOOL_AOI, BOOL_POI=BOOL_POI, BOOL_SEM=BOOL_SEM))

def open_poi_shapefile(osm_view):
    global gdf_poi, BOOL_POI
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Open POI Shapefile", "", "Shapefiles (*.shp)", options=options)
    if file_path:
        gdf_poi = gpd.read_file(file_path)
        if not any(x == 'Point' for x in gdf_poi.geom_type):
            QMessageBox.warning(None, "Warning", "Please load a correct shapefile.")
        display_poi_shapefile(osm_view, gdf_poi)
        BOOL_POI = True
        LABEL1.setText(StrLabel.str_label_1(BOOL_AOI=BOOL_AOI, BOOL_POI=BOOL_POI, BOOL_SEM=BOOL_SEM))

def open_sem_shapefile(satellite_view):
    global gdf_sem, BOOL_SEM
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Open LandCover Shapefile", "", "Shapefiles (*.shp)", options=options)
    if file_path:
        gdf_sem = gpd.read_file(file_path)
        if not any(x == 'Polygon' for x in gdf_sem.geom_type):
            QMessageBox.warning(None, "Warning", "Please load a correct shapefile.")
        display_sem_shapefile(satellite_view, gdf_sem)
        BOOL_SEM = True
        LABEL1.setText(StrLabel.str_label_1(BOOL_AOI=BOOL_AOI, BOOL_POI=BOOL_POI, BOOL_SEM=BOOL_SEM))

def display_aoi_shapefile(osm_view, satellite_view, gdf):
    # Ensure the shapefile is in EPSG:4326 CRS
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Transform to a projected CRS for centroid calculation
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid
    center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

    # Transform centroid back to geographic CRS
    centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
    centroid_geo = centroid_geo.to_crs(epsg=4326)
    center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

    folium_osm_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    for _, row in gdf.iterrows():
        sim_geo = mapping(row.geometry)
        geo_json = folium.GeoJson(data=sim_geo, style_function=lambda x: {'fillColor': 'blue', 'fillOpacity': 0})
        geo_json.add_to(folium_osm_map)

    # 卫星影像地图
    folium_satellite_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 添加卫星图层
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(folium_satellite_map)

    for _, row in gdf.iterrows():
        sim_geo = mapping(row.geometry)
        geo_json = folium.GeoJson(data=sim_geo, style_function=lambda x: {'fillColor': 'blue', 'fillOpacity': 0})
        geo_json.add_to(folium_satellite_map)

    display_map(osm_view, folium_osm_map)
    display_map(satellite_view, folium_satellite_map)

def display_poi_shapefile(osm_view, gdf):
    # Ensure the shapefile is in EPSG:4326 CRS
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Transform to a projected CRS for centroid calculation
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid
    center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

    # Transform centroid back to geographic CRS
    centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
    centroid_geo = centroid_geo.to_crs(epsg=4326)
    center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

    folium_osm_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    for _, row in gdf.iterrows():
        sim_geo = mapping(row.geometry)
        geo_json = folium.GeoJson(data=sim_geo, style_function=lambda x: {'fillColor': 'blue'})
        geo_json.add_to(folium_osm_map)
    display_map(osm_view, folium_osm_map)
    
def display_sem_shapefile(satellite_view, gdf):
    # Ensure the shapefile is in EPSG:4326 CRS
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Transform to a projected CRS for centroid calculation
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid
    center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

    # Transform centroid back to geographic CRS
    centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
    centroid_geo = centroid_geo.to_crs(epsg=4326)
    center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

    # 卫星影像地图
    folium_satellite_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 添加卫星图层
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(folium_satellite_map)

    for _, row in gdf.iterrows():
        sim_geo = mapping(row.geometry)
        geo_json = folium.GeoJson(data=sim_geo, style_function=lambda x: {'fillColor': 'blue'})
        geo_json.add_to(folium_satellite_map)
    display_map(satellite_view, folium_satellite_map)

def clear_poi_shapefile(osm_view):
    if not isinstance(gdf_poi, gpd.GeoDataFrame):
        QMessageBox.warning(None, "Warning", "Please load a POI shapefile.")
    else:
        gdf = gdf_poi
        # Ensure the shapefile is in EPSG:4326 CRS
        if gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs(epsg=4326)

        # Transform to a projected CRS for centroid calculation
        gdf_projected = gdf.to_crs(epsg=3857)
        centroid = gdf_projected.geometry.centroid
        center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

        # Transform centroid back to geographic CRS
        centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
        centroid_geo = centroid_geo.to_crs(epsg=4326)
        center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

        folium_osm_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        display_map(osm_view, folium_osm_map)

def clear_sem_shapefile(satellite_view):
    if not isinstance(gdf_sem, gpd.GeoDataFrame):
        QMessageBox.warning(None, "Warning", "Please load a LC shapefile.")
    else:
        gdf = gdf_sem
        # Ensure the shapefile is in EPSG:4326 CRS
        if gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs(epsg=4326)

        # Transform to a projected CRS for centroid calculation
        gdf_projected = gdf.to_crs(epsg=3857)
        centroid = gdf_projected.geometry.centroid
        center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

        # Transform centroid back to geographic CRS
        centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
        centroid_geo = centroid_geo.to_crs(epsg=4326)
        center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

        # 卫星影像地图
        folium_satellite_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # 添加卫星图层
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        ).add_to(folium_satellite_map)
        display_map(satellite_view, folium_satellite_map)

def color_poi_by_attribute(web_view):
    if gdf_poi is not None:
        attributes = gdf_poi.columns.tolist()
        attributes.remove('geometry')  # remove geometry column
        attribute, ok = QInputDialog.getItem(None, "Select Attribute", "Attribute:", attributes, 0, False)
        if ok and attribute:
            apply_coloring_poi(web_view, gdf_poi, attribute)
    else:
        QMessageBox.warning(None, "Warning", "Please load a POI shapefile.")

def color_sem_by_attribute(satellite_view):
    if gdf_sem is not None:
        attributes = gdf_sem.columns.tolist()
        attributes.remove('geometry')  # remove geometry column
        attribute, ok = QInputDialog.getItem(None, "Select Attribute", "Attribute:", attributes, 0, False)
        if ok and attribute:
            apply_coloring_sem(satellite_view, gdf_sem, attribute)
    else:
        QMessageBox.warning(None, "Warning", "Please load a LandCover shapefile.")

def apply_coloring_poi(osm_view, gdf, attribute):
    # Transform to a projected CRS for centroid calculation
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid
    center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

    # Transform centroid back to geographic CRS
    centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
    centroid_geo = centroid_geo.to_crs(epsg=4326)
    center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

    folium_osm_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Normalize the colors based on the attribute values
    min_val = gdf[attribute].min()
    max_val = gdf[attribute].max()
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for _, row in gdf.iterrows():
        location = [row.geometry.y, row.geometry.x]
        value = row[attribute]
        color = plt.cm.viridis(norm(value))
        hex_color = mpl.colors.to_hex(color)
        folium.CircleMarker(
            location=location,
            radius=8,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            weight=2
        ).add_to(folium_osm_map)

    labels = list(set(gdf[attribute].to_list()))
    POI_legend_show = POI_legend(labels, norm)
    POI_legend_show.add_to(folium_osm_map)
    display_map(osm_view, folium_osm_map)

def apply_coloring_sem(satellite_view, gdf, attribute):
    # Transform to a projected CRS for centroid calculation
    gdf_projected = gdf.to_crs(epsg=3857)
    centroid = gdf_projected.geometry.centroid
    center_lat, center_lon = centroid.y.mean(), centroid.x.mean()

    # Transform centroid back to geographic CRS
    centroid_geo = gpd.GeoSeries([gpd.points_from_xy([center_lon], [center_lat])[0]], crs='EPSG:3857')
    centroid_geo = centroid_geo.to_crs(epsg=4326)
    center_lat, center_lon = centroid_geo.geometry[0].y, centroid_geo.geometry[0].x

    # 卫星影像地图
    folium_satellite_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 添加卫星图层
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(folium_satellite_map)

    # Normalize the colors based on the attribute values
    min_val = gdf[attribute].min()
    max_val = gdf[attribute].max()
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for _, row in gdf.iterrows():
        sim_geo = mapping(row.geometry)

        value = row[attribute]
        color = plt.cm.Spectral(norm(value))
        hex_color = mpl.colors.to_hex(color)

        geo_json = folium.GeoJson(data=sim_geo, 
                                  style_function=lambda x, color=hex_color: {'fillColor': color, 'color': color, 'weight': 0.5, 'fillOpacity': 0.6})
        geo_json.add_to(folium_satellite_map)
    
    labels = list(set(gdf[attribute].to_list()))
    Sem_legend_show = Sem_legend(labels, norm)
    Sem_legend_show.add_to(folium_satellite_map)
    display_map(satellite_view, folium_satellite_map)

class WorkerSignals(QObject):
    finished = pyqtSignal()
class Loader_semantic_graph_construction_MST(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global DT_Graph_poi, MST_Graph_poi, DT_polygon_poi, poi_node
        global BOOL_SEMGRAPH_MST, BOOL_SEMGRAPH_DT
        if not isinstance(MST_Graph_poi, gpd.GeoDataFrame):
            if not isinstance(gdf_poi, gpd.GeoDataFrame):
                self.action.setEnabled(True)
                self.signals.finished.emit()
            else:
                DT_Graph_poi, MST_Graph_poi, DT_polygon_poi, poi_node = build_graph_by_poi(gdf_poi)
                BOOL_SEMGRAPH_MST, BOOL_SEMGRAPH_DT = True, True
                LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                    BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                    BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                    BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                    BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                    BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
                self.action.setEnabled(True)
                self.signals.finished.emit()
        else:
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_semantic_graph_construction_DT(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global DT_Graph_poi, MST_Graph_poi, DT_polygon_poi, poi_node
        global BOOL_SEMGRAPH_MST, BOOL_SEMGRAPH_DT
        if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
            if not isinstance(gdf_poi, gpd.GeoDataFrame):
                self.action.setEnabled(True)
                self.signals.finished.emit()
            else:
                DT_Graph_poi, MST_Graph_poi, DT_polygon_poi, poi_node = build_graph_by_poi(gdf_poi)
                BOOL_SEMGRAPH_MST, BOOL_SEMGRAPH_DT = True, True
                LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                    BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                    BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                    BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                    BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                    BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
                self.action.setEnabled(True)
                self.signals.finished.emit()
        else:
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_semantic_graph_construction_IDT(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global IDT_Graph_poi
        global BOOL_SEMGRAPH_IDT
        if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
            self.action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct DT graph first.")
        else:
            if not isinstance(IDT_Graph_poi, gpd.GeoDataFrame):
                IDT_Graph_poi = build_IDT_graph_by_poi(DT_Graph_poi, DT_polygon_poi)
            BOOL_SEMGRAPH_IDT = True
            LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_physical_graph_construction_MST(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global DT_Graph_sem, MST_Graph_sem, DT_polygon_sem, sem_node
        global BOOL_PHYGRAPH_MST, BOOL_PHYGRAPH_DT
        if not isinstance(MST_Graph_sem, gpd.GeoDataFrame):
            if not isinstance(gdf_sem, gpd.GeoDataFrame):
                self.action.setEnabled(True)
                self.signals.finished.emit()
            else:
                DT_Graph_sem, MST_Graph_sem, DT_polygon_sem, sem_node = build_graph_by_sem(gdf_sem)
                BOOL_PHYGRAPH_MST, BOOL_PHYGRAPH_DT = True, True
                LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                    BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                    BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                    BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                    BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                    BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
                self.action.setEnabled(True)
                self.signals.finished.emit()
        else:
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_physical_graph_construction_DT(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global DT_Graph_sem, MST_Graph_sem, DT_polygon_sem, sem_node
        global BOOL_PHYGRAPH_MST, BOOL_PHYGRAPH_DT
        if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
            if not isinstance(gdf_sem, gpd.GeoDataFrame):
                self.action.setEnabled(True)
                self.signals.finished.emit()
            else:
                DT_Graph_sem, MST_Graph_sem, DT_polygon_sem, sem_node = build_graph_by_sem(gdf_sem)
                BOOL_PHYGRAPH_MST, BOOL_PHYGRAPH_DT = True, True
                LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                    BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                    BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                    BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                    BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                    BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
                self.action.setEnabled(True)
                self.signals.finished.emit()
        else:
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_physical_graph_construction_IDT(QThread):
    def __init__(self,action):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
    def run(self):
        global IDT_Graph_sem
        global BOOL_PHYGRAPH_IDT
        if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
            self.action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct DT graph first.")
        else:
            if not isinstance(IDT_Graph_sem, gpd.GeoDataFrame):
                IDT_Graph_sem = build_IDT_graph_by_sem(DT_Graph_sem, DT_polygon_sem)
            BOOL_PHYGRAPH_IDT = True
            LABEL2.setText(StrLabel.str_label_2(BOOL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST,
                                                BOOL_SEMGRAPH_DT  = BOOL_SEMGRAPH_DT,
                                                BOOL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT,
                                                BOOL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST,
                                                BOOL_PHYGRAPH_DT  = BOOL_PHYGRAPH_DT,
                                                BOOL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()

def semantic_graph_construction_MST(action):
    action.setEnabled(False)
    if not isinstance(MST_Graph_poi, gpd.GeoDataFrame):
        if not isinstance(gdf_poi, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please load a poi shapefile first.")
        else:
            thread = Loader_semantic_graph_construction_MST(action)
            thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
            thread.start()
            QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
def semantic_graph_construction_DT(action):
    action.setEnabled(False)
    if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
        if not isinstance(gdf_poi, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please load a poi shapefile first.")
        else:
            thread = Loader_semantic_graph_construction_DT(action)
            thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
            thread.start()
            QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
def semantic_graph_construction_IDT(action):
    action.setEnabled(False)
    if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
        action.setEnabled(True)
        QMessageBox.warning(None, "Warning", "Please construct DT graph first.")
    else:
        thread = Loader_semantic_graph_construction_IDT(action)
        thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
        thread.start()
        QApplication.instance().threads.append(thread)  # Keep track of the thread
def physical_graph_construction_MST(action):
    action.setEnabled(False)
    if not isinstance(MST_Graph_sem, gpd.GeoDataFrame):
        if not isinstance(gdf_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please load a LC shapefile first.")
        else:
            thread = Loader_physical_graph_construction_MST(action)
            thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
            thread.start()
            QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
def physical_graph_construction_DT(action):
    action.setEnabled(False)
    if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
        if not isinstance(gdf_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please load a LC shapefile first.")
        else:
            thread = Loader_physical_graph_construction_DT(action)
            thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
            thread.start()
            QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
def physical_graph_construction_IDT(action):
    action.setEnabled(False)
    if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
        action.setEnabled(True)
        QMessageBox.warning(None, "Warning", "Please construct DT graph first.")
    else:
        thread = Loader_physical_graph_construction_IDT(action)
        thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
        thread.start()
        QApplication.instance().threads.append(thread)  # Keep track of the thread

class Loader_inference_multimodal(QThread):
    def __init__(self, action, button_txt):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
        self.button_txt = button_txt
    def run(self):
        global graph_attributes_MST_poi, graph_attributes_DT_poi, graph_attributes_IDT_poi, graph_attributes_MST_sem, graph_attributes_DT_sem, graph_attributes_IDT_sem
        global TYPE_GT, TYPE_G2MF_MST, TYPE_G2MF_DT, TYPE_G2MF_IDT
        if self.button_txt == 'MST':
            graph_attributes_MST_poi = generate_graph_attributes_poi(poi_node, MST_Graph_poi, gdf_aoi)
            graph_attributes_MST_sem = generate_graph_attributes_sem(sem_node, MST_Graph_sem, gdf_aoi)
            TYPE_G2MF_MST, TYPE_GT = PredictGraph.predict_graph_mm(graph_attributes_MST_poi, graph_attributes_MST_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        elif self.button_txt == 'DT':
            graph_attributes_DT_poi = generate_graph_attributes_poi(poi_node, DT_Graph_poi, gdf_aoi)
            graph_attributes_DT_sem = generate_graph_attributes_sem(sem_node, DT_Graph_sem, gdf_aoi)
            TYPE_G2MF_DT, TYPE_GT = PredictGraph.predict_graph_mm(graph_attributes_DT_poi, graph_attributes_DT_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        else:
            graph_attributes_IDT_poi = generate_graph_attributes_poi(poi_node, IDT_Graph_poi, gdf_aoi)
            graph_attributes_IDT_sem = generate_graph_attributes_sem(sem_node, IDT_Graph_sem, gdf_aoi)
            TYPE_G2MF_IDT, TYPE_GT = PredictGraph.predict_graph_mm(graph_attributes_IDT_poi, graph_attributes_IDT_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_inference_semantic_object(QThread):
    def __init__(self, action, button_txt):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
        self.button_txt = button_txt
    def run(self):
        global graph_attributes_MST_poi, graph_attributes_DT_poi, graph_attributes_IDT_poi
        global TYPE_GT, TYPE_SEM_MST, TYPE_SEM_DT, TYPE_SEM_IDT
        if self.button_txt == 'MST':
            graph_attributes_MST_poi = generate_graph_attributes_poi(poi_node, MST_Graph_poi, gdf_aoi)
            TYPE_SEM_MST, TYPE_GT = PredictGraph.predict_graph_poi(graph_attributes_MST_poi)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        elif self.button_txt == 'DT':
            graph_attributes_DT_poi = generate_graph_attributes_poi(poi_node, DT_Graph_poi, gdf_aoi)
            TYPE_SEM_DT, TYPE_GT = PredictGraph.predict_graph_poi(graph_attributes_DT_poi)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        else:
            graph_attributes_IDT_poi = generate_graph_attributes_poi(poi_node, IDT_Graph_poi, gdf_aoi)
            TYPE_SEM_IDT, TYPE_GT = PredictGraph.predict_graph_poi(graph_attributes_IDT_poi)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
class Loader_inference_physical_object(QThread):
    def __init__(self, action, button_txt):
        super().__init__()
        self.signals = WorkerSignals()
        self.action = action
        self.button_txt = button_txt
    def run(self):
        global graph_attributes_MST_sem, graph_attributes_DT_sem, graph_attributes_IDT_sem
        global TYPE_GT, TYPE_PHY_MST, TYPE_PHY_DT, TYPE_PHY_IDT
        if self.button_txt == 'MST':
            graph_attributes_MST_sem = generate_graph_attributes_sem(sem_node, MST_Graph_sem, gdf_aoi)
            TYPE_PHY_MST, TYPE_GT = PredictGraph.predict_graph_sem(graph_attributes_MST_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        elif self.button_txt == 'DT':
            graph_attributes_DT_sem = generate_graph_attributes_sem(sem_node, DT_Graph_sem, gdf_aoi)
            TYPE_PHY_DT, TYPE_GT = PredictGraph.predict_graph_sem(graph_attributes_DT_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()
        else:
            graph_attributes_IDT_sem = generate_graph_attributes_sem(sem_node, IDT_Graph_sem, gdf_aoi)
            TYPE_PHY_IDT, TYPE_GT = PredictGraph.predict_graph_sem(graph_attributes_IDT_sem)
            LABEL4.setText(StrLabel.str_label_4(TYPE_GT       = TYPE_GT,
                                                TYPE_G2MF_MST = TYPE_G2MF_MST,
                                                TYPE_G2MF_DT  = TYPE_G2MF_DT,
                                                TYPE_G2MF_IDT = TYPE_G2MF_IDT,
                                                TYPE_SEM_MST  = TYPE_SEM_MST,
                                                TYPE_SEM_DT   = TYPE_SEM_DT,
                                                TYPE_SEM_IDT  = TYPE_SEM_IDT,
                                                TYPE_PHY_MST  = TYPE_PHY_MST,
                                                TYPE_PHY_DT   = TYPE_PHY_DT,
                                                TYPE_PHY_IDT  = TYPE_PHY_IDT))
            self.action.setEnabled(True)
            self.signals.finished.emit()

def inference_multimodal(action):
    action.setEnabled(False)
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choose a mode for inference')
    msg_box.setText('Choose an graph mode:')
    button_MST = QPushButton('MST')
    button_DT = QPushButton('DT')
    button_IDT = QPushButton('IDT')
    msg_box.addButton(button_MST, QMessageBox.AcceptRole)
    msg_box.addButton(button_DT, QMessageBox.AcceptRole)
    msg_box.addButton(button_IDT, QMessageBox.AcceptRole)

    msg_box.exec_()
    if msg_box.clickedButton() == button_MST:
        if not isinstance(MST_Graph_poi, gpd.GeoDataFrame) or not isinstance(MST_Graph_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_multimodal(action, button_MST.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_DT:
        if not isinstance(DT_Graph_poi, gpd.GeoDataFrame) or not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_multimodal(action, button_DT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_IDT:
        if not isinstance(IDT_Graph_poi, gpd.GeoDataFrame) or not isinstance(IDT_Graph_sem, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct IDT graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_multimodal(action, button_IDT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
        QMessageBox.warning(None, "Warning", "Holy Sh*t.")

def inference_semantic_object(action):
    action.setEnabled(False)
    global graph_attributes_MST_poi, graph_attributes_DT_poi, graph_attributes_IDT_poi
    global TYPE_GT, TYPE_SEM_MST, TYPE_SEM_DT, TYPE_SEM_IDT
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choose a mode for inference')
    msg_box.setText('Choose an graph mode:')
    button_MST = QPushButton('MST')
    button_DT = QPushButton('DT')
    button_IDT = QPushButton('IDT')
    msg_box.addButton(button_MST, QMessageBox.AcceptRole)
    msg_box.addButton(button_DT, QMessageBox.AcceptRole)
    msg_box.addButton(button_IDT, QMessageBox.AcceptRole)

    msg_box.exec_()
    if msg_box.clickedButton() == button_MST:
        if not isinstance(MST_Graph_poi, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_semantic_object(action, button_MST.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_DT:
        if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_semantic_object(action, button_DT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_IDT:
        if not isinstance(IDT_Graph_poi, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct IDT graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_semantic_object(action, button_IDT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
        QMessageBox.warning(None, "Warning", "Holy Sh*t.")

def inference_physical_object(action):
    action.setEnabled(False)
    global graph_attributes_MST_sem, graph_attributes_DT_sem, graph_attributes_IDT_sem
    global TYPE_GT, TYPE_PHY_MST, TYPE_PHY_DT, TYPE_PHY_IDT
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choose a mode for inference')
    msg_box.setText('Choose an graph mode:')
    button_MST = QPushButton('MST')
    button_DT = QPushButton('DT')
    button_IDT = QPushButton('IDT')
    msg_box.addButton(button_MST, QMessageBox.AcceptRole)
    msg_box.addButton(button_DT, QMessageBox.AcceptRole)
    msg_box.addButton(button_IDT, QMessageBox.AcceptRole)

    msg_box.exec_()
    if msg_box.clickedButton() == button_MST:
        if not isinstance(MST_Graph_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_physical_object(action, button_MST.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_DT:
        if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_physical_object(action, button_DT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    elif msg_box.clickedButton() == button_IDT:
        if not isinstance(IDT_Graph_sem, gpd.GeoDataFrame):
            action.setEnabled(True)
            QMessageBox.warning(None, "Warning", "Please construct IDT graph first.")
        else:
            if not isinstance(gdf_aoi, gpd.GeoDataFrame):
                action.setEnabled(True)
                QMessageBox.warning(None, "Warning", "Please load a aoi shapefile first.")
            else:
                thread = Loader_inference_physical_object(action, button_IDT.text())
                thread.finished.connect(thread.deleteLater)  # Ensure thread cleanup
                thread.start()
                QApplication.instance().threads.append(thread)  # Keep track of the thread
    else:
        action.setEnabled(True)
        QMessageBox.warning(None, "Warning", "Holy Sh*t.")

def batch_semantic_graph_construction_MST4S():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_semantic_graph_construction_DT4S():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_semantic_graph_construction_IDT4S():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_physical_graph_construction_MST4S():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_physical_graph_construction_DT4S():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_physical_graph_construction_IDT4S():
    QMessageBox.information(None, "Developing...", "Developing...")

def batch_inference_multimodal():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_inference_semantic_object():
    QMessageBox.information(None, "Developing...", "Developing...")
def batch_inference_physical_object():
    QMessageBox.information(None, "Developing...", "Developing...")

def visual_poi_graph(osm_view):
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choose a mode for visualization')
    msg_box.setText('Choose an action:')
    button_MST = QPushButton('MST')
    button_DT = QPushButton('DT')
    button_IDT = QPushButton('IDT')
    msg_box.addButton(button_MST, QMessageBox.AcceptRole)
    msg_box.addButton(button_DT, QMessageBox.AcceptRole)
    msg_box.addButton(button_IDT, QMessageBox.AcceptRole)

    msg_box.exec_()
    if msg_box.clickedButton() == button_MST:
        if not isinstance(MST_Graph_poi, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(osm_view, MST_Graph_poi)
    elif msg_box.clickedButton() == button_DT:
        if not isinstance(DT_Graph_poi, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(osm_view, DT_Graph_poi)
    elif msg_box.clickedButton() == button_IDT:
        if not isinstance(IDT_Graph_poi, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(osm_view, IDT_Graph_poi)
    else:
        QMessageBox.warning(None, "Warning", "Holy Sh*t.")

def visual_sem_graph(satellite_view):
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choose a mode for visualization')
    msg_box.setText('Choose an action:')
    button_MST = QPushButton('MST')
    button_DT = QPushButton('DT')
    button_IDT = QPushButton('IDT')
    msg_box.addButton(button_MST, QMessageBox.AcceptRole)
    msg_box.addButton(button_DT, QMessageBox.AcceptRole)
    msg_box.addButton(button_IDT, QMessageBox.AcceptRole)

    msg_box.exec_()
    if msg_box.clickedButton() == button_MST:
        if not isinstance(MST_Graph_sem, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(satellite_view, MST_Graph_sem)
    elif msg_box.clickedButton() == button_DT:
        if not isinstance(DT_Graph_sem, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(satellite_view, DT_Graph_sem)
    elif msg_box.clickedButton() == button_IDT:
        if not isinstance(IDT_Graph_sem, gpd.GeoDataFrame):
            QMessageBox.warning(None, "Warning", "Please construct graph first.")
        else:
            display_poi_shapefile(satellite_view, IDT_Graph_sem)
    else:
        QMessageBox.warning(None, "Warning", "Holy Sh*t.")

def count_poi_by_attribute(web_view):
    global gdf_poi, attribute_for_poi_count
    if gdf_poi is not None:
        attributes = gdf_poi.columns.tolist()
        attributes.remove('geometry')  # remove geometry column
        attribute_for_poi_count, ok = QInputDialog.getItem(None, "Select Attribute", "Attribute:", attributes, 0, False)
        if ok and attribute_for_poi_count:
            visual_poi_count(web_view)
    else:
        QMessageBox.warning(None, "Warning", "Please load a POI shapefile.")

def count_sem_by_attribute(web_view):
    global gdf_sem, attribute_for_sem_count
    if gdf_sem is not None:
        attributes = gdf_sem.columns.tolist()
        attributes.remove('geometry')  # remove geometry column
        attribute_for_sem_count, ok = QInputDialog.getItem(None, "Select Attribute", "Attribute:", attributes, 0, False)
        if ok and attribute_for_sem_count:
            visual_poi_count(web_view)
    else:
        QMessageBox.warning(None, "Warning", "Please load a LC shapefile.")

def visual_poi_count(poi_sem_count_view):
    fig = create_pie_charts()
    html_file = create_html_file(fig)
    display_pie(poi_sem_count_view, html_file)

def create_pie_charts():
    if gdf_poi is not None and attribute_for_poi_count != None:
        gdf_poi_labels_all = gdf_poi[attribute_for_poi_count].tolist()
        gdf_poi_labels = list(set(gdf_poi_labels_all))
        gdf_poi_label_values = []
        for gdf_poi_label in gdf_poi_labels:
            tmp_value = len(gdf_poi[attribute_for_poi_count][gdf_poi[attribute_for_poi_count] == gdf_poi_label])
            gdf_poi_label_values.append(tmp_value)
    else:
        gdf_poi_labels, gdf_poi_label_values = None, None
    
    if gdf_sem is not None and attribute_for_sem_count != None:
        gdf_sem_labels_all = gdf_sem[attribute_for_sem_count].tolist()
        gdf_sem_labels = list(set(gdf_sem_labels_all))
        gdf_sem_label_values = []
        for gdf_sem_label in gdf_sem_labels:
            tmp_value = len(gdf_sem[attribute_for_sem_count][gdf_sem[attribute_for_sem_count] == gdf_sem_label])
            gdf_sem_label_values.append(tmp_value)
    else:
        gdf_sem_labels, gdf_sem_label_values = None, None

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=gdf_poi_labels, values=gdf_poi_label_values, name="SemanticObject"), 1, 1)
    fig.add_trace(go.Pie(labels=gdf_sem_labels, values=gdf_sem_label_values, name="PhysicalObject"), 1, 2)

    fig.update_layout(
        title_text="Count Semantic Objects & Physical Objects",
        annotations=[dict(text='SemanticObject', x=0.12, y=1.05, font_size=15, showarrow=False),
                     dict(text='PhysicalObject', x=0.83, y=1.05, font_size=15, showarrow=False)],
        showlegend=True
    )

    # Update layout to ensure each pie chart has its own legend
    fig['layout']['annotations'] += tuple(
        dict(
            x=x, y=y,
            xref='paper', yref='paper',
            showarrow=False,
            text='',
            font=dict(size=10)
        ) for x, y in zip([0.15, 0.85], [0.2, 0.2])
    )
    return fig

def create_html_file(fig):
    html_content = pio.to_html(fig, full_html=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_file.write(html_content.encode())
    temp_file.close()
    return temp_file.name

def display_pie(web_view, html_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        web_view.setUrl(QUrl.fromLocalFile(html_file))

def display_map(web_view, folium_map):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        folium_map.save(f.name)
        web_view.setUrl(QUrl.fromLocalFile(os.path.abspath(f.name)))

def restart_program():
    cmd = subprocess.list2cmdline([sys.executable] + sys.argv)
    os.execl(sys.executable, *cmd.split())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.threads = []  # Add this line to store references to threads
    window = create_main_window()
    window.show()
    sys.exit(app.exec_())
