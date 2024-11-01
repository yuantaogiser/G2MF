import geopandas as gpd
import pandas as pd
import torch
from torch_geometric.utils import one_hot
from utils_labels import StrLabel
from utils_model import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QAction, QInputDialog, QMessageBox, QPushButton
import warnings
warnings.filterwarnings('ignore')

from typing import List, Optional
from torch import Tensor


def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None

def node_labels_to_x(node_labels):
    node_label = torch.tensor(node_labels).to(torch.int64)
    if node_label.dim() == 1:
        node_label = node_label.unsqueeze(-1)
    # node_label = node_label - node_label.min(dim=0)[0]
    node_label = node_label - 0 # 2024年5月27日21:49:55，UFZ中可能不存在全局最小
    node_labels = node_label.unbind(dim=-1)
    node_labels = [one_hot(x,num_classes=6) for x in node_labels]
    if len(node_labels) == 1:
        node_label = node_labels[0]
    else:
        node_label = torch.cat(node_labels, dim=-1)
    return node_label

def node_labels_attrbutes_to_x(node_labels, node_attributes):
    node_label = torch.tensor(node_labels).to(torch.int64)
    if node_label.dim() == 1:
        node_label = node_label.unsqueeze(-1)
    # node_label = node_label - node_label.min(dim=0)[0]
    node_label = node_label - 0 # 2024年5月27日21:49:55，UFZ中可能不存在全局最小
    node_labels = node_label.unbind(dim=-1)
    node_labels = [one_hot(x,num_classes=9) for x in node_labels]
    if len(node_labels) == 1:
        node_label = node_labels[0]
    else:
        node_label = torch.cat(node_labels, dim=-1)
    
    node_attributes = torch.tensor(node_attributes)
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    x = cat([node_attributes, node_label])
    return x

def construct_graph_poi(data):
    A_matrix, graph_indicator, graph_labels, node_labels = data
    poi_x = node_labels_to_x(node_labels)
    poi_graph = [poi_x, A_matrix,graph_indicator, graph_labels]
    return poi_graph

def construct_graph_sem(data):
    A_matrix, graph_indicator, graph_labels, node_labels, node_attributes = data
    sem_x = node_labels_attrbutes_to_x(node_labels, node_attributes)
    sem_graph = [sem_x, A_matrix, graph_indicator, graph_labels]
    return sem_graph

class PredictGraph:
    def predict_graph_mm(data_poi, data_sem):
        if not isinstance(data_poi, list):
            QMessageBox.warning(None, "Warning", "Please construct semantic graph first.")
        if not isinstance(data_sem, list):
            QMessageBox.warning(None, "Warning", "Please construct physical graph first.")
        poi_graph = construct_graph_poi(data_poi)
        sem_graph = construct_graph_sem(data_sem)
        if not isinstance(model_mm, MMGIN):
            QMessageBox.warning(None, "Warning", "Please load a multimodal model first.")
            return '?', '?'
        else:
            model_mm.eval()
            model_mm.to('cuda')
            sem_x, sem_x_edge, sem_x_batch = sem_graph[0].to('cuda'), torch.tensor(sem_graph[1].T,dtype=torch.int64).to('cuda'), torch.tensor(sem_graph[2],dtype=torch.int64).to('cuda')
            sem_x_edge = sem_x_edge - 1
            poi_x, poi_edge, poi_batch = poi_graph[0].to('cuda'), torch.tensor(poi_graph[1].T,dtype=torch.int64).to('cuda'), torch.tensor(poi_graph[2],dtype=torch.int64).to('cuda')
            poi_edge = poi_edge - 1
            output = model_mm(sem_x, sem_x_edge, sem_x_batch, poi_x, poi_edge, poi_batch)
            result = output.argmax(1).flatten().detach().cpu().numpy()[0]
            gt = poi_graph[3]
            return result, gt

    def predict_graph_poi(data):
        if not isinstance(data, list):
            QMessageBox.warning(None, "Warning", "Please construct semantic graph first.")
        else:
            poi_graph = construct_graph_poi(data)
        if not isinstance(model_ssm, GIN):
            QMessageBox.warning(None, "Warning", "Please load a correct model.")
            return '?', '?'
        else:
            model_ssm.eval()
            model_ssm.to('cuda')
            poi_x, poi_edge, poi_batch = poi_graph[0].to('cuda'), torch.tensor(poi_graph[1].T,dtype=torch.int64).to('cuda'), torch.tensor(poi_graph[2],dtype=torch.int64).to('cuda')
            poi_edge = poi_edge - 1
            output = model_ssm(poi_x, poi_edge, poi_batch)
            result = output.argmax(1).flatten().detach().cpu().numpy()[0]
            gt = poi_graph[3]
            return result, gt

    def predict_graph_sem(data):
        if not isinstance(data, list):
            QMessageBox.warning(None, "Warning", "Please construct semantic graph first.")
        else:
            sem_graph = construct_graph_sem(data)
        if not isinstance(model_spm, GIN):
            QMessageBox.warning(None, "Warning", "Please load a correct model.")
            return '?', '?'
        else:
            model_spm.eval()
            model_spm.to('cuda')
            sem_x, sem_edge, sem_batch = sem_graph[0].to('cuda'), torch.tensor(sem_graph[1].T,dtype=torch.int64).to('cuda'), torch.tensor(sem_graph[2],dtype=torch.int64).to('cuda')
            sem_edge = sem_edge - 1
            output = model_spm(sem_x, sem_edge, sem_batch)
            result = output.argmax(1).flatten().detach().cpu().numpy()[0]
            gt = sem_graph[3]
            return result, gt

class LoadModel:
    global BOOL_G2MF_MODEL, BOOL_SEM_MODEL, BOOL_PHY_MODEL, BOOL_OWN_MODEL
    BOOL_G2MF_MODEL, BOOL_SEM_MODEL, BOOL_PHY_MODEL, BOOL_OWN_MODEL = False, False, False, False
    global model_mm, model_ssm, model_spm
    model_mm, model_ssm, model_spm = False, False, False
    def load_mm_model(LABEL3):
        global model_mm, BOOL_G2MF_MODEL
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(None, "Load a multimodal model", "", "PyTorch (*.pth)", options=options)
        if file_path:
            model_mm = torch.load(file_path)
            if not model_mm._get_name() == 'MMGIN':
                QMessageBox.warning(None, "Warning", "Please load a correct model.")
            BOOL_G2MF_MODEL = True
            LABEL3.setText(StrLabel.str_label_3(BOOL_G2MF_MODEL=BOOL_G2MF_MODEL, 
                                                BOOL_SEM_MODEL=BOOL_SEM_MODEL, 
                                                BOOL_PHY_MODEL=BOOL_PHY_MODEL,
                                                BOOL_OWN_MODEL=BOOL_OWN_MODEL))
    def load_ssm_model(LABEL3):
        global model_ssm, BOOL_SEM_MODEL
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(None, "Load a single semantic model", "", "PyTorch (*.pth)", options=options)
        if file_path:
            model_ssm = torch.load(file_path)
            if not model_ssm._get_name() == 'GIN':
                QMessageBox.warning(None, "Warning", "Please load a correct model.")
            BOOL_SEM_MODEL = True
            LABEL3.setText(StrLabel.str_label_3(BOOL_G2MF_MODEL=BOOL_G2MF_MODEL, 
                                                BOOL_SEM_MODEL=BOOL_SEM_MODEL, 
                                                BOOL_PHY_MODEL=BOOL_PHY_MODEL,
                                                BOOL_OWN_MODEL=BOOL_OWN_MODEL))
    def load_spm_model(LABEL3):
        global model_spm, BOOL_PHY_MODEL
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(None, "Load a single physical model", "", "PyTorch (*.pth)", options=options)
        if file_path:
            model_spm = torch.load(file_path)
            if not model_spm._get_name() == 'GIN':
                QMessageBox.warning(None, "Warning", "Please load a correct model.")
            BOOL_PHY_MODEL = True
            LABEL3.setText(StrLabel.str_label_3(BOOL_G2MF_MODEL=BOOL_G2MF_MODEL, 
                                                BOOL_SEM_MODEL=BOOL_SEM_MODEL, 
                                                BOOL_PHY_MODEL=BOOL_PHY_MODEL,
                                                BOOL_OWN_MODEL=BOOL_OWN_MODEL))

    def load_custom_model(LABEL3):
        global model_custom, BOOL_OWN_MODEL
        QMessageBox.information(None, "Developing...", "Developing...")
        # options = QFileDialog.Options()
        # options |= QFileDialog.ReadOnly
        # file_path, _ = QFileDialog.getOpenFileName(None, "Load a single physical model", "", "PyTorch (*.pth)", options=options)
        # if file_path:
        #     model_spm = torch.load(file_path)
        #     if not model_spm._get_name() == 'GIN':
        #         QMessageBox.warning(None, "Warning", "Please load a correct model.")
        BOOL_OWN_MODEL = True
        LABEL3.setText(StrLabel.str_label_3(BOOL_G2MF_MODEL=BOOL_G2MF_MODEL, 
                                            BOOL_SEM_MODEL=BOOL_SEM_MODEL, 
                                            BOOL_PHY_MODEL=BOOL_PHY_MODEL,
                                            BOOL_OWN_MODEL=BOOL_OWN_MODEL))
        