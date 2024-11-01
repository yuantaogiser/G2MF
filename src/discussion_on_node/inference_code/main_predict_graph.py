import torch
from mydataset import MyDataset
from torch_geometric.loader import DataLoader

import numpy as np

from demo_model import *


if __name__ == "__main__":

    dataset_name_list = [
                    'bjshOnlyBuildingNodeWithoutFeature',
                    'bjshOnlyBuildingNodeWithFeature',
                    'bjshNoAllNodeFeature', 
                    'bjshAllNodeFeature', 
                    'bjshAllNodeOnlyWithBuildingFeature', 
                     ]

    dataset_name = dataset_name_list[0]

    model = torch.load(f'./trained_model/fusion_{dataset_name}.pth')

    model.eval()
    model.to('cuda')

    dataset_block = MyDataset(root='graph_data/mydataset_revision', name=f'block{dataset_name}')
    dataset_poi = MyDataset(root='graph_data/mydataset_revision', name=f'poi{dataset_name}')

    test_dataset_block = dataset_block
    test_dataset_poi = dataset_poi

    test_loader_poi = DataLoader(test_dataset_poi, batch_size=1, shuffle=False)
    test_loader_block = DataLoader(test_dataset_block, batch_size=1, shuffle=False)

    from demo_utils import ConfusionMatrix
    confmat = ConfusionMatrix(num_classes=6)

    for data_block, data_poi in zip(test_loader_block, test_loader_poi):  # Iterate in batches over the training dataset.
        
        data_block, data_poi = data_block.to('cuda'), data_poi.to('cuda')
        output = model(data_block.x, data_block.edge_index, data_block.batch,
                       data_poi.x, data_poi.edge_index, data_poi.batch)
        
        predict = output.argmax(1)

        ground_true = data_block.y[0]

        target = data_block.y.cpu().numpy().tolist()[0]
        confmat.update(ground_true.flatten(), output.argmax(1).flatten())

    confmat.reduce_from_all_processes()
    print('OA: ', confmat.compute()[0].cpu().numpy().flatten()[0])
    print('Macro-average precision: ', np.mean(confmat.compute()[1].cpu().numpy().flatten()))
    print('Macro-average recall: ', np.mean(confmat.compute()[2].cpu().numpy().flatten()))
    print('Macro-average F1-score: ', np.mean(confmat.compute()[3].cpu().numpy().flatten()))
    print(confmat.mat)

