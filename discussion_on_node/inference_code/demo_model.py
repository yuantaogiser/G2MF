import torch
import torch.nn as nn

from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F

from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential

class MMGIN(torch.nn.Module):
    def __init__(self, in_channels_poi, in_channels_block, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs_poi = torch.nn.ModuleList()
        self.convs_block = torch.nn.ModuleList()
        self.batch_norms_poi = torch.nn.ModuleList()
        self.batch_norms_block = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp_poi = Sequential(
                Linear(in_channels_poi, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv_poi = GINConv(mlp_poi, train_eps=True)
            self.convs_poi.append(conv_poi)
            self.batch_norms_poi.append(BatchNorm(hidden_channels))
            in_channels_poi = hidden_channels
        for i in range(num_layers):
            mlp_block = Sequential(
                Linear(in_channels_block, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv_block = GINConv(mlp_block, train_eps=True)
            self.convs_block.append(conv_block)
            self.batch_norms_block.append(BatchNorm(hidden_channels))
            in_channels_block = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.attention = Linear(hidden_channels * 2, 1)
        self.fuse = Linear(hidden_channels*2, hidden_channels*2)
        self.out = Linear(hidden_channels*2, out_channels)

    def forward(self, x, edge_index_x, batch_x, y, edge_index_y, batch_y):
        for conv, batch_norm in zip(self.convs_block, self.batch_norms_block):
            x = F.relu(batch_norm(conv(x, edge_index_x)))
            # x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch_x)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        for conv, batch_norm in zip(self.convs_poi, self.batch_norms_poi):
            y = F.relu(batch_norm(conv(y, edge_index_y)))
            # y = F.dropout(y, p=0.5, training=self.training)
        y = global_add_pool(y, batch_y)
        y = F.relu(self.batch_norm1(self.lin1(y)))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.lin2(y)

        x = torch.cat((x,y), 1)
        attention_weights = torch.nn.functional.sigmoid(self.attention(x))
        x = attention_weights * x

        x = self.fuse(x)
        x = F.relu(x)
        x = self.out(x)

        return F.softmax(x, dim=-1)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch)
        # x = scatter_add(x, batch, dim=0)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

class JointLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        初始化联合损失函数。
        参数:
        alpha (float): 用于平衡分类损失和模态对齐损失的权重因子。
        """
        super(JointLoss, self).__init__()
        self.alpha = alpha
        self.classification_loss = nn.CrossEntropyLoss()
        self.alignment_loss = nn.CosineEmbeddingLoss()

    def forward(self, pred_logits, true_labels, feature_mod1, feature_mod2):
        """
        计算联合损失。
        参数:
        pred_logits (Tensor): 分类任务的预测输出。
        true_labels (Tensor): 真实标签。
        feature_mod1 (Tensor): 第一模态的特征。
        feature_mod2 (Tensor): 第二模态的特征。
        返回:
        total_loss (Tensor): 总的损失值。
        """
        # 计算分类损失
        class_loss = self.classification_loss(pred_logits, true_labels)
        
        # 为余弦相似度损失准备标签（总是1，表示我们希望特征之间相似）
        target = torch.ones((feature_mod1.size(0),), device=feature_mod1.device)
        
        # 计算模态对齐损失
        align_loss = self.alignment_loss(feature_mod1, feature_mod2, target)
        
        # 组合损失
        total_loss = (1 - self.alpha) * class_loss + self.alpha * align_loss
        
        return total_loss


class InteractionLoss(nn.Module):
    def __init__(self, beta=0.5):
        """
        初始化交互损失函数。
        参数:
        beta (float): 用于控制交互损失在总损失中的比重。
        """
        super(InteractionLoss, self).__init__()
        self.beta = beta
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, pred_logits, true_labels, feature_mod1, feature_mod2):
        """
        计算包括交互损失的总损失。
        参数:
        pred_logits (Tensor): 分类任务的预测输出。
        true_labels (Tensor): 真实标签。
        feature_mod1 (Tensor): 第一模态的特征。
        feature_mod2 (Tensor): 第二模态的特征。
        返回:
        total_loss (Tensor): 总的损失值。
        """
        # 计算分类损失
        class_loss = self.classification_loss(pred_logits, true_labels)

        # 计算交互损失，使用点积衡量特征间的交互强度
        interaction = torch.sum(feature_mod1 * feature_mod2, dim=1)
        interaction_loss = -torch.mean(interaction)  # 最大化交互强度

        # 组合损失，调整正负号以确保交互项被最大化
        total_loss = class_loss - self.beta * interaction_loss

        return total_loss
