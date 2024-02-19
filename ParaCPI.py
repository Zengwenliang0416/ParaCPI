import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict

'''
使用A4+A5
'''


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)





class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, epochs, steps_per_epoch,n,num_input_features, out_dim):
        super().__init__()
        self.dropout_late = math.ceil((epochs * steps_per_epoch) / n)
        self.convs = nn.ModuleList([gnn.GraphConv(num_input_features, out_dim) for _ in range(5)])
        self.norm = NodeLevelBatchNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifer = nn.Linear(out_dim, 96)

    def forward(self, data,i):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = sum([conv(x, edge_index) for conv in self.convs])
        x = self.norm(x)
        x = gnn.global_max_pool(x, data.batch)
        x = F.relu(x)
        if i >= (self.dropout_late)//2:
            x = self.dropout(x)
            x = self.classifer(x)
        else:
            x = self.classifer(x)

        return x

class ProteinTransformerEncoder(nn.Module):
    def __init__(self, embedding_num, out_dim, num_layers=6, num_heads=8):
        super().__init__()
        self.embedding_num = embedding_num
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_num, num_heads), num_layers)
        self.linear = nn.Linear(embedding_num, out_dim)
        self.embedding = nn.Embedding(26, embedding_num)



    def forward(self, protein_features):
        # 将蛋白质序列特征嵌入到[batch_size, protein_len, embedding_num]的向量空间中
        # embeddings = self.embedding(protein_features)
        embeddings = self.embedding(protein_features)

        # 将嵌入向量输入Transformer编码器
        encoded = self.transformer(embeddings)

        # 获取编码后的向量，并转换为[batch_size,out_dim]的向量空间
        encoded = encoded.mean(dim=1)
        encoded = self.linear(encoded)

        return encoded
class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x
class ParaCPI(nn.Module):
    def __init__(self, epochs, steps_per_epoch,n,filter_num=32, out_dim=2, drop_rate = 0.2):
        super().__init__()
        embedding_num = 64
        hid_dim = 96
        protein_len = 1200
        self.dropout_late = math.ceil((epochs * steps_per_epoch) / n)
        self.protein_encoder = TargetRepresentation(block_num=3, vocab_size=25 + 1, embedding_num=128)

        self.ligand_encoder = GraphDenseNet(epochs, steps_per_epoch,n,num_input_features=87, out_dim=228)

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(324, 1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, out_dim)
        )

    def forward(self, data,i=0):
        target = data.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data,i)

        x = torch.cat([protein_x, ligand_x], dim=-1)
        if i >= (self.dropout_late) // 2:
            x = self.classifier(x)
        else:
            x = self.classifier(x)

        return x


