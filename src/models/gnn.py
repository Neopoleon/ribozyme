"""Graph Neural Network models for RNA classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    NNConv,
    global_mean_pool,
    global_max_pool,
)


class RNAGCN(nn.Module):
    """
    Graph Convolutional Network for RNA classification.

    Uses GCN layers followed by global pooling and MLP classifier.
    Note: GCN does not use edge features (edge_attr is ignored).
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,  # Unused but kept for API consistency
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class RNAGAT(nn.Module):
    """
    Graph Attention Network for RNA classification.

    Uses GAT layers with multi-head attention.
    Note: GAT does not use edge features (edge_attr is ignored).
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,  # Unused but kept for API consistency
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))

        # Last layer: concat=False to output hidden_dim instead of hidden_dim * num_heads
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer
        x = self.convs[-1](x, edge_index)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class RNAGIN(nn.Module):
    """
    Graph Isomorphism Network for RNA classification.

    Uses GIN layers which are more expressive than GCN.
    Note: GIN does not use edge features (edge_attr is ignored).
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,  # Unused but kept for API consistency
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GIN layers
        self.convs = nn.ModuleList()

        # First layer
        mlp = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class RNAGCNEdge(nn.Module):
    """
    Edge-Aware GCN using NNConv for RNA classification.

    Uses NNConv layers that process 5D edge features through neural networks
    to generate edge-specific message passing weights.
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # NNConv layers with edge networks
        self.convs = nn.ModuleList()

        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features * hidden_dim),
        )
        self.convs.append(NNConv(num_node_features, hidden_dim, edge_nn, aggr='mean'))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # NNConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class RNAGATEdge(nn.Module):
    """
    Edge-Aware GAT using NNConv for RNA classification.

    Combines attention mechanism with edge feature processing.
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # NNConv layers
        self.convs = nn.ModuleList()

        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features * hidden_dim),
        )
        self.convs.append(NNConv(num_node_features, hidden_dim, edge_nn, aggr='mean'))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # NNConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class RNAGINEdge(nn.Module):
    """
    Edge-Aware GIN using NNConv for RNA classification.

    Combines GIN's expressiveness with edge feature processing.
    """

    def __init__(
        self,
        num_node_features: int = 14,
        num_classes: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_edge_features: int = 5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # NNConv layers
        self.convs = nn.ModuleList()

        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features * hidden_dim),
        )
        self.convs.append(NNConv(num_node_features, hidden_dim, edge_nn, aggr='mean'))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # NNConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


def get_model(model_name: str, **kwargs):
    """
    Factory function to get model by name.

    Args:
        model_name: One of 'gcn', 'gat', 'gin', 'gcn_edge', 'gat_edge', 'gin_edge'
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    models = {
        # Baseline models (no edge features)
        'gcn': RNAGCN,
        'gat': RNAGAT,
        'gin': RNAGIN,
        # Edge-aware models (use 5D edge features)
        'gcn_edge': RNAGCNEdge,
        'gat_edge': RNAGATEdge,
        'gin_edge': RNAGINEdge,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name.lower()](**kwargs)
