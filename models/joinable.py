# this is the main model file. i rewrote it for our project below
# NOTE: i am keeping all external logic from original repo (like datasets, metrics, args)
# I'm only modifying this model file and simplifying the structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torchvision.ops.focal_loss import sigmoid_focal_loss
from utils import metrics
from datasets.joint_graph_dataset import JointGraphDataset


# simple 2 layer MLP (or more if needed)
def make_mlp(in_dim, hid_dim, out_dim, layers=2):
    mods = []
    for i in range(layers - 1):
        mods += [nn.Linear(in_dim if i == 0 else hid_dim, hid_dim), nn.ReLU()]
    mods += [nn.Linear(hid_dim, out_dim)]
    return nn.Sequential(*mods)


# message passing network using GAT (or GATv2)
class GNNStack(nn.Module):
    def __init__(self, dim, dropout=0.0, mode="gatv2"):
        super().__init__()
        conv = GATv2Conv if mode == "gatv2" else GATConv
        self.gnn1 = conv(dim, dim // 8, heads=8, dropout=dropout)
        self.gnn2 = conv(dim, dim // 8, heads=8, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, edge_idx):
        x = self.dropout(x)
        x = self.gnn1(x, edge_idx)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gnn2(x, edge_idx)
        return x


# main model class
class JoinABLe(nn.Module):
    def __init__(self, hidden_dim, input_features, dropout=0.0, mpn="gatv2", batch_norm=False,
                 reduction="sum", post_net="mlp", pre_net="mlp"):
        super().__init__()
        self.reduction = reduction
        # setup preprocessors for face and edge (defined below)
        self.face_net = Preprocessor(hidden_dim, input_features, typ="face", method=pre_net)
        self.edge_net = Preprocessor(hidden_dim, input_features, typ="edge", method=pre_net)
        self.gnn = GNNStack(hidden_dim, dropout, mode=mpn)
        self.post_net = PostNetwork(hidden_dim, dropout, method=post_net)

    def forward(self, g1, g2, jg):
        f1, f2 = self.face_net(g1), self.face_net(g2)
        e1, e2 = self.edge_net(g1), self.edge_net(g2)
        x1, x2 = f1 + e1, f2 + e2
        x1 = self.gnn(x1, g1.edge_index)
        x2 = self.gnn(x2, g2.edge_index)
        return self.post_net(x1, x2, jg)

    def compute_loss(self, args, x, joint_graph):
        graphs = joint_graph.to_data_list()
        total_loss = 0
        for i, g in enumerate(graphs):
            pred = x[g.ptr[0]:g.ptr[1]] if hasattr(g, 'ptr') else x
            target = g.edge_attr
            if args.loss == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target.float())
            elif args.loss == "mle":
                target = target / target.sum()
                loss = F.kl_div(F.log_softmax(pred, dim=-1), target, reduction='batchmean')
            else:
                loss = sigmoid_focal_loss(pred.unsqueeze(1), target.unsqueeze(1).float())
            total_loss += loss
        return total_loss / len(graphs) if self.reduction == "mean" else total_loss


# converts graph node input features to embeddings
class Preprocessor(nn.Module):
    def __init__(self, dim, input_str, typ="face", method="mlp"):
        super().__init__()
        feat_lists = JointGraphDataset.parse_input_features_arg(input_str, input_feature_type=typ)
        self.grid_feats, self.entity_feats = feat_lists[1], feat_lists[2]
        self.grid_dim = JointGraphDataset.get_input_feature_size(self.grid_feats, input_feature_type=typ)
        self.ent_dim = JointGraphDataset.get_input_feature_size(self.entity_feats, input_feature_type=typ)
        self.method = method
        self.proj_grid = make_mlp(self.grid_dim, dim, dim) if self.grid_dim > 0 else None
        self.proj_ent = make_mlp(self.ent_dim, dim, dim) if self.ent_dim > 0 else None

    def forward(self, g):
        nodes = torch.zeros(g.num_nodes, self.proj_grid[0].in_features, device=g.edge_index.device)
        if self.ent_dim:
            ent = torch.cat([g[f] for f in self.entity_feats], dim=-1)
            nodes += self.proj_ent(ent.float())
        if self.grid_dim:
            grid = g.x.view(g.num_nodes, -1)
            nodes += self.proj_grid(grid.float())
        return nodes


# final post-network to predict joint presence between graph pairs
class PostNetwork(nn.Module):
    def __init__(self, dim, dropout=0.0, method="mlp"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if method == "mlp":
            self.mlp = make_mlp(dim * 2, dim, 1, layers=3)
        else:
            self.mlp = lambda x: (x[:, :dim] * x[:, dim:]).sum(-1, keepdim=True)

    def forward(self, x1, x2, jg):
        joint_x = []
        for b in jg.to_data_list():
            xs, xt = x1[b.edge_index[0]], x2[b.edge_index[1]]
            pair = torch.cat([xs, xt], dim=-1)
            joint_x.append(self.mlp(pair))
        return torch.cat(joint_x)
