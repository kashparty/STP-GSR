import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
import networkx as nx
from src.dual_graph_utils import create_dual_graph, create_dual_graph_feature_matrix
from src.matrix_vectorizer import MatrixVectorizer
import numpy as np
from itertools import product


def weight_variable_glorot(output_dim):

    input_dim = 268
    init_range = math.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand((input_dim, output_dim)) * init_range * 2 - init_range

    return initial


def normalize_adj_torch(mx):
    # mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


class TargetEdgeInitializer(nn.Module):
    """TransformerConv based taregt edge initialization model"""
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.conv1 = TransformerConv(n_source_nodes, n_target_nodes // num_heads,
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        # ks = [0.9, 0.7, 0.6, 0.5]
        # self.gsrnet = GSRNet(ks, 160, 268, 320)
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        # Super-resolve source graph using matrix multiplication
        xt = x.T @ x    # xt will be treated as the adjacency matrix of the target graph

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        return xt


class DualGraphLearner(nn.Module):
    """Update node features of the dual graph"""
    def __init__(self, in_dim, out_dim=1, num_heads=1, 
                 dropout=0.2, beta=False):
        super().__init__()

        # Here, we override num_heads to be 1 since we output scalar primal edge weights
        # In future work, we can experiment with multiple heads
        self.conv1 = TransformerConv(in_dim, out_dim, 
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)

    def forward(self, x, edge_index):
        # Update embeddings for the dual nodes/ primal edges
        edge_index = edge_index.to(x.device)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        return xt


class GSRLayer(nn.Module):

    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()
        self.weights = weight_variable_glorot(hr_dim).type(torch.float32)
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad=True)

    def forward(self, A, X):
        lr = A
        lr_dim = lr.shape[0]
        f = X
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO="U")
        # U_lr = torch.abs(U_lr)
        eye_mat = torch.eye(lr_dim).type(torch.float32)
        s_d = torch.cat((eye_mat, eye_mat), 0)

        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a, torch.t(U_lr))
        f_d = torch.matmul(b, f)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)
        adj = normalize_adj_torch(self.f_d)
        X = torch.mm(adj, adj.t())
        X = (X + X.t()) / 2
        idx = torch.eye(268, dtype=torch.bool)
        X[idx] = 1
        return adj, torch.abs(X)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    # 160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        # output = self.act(output)
        return output


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0)

    def forward(self, A, X):

        X = self.drop(X)
        # X = torch.matmul(A, X)
        X = self.proj(X)
        return X


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks

        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2 * dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim))
            self.up_gcns.append(GCN(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):

            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)

        return X, start_gcn_outs


class GSRNet(nn.Module):
    def __init__(self, ks, lr_dim, hr_dim, hidden_dim):
        super(GSRNet, self).__init__()

        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.hidden_dim = hidden_dim
        self.layer = GSRLayer(self.hidden_dim)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
        self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu)

    def forward(self, lr):
        I = torch.eye(self.lr_dim).type(torch.float32)
        A = normalize_adj_torch(lr).type(torch.float32)

        self.net_outs, self.start_gcn_outs = self.net(A, I)
        self.outputs, self.Z = self.layer(A, self.net_outs)

        self.hidden1 = self.gc1(self.Z, self.outputs)
        self.hidden2 = self.gc2(self.hidden1, self.outputs)

        z = self.hidden2
        z = (z + z.t()) / 2
        idx = torch.eye(self.hr_dim, dtype=torch.bool)
        z[idx] = 1

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs


class STPGSR(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        n_source_nodes = config.dataset.n_source_nodes
        n_target_nodes = config.dataset.n_target_nodes

        self.target_edge_initializer = TargetEdgeInitializer(
                            n_source_nodes,
                            n_target_nodes,
                            num_heads=config.model.target_edge_initializer.num_heads,
                            edge_dim=config.model.target_edge_initializer.edge_dim,
                            dropout=config.model.target_edge_initializer.dropout,
                            beta=config.model.target_edge_initializer.beta
        )
        self.dual_learner = DualGraphLearner(
                            in_dim=config.model.dual_learner.in_dim,
                            out_dim=config.model.dual_learner.out_dim,
                            num_heads=config.model.dual_learner.num_heads,
                            dropout=config.model.dual_learner.dropout,
                            beta=config.model.dual_learner.beta
        )

        self.discriminator = Discriminator(n_target_nodes)

        ut_mask = torch.triu(torch.ones((n_target_nodes, n_target_nodes)), diagonal=1).bool()
        self.register_buffer("ut_mask", ut_mask)

    def forward(self, source_pyg, target_mat):
        # Initialize target edges
        target_edge_init_sq = self.target_edge_initializer(source_pyg)

        # Create the dual graph, ensure no zeros to prevent edges being removed
        target_edge_init_sq = torch.where(
            target_edge_init_sq == 0,
            1e-10 * torch.ones_like(target_edge_init_sq),
            target_edge_init_sq
        )
        dual_edge_index, _ = create_dual_graph(target_edge_init_sq)

        # Fetch and reshape upper triangular part to get dual graph's node feature matrix
        target_edge_init = torch.masked_select(target_edge_init_sq, self.ut_mask).view(-1, 1)
        
        target_edge_init_sq_cpu = target_edge_init_sq.detach().cpu().numpy()

        G = nx.Graph()
        weighted_edges = []

        for u in range(len(target_edge_init_sq_cpu[0])):
            for v in range(len(target_edge_init_sq_cpu[1])):
                # skip self connections
                if u == v:
                    continue
                # add weighted edge manually as nx discards edges of weight 0
                w = target_edge_init_sq_cpu[u, v]
                weighted_edges.append((u, v, w))

        G.add_weighted_edges_from(weighted_edges)

        betweenness_nodewise = torch.tensor(np.array(list(nx.edge_betweenness_centrality(G).values()))).unsqueeze(-1).to(target_edge_init_sq.device)

        degree_centrality_node = list(nx.degree_centrality(G).values())
        node_centrality_pairs = product(enumerate(degree_centrality_node), enumerate(degree_centrality_node))
        edge_centrality = [(i_centrality + j_centrality) / 2 for ((i_idx, i_centrality), (j_idx, j_centrality)) in node_centrality_pairs if i_idx > j_idx]
        edge_centrality = torch.tensor(np.array(edge_centrality)).unsqueeze(-1).to(target_edge_init_sq.device)

        edge_features = torch.cat([target_edge_init, betweenness_nodewise, edge_centrality], dim=1).to(torch.float).to(target_edge_init_sq.device)

        # Update target edges in the dual space 
        dual_pred_x = self.dual_learner(edge_features, dual_edge_index)

        if target_mat is not None:
            dual_target_x = create_dual_graph_feature_matrix(target_mat)
        else:
            dual_target_x = None
        # Ensure correct shape for Discriminator input
        n_target_nodes = 268  # This should match your dataset's target nodes

        pred_graph_flat = torch.zeros((n_target_nodes, n_target_nodes), device=dual_pred_x.device)
        real_graph_flat = torch.zeros((n_target_nodes, n_target_nodes), device=dual_target_x.device) if dual_target_x is not None else None

        triu_indices = torch.triu_indices(n_target_nodes, n_target_nodes, offset=1)

        pred_graph_flat[triu_indices[0], triu_indices[1]] = dual_pred_x.squeeze()

        if dual_target_x is not None:
            real_graph_flat[triu_indices[0], triu_indices[1]] = dual_target_x.squeeze()

        pred_graph_flat = pred_graph_flat + pred_graph_flat.T
        real_graph_flat = real_graph_flat + real_graph_flat.T

        torch.diagonal(pred_graph_flat).fill_(1)
        torch.diagonal(real_graph_flat).fill_(1)

        # Get Discriminator Scores
        real_labels = self.discriminator(real_graph_flat)
        fake_labels = self.discriminator(pred_graph_flat)

        return dual_pred_x, dual_target_x, fake_labels, real_labels


class Dense(nn.Module):
    """Fully connected (Dense) layer with trainable weights"""
    def __init__(self, n1, n2, mean=0, std=0.01):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(n1, n2))
        nn.init.normal_(self.weights, mean=mean, std=std)

    def forward(self, x):
        return torch.mm(x, self.weights)


class Discriminator(nn.Module):
    """Discriminator network for adversarial training"""
    def __init__(self, hr_dim, mean_dense=0, std_dense=0.01):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(hr_dim, hr_dim, mean_dense, std_dense)
        self.relu_1 = nn.ReLU(inplace=True)
        self.dense_2 = Dense(hr_dim, hr_dim, mean_dense, std_dense)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dense_3 = Dense(hr_dim, 1, mean_dense, std_dense)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = self.sigmoid(self.dense_3(dc_den2))
        return torch.abs(output)
