"""
GAT classes are taken from https://github.com/Diego999/pyGAT/blob/master/layers.py.
"""

import numpy as np
import torch
from torch import nn

from config import cfg
from lib.models.misc import MemoryEfficientSwish as Swish


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        with torch.set_grad_enabled(self.training):
            return SpecialSpmmFunction.apply(indices, values, shape, b)


class GraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, adj_mat, alpha=0.2, out_elu=True):
        super().__init__()
        self.adj = adj_mat
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.out_elu = out_elu

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            dv = 'cuda' if input.is_cuda else 'cpu'

            N = input.size()[0]
            edge = self.adj.nonzero().t()

            h = torch.mm(input, self.W)
            # h: N x out
            assert not torch.isnan(h).any()

            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
            # edge: 2*D x E

            edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
            assert not torch.isnan(edge_e).any()
            # edge_e: E

            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
            # e_rowsum: N x 1

            edge_e = self.dropout(edge_e)
            # edge_e: E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            assert not torch.isnan(h_prime).any()
            # h_prime: N x out

            h_prime = h_prime.div(e_rowsum)
            # h_prime: N x out
            assert not torch.isnan(h_prime).any()

            if self.out_elu:
                # if this layer is not last layer,
                return nn.functional.elu(h_prime)
            else:
                # if this layer is last layer,
                return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionConcatLayer(nn.Module):
    def __init__(self, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            return nn.functional.dropout(torch.cat([att(x) for att in self.layers], dim=1), self.dropout, training=self.training)


class AbstractGCN(nn.Module):
    def __init__(self, input_dim=512, gc_dims=(256, 128), train_z=True, gat=None):
        super().__init__()
        if gat is None:
            gat = cfg.gat
        self.input_dim = input_dim

        self.adj = nn.Parameter(self._build_adj_matrix(), requires_grad=False)
        self.z = nn.Parameter(self._get_initial_z(), requires_grad=train_z)

        self.gc_dims = gc_dims
        self.gc_layers = nn.ModuleList(self._build_gcn(gat=gat))

    @property
    def output_dim(self):
        return self.gc_dims[-1]

    def forward(self, *args, **kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(*args, **kwargs)

    def _build_adj_matrix(self):
        raise NotImplementedError

    def _get_initial_z(self):
        z = torch.empty(self.adj.shape[0], self.input_dim)
        nn.init.normal_(z)  # TODO? Experiment more with different initialisation
        return z

    def _build_gcn(self, gat=False):
        if gat:
            print('Using GAT.')
        else:
            print('Using standard GCN.')
        gc_layers = []
        num_gc_layers = len(self.gc_dims)
        for i in range(num_gc_layers):
            in_dim = self.gc_dims[i - 1] if i > 0 else self.input_dim
            out_dim = self.gc_dims[i]
            if i < num_gc_layers - 1:
                if gat:
                    num_heads = cfg.gat_heads
                    if out_dim % num_heads > 0:
                        raise ValueError(f'{out_dim}, {num_heads}')
                    gat_layers = [GraphAttentionLayer(in_dim, out_dim // num_heads, dropout=cfg.gcdropout, adj_mat=self.adj, out_elu=True)
                                  for _ in range(num_heads)]
                    layer = GraphAttentionConcatLayer(layers=gat_layers, dropout=cfg.gcdropout)
                else:
                    layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                          (nn.ReLU(inplace=True) if not cfg.gcswish else Swish()),  # TODO? Try LeakyReLU
                                          nn.Dropout(p=cfg.gcdropout))
                    nn.init.xavier_normal_(layer[0].weight, gain=torch.nn.init.calculate_gain('relu'))
            else:
                if gat:
                    layer = GraphAttentionLayer(in_dim, out_dim, dropout=cfg.gcdropout, adj_mat=self.adj, out_elu=False)
                else:
                    layer = nn.Linear(in_dim, out_dim)
                    nn.init.xavier_normal_(layer.weight, gain=torch.nn.init.calculate_gain('linear'))
            gc_layers.append(layer)
        return gc_layers

    def _forward(self, input_repr=None):
        raise NotImplementedError


class BipartiteGCN(AbstractGCN):
    def __init__(self, adj_block, **kwargs):
        self.adj_block = adj_block
        self.num_nodes1, self.num_nodes2 = adj_block.shape
        super().__init__(**kwargs)

    def _build_adj_matrix(self):
        # Normalised adjacency matrix. Note: the identity matrix that is supposed to be added for the "renormalisation trick" (Kipf 2016) is
        # implicitly included by initialising the adjacency matrix to an identity instead of zeros.
        adj = torch.eye(self.num_nodes1 + self.num_nodes2).float()
        adj[:self.num_nodes1, self.num_nodes1:] = self.adj_block  # top right
        adj[self.num_nodes1:, :self.num_nodes1] = self.adj_block.t()  # bottom left
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())
        return adj

    def _forward(self, input_repr=None):
        if input_repr is not None:
            z = input_repr
        else:
            z = self.z
        for gcl in self.gc_layers:
            z = gcl(self.adj @ z)
        nodes1_embs = z[:self.num_nodes1]
        nodes2_embs = z[self.num_nodes1:]
        return nodes1_embs, nodes2_embs


class HoiGCN(AbstractGCN):
    def __init__(self, interactions, **kwargs):
        self.num_objects = np.unique(interactions[:, 1]).size
        self.num_actions = np.unique(interactions[:, 0]).size
        self.num_interactions = interactions.shape[0]
        self.interactions = interactions
        super().__init__(**kwargs)

    def _build_adj_matrix(self):
        n = self.num_objects + self.num_actions + self.num_interactions
        adj = torch.zeros((n, n)).float()
        adj[self.interactions[:, 1],
            self.num_objects + self.num_actions + torch.arange(self.num_interactions)] = 1
        adj[self.num_objects + self.interactions[:, 0],
            self.num_objects + self.num_actions + torch.arange(self.num_interactions)] = 1
        adj = adj + adj.t() + torch.eye(n).float()  # eye is added for the "renormalisation trick" (Kipf 2016)
        assert ((adj == 0) | (adj == 1)).all()
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())
        return adj

    def _forward(self, input_repr=None):
        if input_repr is not None:
            z = input_repr
        else:
            z = self.z
        for gcl in self.gc_layers:
            z = gcl(self.adj @ z)
        obj_embs = z[:self.num_objects]
        act_embs = z[self.num_objects:self.num_objects + self.num_actions + 1]
        hoi_embs = z[-self.num_interactions:]
        return obj_embs, act_embs, hoi_embs
