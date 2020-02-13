import torch
from torch import nn

from config import cfg
from lib.models.misc import MemoryEfficientSwish as Swish
import numpy as np


class AbstractGCN(nn.Module):
    def __init__(self, input_dim=512, gc_dims=(256, 128), train_z=True):
        super().__init__()
        self.input_dim = input_dim

        self.adj = nn.Parameter(self._build_adj_matrix(), requires_grad=False)
        self.z = nn.Parameter(self._get_initial_z(), requires_grad=train_z)

        self.gc_dims = gc_dims
        self.gc_layers = nn.ModuleList(self._build_gcn())

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

    def _build_gcn(self):
        gc_layers = []
        num_gc_layers = len(self.gc_dims)
        for i in range(num_gc_layers):
            in_dim = self.gc_dims[i - 1] if i > 0 else self.input_dim
            out_dim = self.gc_dims[i]
            if i < num_gc_layers - 1:
                layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      (nn.ReLU(inplace=True) if not cfg.gcswish else Swish()),  # TODO? Try LeakyReLU
                                      nn.Dropout(p=cfg.gcdropout))
                nn.init.xavier_normal_(layer[0].weight, gain=torch.nn.init.calculate_gain('relu'))
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
