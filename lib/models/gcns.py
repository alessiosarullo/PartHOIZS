import torch
from torch import nn

from config import cfg
from lib.dataset.hico import HicoSplit


class AbstractGCN(nn.Module):
    def __init__(self, dataset: HicoSplit, input_dim=512, gc_dims=(256, 128), train_z=True):
        super().__init__()
        self.input_dim = input_dim
        self.dataset = dataset
        self.num_objects = dataset.full_dataset.num_objects
        self.num_actions = dataset.full_dataset.num_actions
        self.num_interactions = dataset.full_dataset.num_interactions

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
        return torch.empty(self.adj.shape[0], self.input_dim).normal_()

    def _build_gcn(self):
        gc_layers = []
        num_gc_layers = len(self.gc_dims)
        for i in range(num_gc_layers):
            in_dim = self.gc_dims[i - 1] if i > 0 else self.input_dim
            out_dim = self.gc_dims[i]
            if i < num_gc_layers - 1:
                gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=cfg.gcdropout))
                                 )
            else:
                gc_layers.append(nn.Linear(in_dim, out_dim))
        return gc_layers

    def _forward(self, input_repr=None):
        raise NotImplementedError


class HicoGCN(AbstractGCN):
    def __init__(self, dataset, oa_adj, **kwargs):
        self.oa_adj = oa_adj
        super().__init__(dataset=dataset, **kwargs)

    def _build_adj_matrix(self):
        # Normalised adjacency matrix. Note: the identity matrix that is supposed to be added for the "renormalisation trick" (Kipf 2016) is
        # implicitly included by initialising the adjacency matrix to an identity instead of zeros.
        adj = torch.eye(self.num_objects + self.num_actions).float()
        adj[:self.num_objects, self.num_objects:] = self.oa_adj  # top right
        adj[self.num_objects:, :self.num_objects] = self.oa_adj.t()  # bottom left
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
        act_embs = z[self.num_objects:]
        return obj_embs, act_embs
