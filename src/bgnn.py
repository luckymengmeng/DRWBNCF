import torch
from torch import nn
from torch.nn import functional as F

def bgnn_pool(xw, adj):
    sum = adj@xw
    sum_squared = sum.square()
    # step2 squared_sum
    squared = xw.square()
    squared_sum = torch.square(adj)@squared
    # step3
    new_embedding = sum_squared - squared_sum
    return new_embedding

def bgnn_a_norm(edge_index, add_self_loop=True):
    adj_t = edge_index.to_dense()
    if add_self_loop:
        adj_all = adj_t+torch.eye(adj_t.shape[0], device=adj_t.device)
    # num_nei = adj_all.sum(dim=-1)
    norm = (adj_all.sum(dim=-1).square()-adj_all.square().sum(dim=-1))
    # norm = num_nei*(num_nei-1)
    norm = norm.pow(-1)
    norm.masked_fill_(torch.isinf(norm), 0.)
    norm = torch.diag(norm)
    norm = norm.to_sparse()
    adj_all = adj_all.to_sparse()
    return adj_all, norm


class BGNNA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        super(BGNNA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self._cache = None
        self.add_self_loops = add_self_loops
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index):
        xw = x@self.weight
        if self.cached:
            if not hasattr(self, "cached_adj") or not hasattr(self, "cached_norm"):
                adj, norm = bgnn_a_norm(edge_index, add_self_loop=self.add_self_loops)
                self.register_buffer("cached_adj", adj)
                self.register_buffer("cached_norm", norm)
            else:
                adj, norm = self.cached_adj, self.cached_norm
        else:
            adj, norm = bgnn_a_norm(edge_index, add_self_loop=self.add_self_loops)
        out = bgnn_pool(xw, adj)
        out = norm@out
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=adj_t.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = True, add_self_loops: bool = False,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        if self.cached:
            if not hasattr(self, "cached_adj"):
                edge_index, edge_weight = gcn_norm(
                    edge_index, self.add_self_loops)
                self.register_buffer("cached_adj", edge_index)
            edge_index = self.cached_adj
        else:
            edge_index, _ = gcn_norm(edge_index, self.add_self_loops)
        x = torch.matmul(x, self.weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class BGCNA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, lamda=0.8, share=True, **kwargs):
        super(BGCNA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.bgnn = BGNNA(in_channels=in_channels, out_channels=out_channels, cached=cached,
                          add_self_loops=add_self_loops, bias=bias)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=out_channels, cached=cached,
                           add_self_loops=add_self_loops, bias=bias)

        self.register_buffer("alpha", torch.tensor(lamda))
        self.register_buffer("beta", torch.tensor(1-lamda))
        self.reset_parameters()
        if share:
            self.bgnn.weight = self.gcn.weight

    def reset_parameters(self):
        self.bgnn.reset_parameters()
        self.gcn.reset_parameters()

    def forward(self, x, edge_index):
        x1 = self.gcn(x, edge_index)
        x2 = self.bgnn(x, edge_index)
        x = self.beta*F.relu(x1)+self.alpha*F.relu(x2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
