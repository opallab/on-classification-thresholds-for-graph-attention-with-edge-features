from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class my_MLP_GATConv_edges(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        in_edge_channels: int,
        out_edge_channels: int,
        att_in_channels: int,
        att_out_channels: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_in_channels = att_in_channels
        self.att_out_channels = att_out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        self.att_in = Linear(att_in_channels, att_out_channels, bias=bias,
                            weight_initializer='glorot') 

        self.att_out = Linear(att_out_channels, 1, bias=bias,
                            weight_initializer='glorot') 

        self.lin = Linear(in_channels, out_channels, bias=bias,
                            weight_initializer='glorot')
        
        self.lin_edge = Linear(in_edge_channels, out_edge_channels, bias=bias,
                            weight_initializer='glorot')

        self.reset_parameters()
        
        self.gammas = []

    def reset_parameters(self):
        self.att_in.reset_parameters()
        self.att_out.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        
        C = self.out_channels

        x_: OptTensor = None
        edge_attr_: OptTensor = None
            
        x_ = self.lin(x).view(-1, C)
        edge_attr_ = self.lin_edge(edge_attr).view(-1, 1)
        

        if self.add_self_loops:
            num_nodes = x_.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, edge_attr=edge_attr_, x=x_, size=None)
        out = out.mean(dim=1)
        
        return out, self.gammas


    def message(self, edge_attr_j: Tensor, edge_attr_i: Tensor, x_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        tmp = torch.cat([edge_attr_i, edge_attr_j], dim=1)
        tmp = self.att_in(tmp)
        tmp = F.leaky_relu(tmp, self.negative_slope)
        tmp = self.att_out(tmp)
        alpha = softmax(tmp, index, ptr, size_i)
        self.gammas = alpha

        return x_j * alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

