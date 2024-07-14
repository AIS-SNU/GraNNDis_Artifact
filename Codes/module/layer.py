import math

from typing import Optional

import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

import dgl.function as fn

from ogb.graphproppred.mol_encoder import BondEncoder
from dgl.nn.functional import edge_softmax

from module.others import MLP, MessageNorm



class DeeperGCNLayer(nn.Module):
    def __init__(self,
                conv: Optional[nn.Module] = None,
                norm: Optional[nn.Module] = None,
                act: Optional[nn.Module] = None,
                block: str = 'res+',
                dropout: float = 0.,
                ckpt_grad: bool = False) -> None:
        super(DeeperGCNLayer, self).__init__()
        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad
        self.num_inner = None
    
    def reset_parameters(self):
        self.conv.reset_paramters()
        self.norm.reset_parameters()

    def forward(self, graph, feat, in_deg) -> torch.Tensor:
        
        with graph.local_scope():

            
            
            
            
                      
            
            
            
            

            res_feat = feat

            
            
            if self.block == 'res+':
                h = feat
                if self.norm is not None:
                    h = self.norm(h)
                if self.act is not None:
                    h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.conv is not None and self.ckpt_grad and h.requires_grad:
                    h = checkpoint(self.conv, graph, h, in_deg)
                else:
                    h = self.conv(graph, h, in_deg)

                return res_feat[:h.shape[0]] + h 
            else:


                if self.conv is not None and self.ckpt_grad and feat.requires_grad:
                    h = checkpoint(self.conv, graph, feat, in_deg)
                else:
                    h = self.conv(graph, feat, in_deg)
                if self.norm is not None:
                    h = self.norm(h)
                if self.act is not None:
                    h = self.act(h)        
                
                
                if self.block == 'res':
                    h = res_feat[:h.shape[0]] + h
                elif self.block == 'dense':
                    h = torch.cat([res_feat[:h.shape[0]], h], dim=-1)
                elif self.block == 'plain':
                    pass
                return F.dropout(h, p=self.dropout, training=self.training)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'
            

class GraphSAGELayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 use_pp=False):
        super(GraphSAGELayer, self).__init__()
        self.use_pp = use_pp
        if self.use_pp:
            self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        else:
            self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
            self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_pp:
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.weight.data.uniform_(-stdv, stdv)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / math.sqrt(self.linear1.weight.size(1))
            self.linear1.weight.data.uniform_(-stdv, stdv)
            self.linear2.weight.data.uniform_(-stdv, stdv)
            if self.linear1.bias is not None:
                self.linear1.bias.data.uniform_(-stdv, stdv)
                self.linear2.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_deg):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    degs = in_deg.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
            else:
                assert in_deg is None
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
        return feat
