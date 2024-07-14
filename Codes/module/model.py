from torch import nn

from module.layer import *
from module.sync_bn import SyncBatchNorm
from helper import intra_context as ctx


class GNNBase(nn.Module):

    def __init__(self, layer_size, activation, use_pp=False, dropout=0.5, norm='layer', n_linear=0):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_pp = use_pp
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

class DeeperGCN(GNNBase):
    def __init__(self, layer_size, activation, use_pp, n_feat, n_class, \
                dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(DeeperGCN, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)

        
        self.node_encoder = nn.Linear(n_feat, layer_size[0])

        
        

        use_pp = False
        
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                
                conv = GraphSAGELayer(layer_size[i], layer_size[i+1], use_pp=use_pp) 
                if norm == 'layer':
                    norm_fn = nn.LayerNorm(layer_size[i+1], elementwise_affine=True)
                elif norm == 'batch':
                    norm_fn = SyncBatchNorm(layer_size[i+1], train_size)
                act = activation
                layer = DeeperGCNLayer(conv=conv, norm=norm_fn, act=act, block='res+',
                                        
                                        dropout=dropout, ckpt_grad=False)
                self.layers.append(layer)
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i+1]))

    def forward(self, g, feat, in_deg=None, masks=None):
        
        h = self.node_encoder(feat)

        
        

        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp): 
                    h = ctx.buffer.update(i, h)
                    if masks is not None:
                        h[masks[i]] = 0.0
                h = self.layers[i](g, h, in_deg)
            else:
                h = self.layers[0].act(self.layers[0].norm(h))
                h = self.dropout(h)
                h = self.layers[i](h)
        return h


class GraphSAGE(GNNBase):
    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGE, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_deg=None, masks=None):
        h = feat

        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                    if masks is not None:
                        h[masks[i]] = 0.0
                h = self.dropout(h)
                h = self.layers[i](g, h, in_deg)
            else:
                h = self.dropout(h)
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)
        return h
