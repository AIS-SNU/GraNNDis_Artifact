import dgl
from dgl.data import RedditDataset
from dgl import transforms
from dgl.dataloading.base import set_node_lazy_features, set_edge_lazy_features, Sampler
from dgl.sampling.utils import EidExcluder
from helper.utils import *
import numpy as np

def _load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset/')
        g = data[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif dataset == 'yelp':
        g = load_yelp()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class

"""
Use default sampler
"""

class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier

"""
Shadow-GNN subgraph samplers
"""
class ShaDowKHopSampler(Sampler):
    def __init__(self, fanouts, use_mask=False, replace=False, prob=None, prefetch_node_feats=None,
                 prefetch_edge_feats=None, output_device=None):
        super().__init__()
        self.fanouts = fanouts
        self.use_mask = use_mask
        self.replace = replace
        self.prob = prob
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device


    def sample(self, g, seed_nodes, relabel_nodes=True, exclude_eids=None):     
        """Sampling function.

        Parameters
        ----------
        g : DGLGraph
            The graph to sampler from.
        seed_nodes : Tensor or dict[str, Tensor]
            The nodes sampled in the current minibatch.
        exclude_eids : Tensor or dict[etype, Tensor], optional
            The edges to exclude from neighborhood expansion.

        Returns
        -------
        input_nodes, output_nodes, subg, masks
            A triplet containing (1) the node IDs inducing the subgraph, (2) the node
            IDs that are sampled in this minibatch, and (3) the subgraph itself.
            (4) mask
        """
        output_nodes = seed_nodes
        masks = []
                
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, output_device=self.output_device,
                replace=self.replace, prob=self.prob, exclude_edges=exclude_eids)
            block = transforms.to_block(frontier, seed_nodes)
            
            """
            Code Snippet: https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
            """
            combined = torch.cat((block.srcdata[dgl.NID], block.dstdata[dgl.NID], block.dstdata[dgl.NID]))
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            masks.insert(0, diff)
            
            seed_nodes = block.srcdata[dgl.NID]

        subg = dgl.node_subgraph(g, seed_nodes, relabel_nodes=relabel_nodes, output_device=self.output_device)
        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, subg, masks


class SAINTSampler(Sampler):
    """Random node/edge/walk sampler from
    `GraphSAINT: Graph Sampling Based Inductive Learning Method
    <https://arxiv.org/abs/1907.04931>`__

    For each call, the sampler samples a node subset and then returns a node induced subgraph.
    There are three options for sampling node subsets:

    - For :attr:`'node'` sampler, the probability to sample a node is in proportion
      to its out-degree.
    - The :attr:`'edge'` sampler first samples an edge subset and then use the
      end nodes of the edges.
    - The :attr:`'walk'` sampler uses the nodes visited by random walks. It uniformly selects
      a number of root nodes and then performs a fixed-length random walk from each root node.

    Parameters
    ----------
    mode : str
        The sampler to use, which can be :attr:`'node'`, :attr:`'edge'`, or :attr:`'walk'`.
    budget : int or tuple[int]
        Sampler configuration.

        - For :attr:`'node'` sampler, budget specifies the number of nodes
          in each sampled subgraph.
        - For :attr:`'edge'` sampler, budget specifies the number of edges
          to sample for inducing a subgraph.
        - For :attr:`'walk'` sampler, budget is a tuple. budget[0] specifies
          the number of root nodes to generate random walks. budget[1] specifies
          the length of a random walk.

    cache : bool, optional
        If False, it will not cache the probability arrays for sampling. Setting
        it to False is required if you want to use the sampler across different graphs.
    prefetch_ndata : list[str], optional
        The node data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    prefetch_edata : list[str], optional
        The edge data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    output_device : device, optional
        The device of the output subgraphs.

    Examples
    --------

    >>> import torch
    >>> from dgl.dataloading import SAINTSampler, DataLoader
    >>> num_iters = 1000
    >>> sampler = SAINTSampler(mode='node', budget=6000)
    >>> 
    >>> dataloader = DataLoader(g, torch.arange(num_iters), sampler, num_workers=4)
    >>> for subg in dataloader:
    ...     train_on(subg)
    """
    def __init__(self, mode, cache=True, prefetch_ndata=None,
                 prefetch_edata=None, output_device='cpu'):
        super().__init__()
        
        if mode == 'node':
            self.sampler = self.node_sampler
        else:
            raise DGLError(f"Expect mode to be 'node', 'edge' or 'walk', got {mode}.")

        self.cache = cache
        self.prob = None
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device

    def node_sampler(self, g, budget):
        """Node ID sampler for random node sampler"""
        
        
        
        if self.cache and self.prob is not None:
            prob = self.prob
        else:
            prob = g.out_degrees().float().clamp(min=1)
            if self.cache:
                self.prob = prob
        return torch.multinomial(prob, num_samples=budget,
                                 replacement=True).unique().type(g.idtype)

    def sample(self, g, indices, budget):
        """Sampling function

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor
            Placeholder not used.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        node_ids = self.sampler(g, budget)
        sg = dgl.node_subgraph(g, node_ids, relabel_nodes=True, output_device=self.output_device)
        
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return None, node_ids, sg, None


"""
Merge subs into single graph by dgl.merge
"""

def merge_subgraphs(subgs):
    return dgl.merge(subgs)

if __name__ == '__main__':
    print('Mini-batch Partitioning Unit Test')
    
    print('[Step 0] Load Reddit Dataset')
    g, n_feat, n_class = _load_data('reddit')
    

    print('[Step 1] Get Dedicated IDs')
    print(' '*4, '> Train Mask')
    print(' '*8, g.ndata['train_mask'].shape)
    
    print(' '*4, '> Train NIDs')
    train_nids = g.ndata['train_mask'].nonzero().squeeze()
    test_nids = g.ndata['test_mask'].nonzero().squeeze()
    val_nids = g.ndata['val_mask'].nonzero().squeeze()
    print(len(train_nids))
    print(len(test_nids))
    print(len(val_nids))
    print(len(train_nids)+len(test_nids)+len(val_nids))
    print(len(train_nids)/(len(train_nids)+len(test_nids)+len(val_nids)))

    print(' '*4, '> Split NIDs')
    seed_node_array = np.array_split(np.array(train_nids), 4)
    print(len(seed_node_array[0]))

    print('[Step 2] Make Sampler and Make Subgraph')
    sampler = dgl.dataloading.NeighborSampler([10, 10])
    

    

    print(' '*4, '> Sample Subgraph from 0th NIDs')
    train_sum = 0
    input_nodes, output_nodes, subgs = sampler.sample(g, seed_node_array[3])
    train_sum = train_sum + len(output_nodes)
    input_nodes, output_nodes, subgs = sampler.sample(g, seed_node_array[1])
    train_sum = train_sum + len(output_nodes)
    input_nodes, output_nodes, subgs = sampler.sample(g, seed_node_array[2])
    train_sum = train_sum + len(output_nodes)
    input_nodes, output_nodes, subgs = sampler.sample(g, seed_node_array[0])
    train_sum = train_sum + len(output_nodes)
    print(len(input_nodes))
    print(train_sum)
    print(len(output_nodes))
    
    

    print('[Step 3] Merge Subgraphs')
    full_g_0 = dgl.merge(subgs)
    print(full_g_0.nodes())
    print(len(full_g_0.ndata['train_mask'].nonzero().squeeze()) /
        (len(full_g_0.ndata['train_mask'].nonzero().squeeze()) +\
            len(full_g_0.ndata['test_mask'].nonzero().squeeze()) +\
                len(full_g_0.ndata['val_mask'].nonzero().squeeze())))
    print(len(full_g_0.ndata['train_mask'].nonzero().squeeze()))
    
    print(len(full_g_0.ndata['train_mask'].nonzero().squeeze()) +\
            len(full_g_0.ndata['test_mask'].nonzero().squeeze()) +\
                len(full_g_0.ndata['val_mask'].nonzero().squeeze()))

    print('[Step 4] Print Graph per Node Info')
    print(full_g_0)


