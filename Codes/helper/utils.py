import os
import os.path as osp
import warnings
import argparse
warnings.filterwarnings("ignore")

import scipy
import torch
import dgl
from dgl.data import DGLDataset
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
import torch.distributed as dist
import time
from contextlib import contextmanager
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

class IGB260M(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        if self.size == 'experimental':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        
        if self.size == 'large' or self.size == 'full':
            path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_feat.npy')
            emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:

        if self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                
            else:
                path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                

        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        
        
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260MDGLDataset')

    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, \
            classes=self.args.num_classes, synthetic=self.args.synthetic)

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        if self.args.dataset_size == 'full':
            
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val   = int(n_labeled_idx * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)

def load_igb_dataset(args, name):
    igb_args = args
    igb_args.path = args.dataset_path + 'igb/'

    def _select_size(name):
        if 'tiny' in name:
            return 'tiny'
        elif 'small' in name:
            return 'small'
        elif 'medium' in name:
            return 'medium'
        elif 'large' in name:
            return 'large'
        elif 'full' in name:
            return 'full'

    igb_args.dataset_size = _select_size(name)

    
    
    igb_args.num_classes = 19 
    igb_args.in_memory = 0 
    igb_args.synthetic = 0 
    dataset = IGB260MDGLDataset(args=igb_args)
    g = dataset[0]
    
    g = dgl.add_reverse_edges(g)
    return g

def load_ogb_dataset(args, name):
    dataset = DglNodePropPredDataset(name=name, root=args.dataset_path)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    
    if name == 'ogbn-arxiv' or name == 'ogbn-papers100M':
        g = dgl.add_reverse_edges(g)
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_yelp(args):
    prefix = args.dataset_path + 'yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(
        torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g

def graph_scaler(orig_g, orig_n, orig_e, scale=1.0):
    scale = int(scale) 
    scaled_g = dgl.rand_graph(int(orig_n*scale), int(orig_e*scale), device='cpu')
    scaled_g.ndata['train_mask'] = orig_g.ndata['train_mask'].repeat_interleave(scale)
    scaled_g.ndata['val_mask'] = orig_g.ndata['val_mask'].repeat_interleave(scale)
    scaled_g.ndata['test_mask'] = orig_g.ndata['test_mask'].repeat_interleave(scale)
    scaled_g.ndata['feat'] = orig_g.ndata['feat'].repeat_interleave(scale, dim=0)
    scaled_g.ndata['label'] = orig_g.ndata['label'].repeat_interleave(scale)
    scaled_g.edata['__orig__'] = orig_g.edata['__orig__'].repeat_interleave(scale)
    return scaled_g

def load_data(args, dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir=args.dataset_path)
        reddit_num_nodes = 232_965
        reddit_num_edges = 114_615_892
        g = data[0]
        if args.dataset_scale != 1:
            g = graph_scaler(g, reddit_num_nodes, reddit_num_edges, scale=args.dataset_scale)
    elif 'igb' in dataset:
        g = load_igb_dataset(args, dataset)
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset(args, 'ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset(args, 'ogbn-papers100M')
    elif dataset == 'ogbn-arxiv':
        g = load_ogb_dataset(args, 'ogbn-arxiv')
    elif dataset == 'yelp':
        
        g = load_yelp(args)
        
        
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


def load_partition(args, rank):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb


def graph_partition(g, args):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_degree'] = g.in_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir, part_method=args.partition_method,
                            balance_edges=False, objtype=args.partition_obj)


def load_intra_partition(args, local_rank, iter):
    graph_dir = 'intra-partitions/%s_%dlayers/total_%d_nodes_local_%d_gpus/iter%d/rank%d/' % (args.dataset, (args.n_layers - args.n_linear), args.total_nodes,
                                                                args.local_device_cnt, iter, args.node_rank)
    part_config = graph_dir + args.graph_name + '.json'

    
    

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, local_rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    
    
    
    
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['seed_node'] = node_feat[node_type + '/seed_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/seed_node')
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb

def server_node_graph_partition(g, iter, args, map_dict=None, my_seed=None):
    """Does not manipulate anything; just save the partitioned graphs in the corresponding directory.
    g: subgraph for current server."""
    sid_from_psid = None
    graph_dir = 'intra-partitions/%s_%dlayers/total_%d_nodes_local_%d_gpus/iter%d/rank%d/' % (args.dataset, (args.n_layers - args.n_linear), args.total_nodes,
                                                                args.local_device_cnt, iter, args.node_rank)
    part_config = graph_dir + args.graph_name + '.json'

    
    

    print('!!!PARTITIONING GRAPH PER SERVER.... BE PATIENT!!!')

    
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_degree'] = g.in_degrees()
            sid_from_psid, _ = partition_graph(g, args.graph_name, args.local_device_cnt, graph_dir, part_method=args.partition_method,
                            balance_edges=False, objtype=args.partition_obj, return_mapping=True)

    print(g.nodes(), ' ', g.nodes().shape)
    print(g.ndata[dgl.NID], 'MID ID MAPPER')
    return sid_from_psid


def get_layer_size(args, n_feat, n_hidden, n_class, n_layers):
    
    if args.model in ['deepgcn']:
        layer_size = [n_hidden] 
    else:
        layer_size = [n_feat]
    
    
    layer_size.extend([n_hidden] * (n_layers - 1))
    
    layer_size.append(n_class)
    return layer_size


def get_intra_boundary(args, node_dict, gpb):
    """Get boundary of the intra graph of this server.

    Parameters
    ----------
    args : Args
    node_dict : NodeDict
    gpb : GraphPartitionBook

    Returns
    -------
    boundary : list[Tensor | None]
        boundary[i] is the gpu_ids of inner vertices on my GPU partition that also exist on the i-th GPU.
        In other words, boundary vertex has at least one inner vertex of the i-th GPU partition as a neighbor.
        Also, boundary[local_rank] is None.
    """    
    local_rank, local_size = args.rank % args.local_device_cnt, args.local_device_cnt
    start_id = args.local_device_cnt * args.node_rank

    boundary = [None] * local_size
    buffer_size = list()

    """
    Send how much i need from other rank....
    Therefore, finally each worker has an array,
    which contains information about how much i need to send to other ranks.
    (So, array[my_rank_id] is None)
    """

    for i in range(1, local_size):
        left = (local_rank - i + local_size) % local_size
        right = (local_rank + i) % local_size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)

        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device='cuda')
        if local_rank < i:
            dist.send(num_right, dst=start_id + right)
        dist.recv(num_left, src=start_id + left)
        if local_rank >= i:
            dist.send(num_right, dst=start_id + right)
        buffer_size.append(num_left)

    for i in range(1, local_size):
        left = (local_rank - i + local_size) % local_size
        right = (local_rank + i) % local_size
        belong_right = (node_dict['part_id'] == right)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start  
        
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(buffer_size[i-1], dtype=torch.long)
        else:
            u = torch.zeros(buffer_size[i-1], dtype=torch.long, device='cuda')

        if local_rank < i:
            dist.send(v, dst=start_id + right)
        
        dist.recv(u, src=start_id + left)
        
        if local_rank >= i:
            dist.send(v, dst=start_id + right)
        
        u, _ = torch.sort(u)
        
        if dist.get_backend() == 'gloo':
            boundary[left] = u.to('cuda')
        else:
            boundary[left] = u
    return boundary


def get_boundary(node_dict, gpb, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    boundary = [None] * size
    buffer_size = list()
    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        if rank < i:
            dist.send(num_right, dst=right)
        dist.recv(num_left, src=left)
        if rank >= i:
            dist.send(num_right, dst=right)
        buffer_size.append(num_left)
    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(buffer_size[i-1], dtype=torch.long)
        else:
            u = torch.zeros(buffer_size[i-1], dtype=torch.long, device=device)
        if rank < i:
            dist.send(v, dst=right)
        dist.recv(u, src=left)
        if rank >= i:
            dist.send(v, dst=right)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.to(device)
        else:
            boundary[left] = u
    return boundary



def data_transfer(data, recv_shape, backend, dtype=torch.float, tag=0):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        if backend == 'gloo':
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype)
        else:
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype, device='cuda')

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        if backend == 'gloo':
            req = dist.isend(data[right].cpu(), dst=right, tag=tag)
        else:
            req = dist.isend(data[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda()
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))
