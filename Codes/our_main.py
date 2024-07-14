import os
import shutil
import random
import warnings
from typing import TYPE_CHECKING

import torch.multiprocessing as mp

from helper.parser import *
from helper.utils import *
import our_train


if __name__ == '__main__':

    args = create_parser()

    assert args.bandwidth_aware, 'This main file use bandwidth aware method!!'
    if args.use_flexible:
        assert args.flexible_hop <= (args.n_layers - 1), 'flexible preloading allows max n_conv layers'
    if args.fix_seed is False:
        warnings.warn('Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)

    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%s-induc' % (args.partition_method, args.partition_obj)
        else:
            args.graph_name = '%s-%s-trans' % (args.partition_method, args.partition_obj)
    if args.use_flexible:
        if args.inductive:
            args.flexible_graph_name = '%s-%s-flex-induc' % (args.partition_method, args.partition_obj)
        else:
            args.flexible_graph_name = '%s-%s-flex-trans' % (args.partition_method, args.partition_obj)

    if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
        warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
        
        g, n_feat, n_class = load_data(args, args.dataset)
        if args.node_rank == 0 and args.use_flexible:
            if args.inductive:
                graph_partition(g.node_subgraph(g.ndata['train_mask']), args)
            else:
                graph_partition(g, args)
        args.n_feat = n_feat
        args.n_class = n_class
        args.n_train = g.ndata['train_mask'].int().sum().item()
        del g

    print(args)

    
    

    if args.node_rank == 0 and args.remove_tmp:
        if os.path.exists('./intra-partitions'):
            shutil.rmtree('./intra-partitions')

    if args.backend == 'gloo':
        processes = []
        args.local_device_cnt = torch.cuda.device_count()
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = args.local_device_cnt
            devices = [f'{i}' for i in range(n)]
        
        mp.set_start_method('spawn', force=True)
        start_id = args.node_rank * args.local_device_cnt 
        for i in range(start_id, start_id + args.local_device_cnt):
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
            p = mp.Process(target=our_train.init_processes, args=(i, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif args.backend == 'nccl':
        processes = []
        args.local_device_cnt = torch.cuda.device_count()
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = args.local_device_cnt
            devices = [f'{i}' for i in range(n)]
        
        mp.set_start_method('spawn', force=True)
        start_id = args.node_rank * args.local_device_cnt 
        for i in range(start_id, start_id + args.local_device_cnt):
            
            p = mp.Process(target=our_train.init_processes, args=(i, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif args.backend == 'mpi':
        raise NotImplementedError
    else:
        raise ValueError
