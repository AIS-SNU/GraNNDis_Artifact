import time
import copy
import random
import os
import traceback
import datetime
import gc
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Literal, NewType
import concurrent.futures

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score

from module.model import *
from helper.utils import *
from helper.sampler import *
from helper.MongoManager import *
from helper.mapper import Mapper

if TYPE_CHECKING:
    from dgl import DGLGraph
    from dgl.distributed.graph_partition_book import RangePartitionBook
    from helper.parser import Args

NodeDict = NewType("NodeDict", dict[str, torch.Tensor])


def _calc_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro').item()


@torch.no_grad()
def _evaluate_induc(name: str, model: torch.nn.Module, g: 'DGLGraph', mode: Literal['val', 'test'], result_file_name: str or None = None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = _calc_acc(logits, labels)
    del logits
    del labels
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)

    return model, acc


@torch.no_grad()
def _evaluate_trans(name: str, model: torch.nn.Module, g: 'DGLGraph', result_file_name: str or None = None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    
    val_acc = _calc_acc(val_logits, val_labels)
    del val_labels
    del val_logits
    test_acc = _calc_acc(test_logits, test_labels)
    del test_labels
    del test_logits
    
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)

    return model, val_acc


def _move_to_cuda(graph: 'DGLGraph', node_dict: NodeDict) -> tuple['DGLGraph', NodeDict]:
    """Move graph, node_dict to cuda memory.
    Also convert idtype of graph into int32.

    Parameters
    ----------
    graph : DGLGraph
        GPU-graph
    node_dict : NodeDict
        Dictionary of node features

    Returns
    -------
    graph : DGLGraph
    node_dict : NodeDict
        Return back graph, node_dict tuple.
    """
    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))

    return graph, node_dict


def _get_intra_graph(args: 'Args', graph: 'DGLGraph', node_dict: NodeDict) -> tuple['DGLGraph', torch.Tensor]:
    """Get intra graph from partitioned subgraph.
    Intra graph is a graph that consists of edges which map into inner nodes.
    Therefore, the output graph is a heterograph ('_U', '_E', '_V'),
        '_U': 0, 1, ..., len(graph.nodes())-1
        '_V': 0, 1, ..., len(inner_nodes)-1
    Also, '_V' nodes will not be reordered.

    Parameters
    ----------
    args : Args
    graph : DGLGraph
        GPU-graph
    node_dict : NodeDict
        Dictionary of node features

    Returns
    -------
    g : DGLGraph
        Reordered intra graph
    mapper : Tensor
        mapper[new_ids] = old_ids
        Note that for id 0, 1, ..., len(inner_nodes)-1, the old_ids and new_ids are same (i.e. identity map)
        This method reorders id len(inner_nodes), ..., len(graph.nodes())-1.
    """
    local_rank, local_size = dist.get_rank() % args.local_device_cnt, args.local_device_cnt

    
    one_hops_part = []
    for i in range(local_size):
        nodes = []
        part_nodes = []
        for j, part_id in enumerate(node_dict['part_id']):
            if part_id == i:  
                part_nodes.append(j)
                nodes.append(node_dict[dgl.NID][j])
        nodes = torch.tensor(nodes)
        part_nodes = torch.tensor(part_nodes, dtype=torch.int32)
        _, mapper = torch.sort(nodes)
        part_nodes = part_nodes[mapper]  
        one_hops_part.append(part_nodes)
    
    one_hops_part[0], one_hops_part[1:local_rank+1] = one_hops_part[local_rank], one_hops_part[:local_rank]

    
    tot = 0
    u_list, v_list = [], []
    original_node_ids = []
    for u in one_hops_part:
        original_node_ids.append(u)
        if u.shape[0] == 0:
            continue
        u = u.to('cuda')
        u_ = torch.repeat_interleave(graph.out_degrees(u)) + tot  
        tot += u.shape[0]
        _, v = graph.out_edges(u)
        u_list.append(u_)
        v_list.append(v)

    u = torch.cat(u_list)
    v = torch.cat(v_list)
    sel = node_dict['inner_node'][v.long()]  
    u = u[sel]
    v = v[sel]
    original_node_ids = torch.cat(original_node_ids).to(torch.int32).to('cuda')
    
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    
    return g, original_node_ids


def _get_intra_recv_shape(args: 'Args', node_dict: NodeDict) -> list[int]:
    """Get recv_shape of the intra graph of this server.

    Parameters
    ----------
    args : Args
    node_dict : NodeDict

    Returns
    -------
    recv_shape : list[int]
        recv_shape[i] is the number of vertices on my GPU that exist as inner vertices of the i-th GPU partition.
        In other words, recv_shape is the opposite version of boundary.
    """
    local_rank, local_size = args.rank % args.local_device_cnt, args.local_device_cnt
    recv_shape = []
    for i in range(local_size):
        if i == local_rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def _create_model(layer_size: list[int], args: 'Args') -> GraphSAGE or DeeperGCN:
    """Create GraphSAGE or DeeperGCN model.

    Parameters
    ----------
    layer_size : list[int]
        Layer size of each layer.
    args : Args

    Returns
    -------
    model : GraphSAGE | DeeperGCN
        The type depends on args.model parameter.
        This model uses parameters like use_pp, n_feat, n_class, norm, dropout, n_linear, n_train from args.

    Raises
    ------
    NotImplementedError
        Raises if args.model is neither 'graphsage' nor 'deepgcn'.
    """
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'deepgcn':
        return DeeperGCN(layer_size, nn.ReLU(inplace=True), args.use_pp, args.n_feat, args.n_class, norm=args.norm, dropout=args.dropout,
                        n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError


def _fix_seeds(args: 'Args') -> None:
    """Set random seed for torch, torch.cuda, np.random, random, dgl, dgl.random."""
    if dist.get_rank() == 0:
        print('!!! FIXING SEEDS !!!')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    
    random.seed(args.seed)

    
    cudnn.benchmark = True

    
    dgl.seed(args.seed)
    dgl.random.seed(args.seed)
    
    
    


def _init_server_node_group(args: 'Args') -> None:
    """Initialize head processes' distributed environment of each server."""
    head_ranks = [i for i in range(0, dist.get_world_size(), args.local_device_cnt)]
    args.head_group = dist.new_group(head_ranks, backend=args.backend)


def _init_local_device_group(args: 'Args') -> None:
    """Initialize local device distributed environment."""
    start_id = args.node_rank * args.local_device_cnt
    device_group = None
    for node_id in range(args.total_nodes):
        start_id = node_id * args.local_device_cnt
        local_rank = [i for i in range(start_id, start_id + args.local_device_cnt)]
        temp_group = dist.new_group(local_rank, backend=args.backend, timeout=datetime.timedelta(seconds=60))
        if node_id == args.node_rank:
            device_group = temp_group
    args.device_group =  device_group


def _set_additional_args_parameters(args: 'Args') -> None:
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.local_rank = args.rank % args.local_device_cnt

    if args.norm == 'none':
        args.norm = None

    _init_server_node_group(args)
    _init_local_device_group(args)


def _preprocess_partition_graph(args: 'Args') -> None:
    """Samples current server-graph from full-graph, and register the mapping.
    Then partitions the server-graph into gpu-graphs and saves them at "intra-partitions" folder.
    As you expected, this method must be called only once per each server.

    Parameters
    ----------
    args : Args
    """
    g, n_feat, n_class = load_data(args, args.dataset)
    train_nids = g.ndata['train_mask'].nonzero().squeeze()

    if args.full_neighbor:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers-args.n_linear)
    elif args.sampler == 'saint':
        assert args.use_mask == False, 'saint does not use masking'
        sampler = SAINTSampler(mode='node')
    elif args.sampler == 'cluster':
        sampler = ShaDowKHopSampler([0]*args.subgraph_hop, use_mask=False)
    elif args.sampler == 'sage':
        sampler = ShaDowKHopSampler([args.fanout]*args.subgraph_hop, use_mask=args.use_mask)
    else:
        raise NotImplementedError

    
    seed_node_array = np.array_split(np.array(train_nids), args.total_nodes*args.epoch_iter)
    for iter in range(args.epoch_iter):
        start_id = args.total_nodes * iter
        my_id = start_id + args.node_rank
        my_seed = seed_node_array[my_id]
        print(my_seed, my_seed.shape)

        if args.sampler == 'saint':
            
            _, output_nodes, server_node_graph, masks = sampler.sample(g, None, my_seed.shape[0])
        else:
            _, output_nodes, server_node_graph, masks = sampler.sample(g, my_seed)
        
        mask_dir = 'masks'
        
        if not os.path.exists(f'{mask_dir}/{args.node_rank}'): 
            os.makedirs(f'{mask_dir}/{args.node_rank}') 
        
        if args.use_mask and len(masks) < args.n_layers:
            all_false = torch.full(masks[-1].shape, False)
            for _ in range(args.n_layers - len(masks)):
                masks.insert(0, all_false)
                
        
        torch.save(masks, f'{mask_dir}/{args.node_rank}/{iter}.pt')
        
        
        

        print('Output nodes...', output_nodes, output_nodes.shape)

        print('Original node id...', server_node_graph.ndata[dgl.NID], server_node_graph.ndata[dgl.NID].shape)

        print('New node id...', server_node_graph.nodes(), server_node_graph.nodes().shape)
        

        full_ids = server_node_graph.ndata[dgl.NID]

        if args.inductive:
            server_node_graph = server_node_graph.subgraph(server_node_graph.ndata['train_mask'].nonzero().squeeze())
            assert torch.equal(server_node_graph.nodes(), torch.arange(len(server_node_graph.nodes())))
            
            full_ids = full_ids[server_node_graph.ndata[dgl.NID]]  

        
        if args.sampler != 'saint':
            assert set(full_ids[:len(output_nodes)].tolist()) == set(output_nodes.tolist())
            assert torch.equal(full_ids[:len(output_nodes)], torch.tensor(output_nodes))
        

        server_node_graph.ndata['seed_node'] = torch.zeros(len(server_node_graph.nodes()), dtype=bool)
        server_node_graph.ndata['seed_node'][:len(output_nodes)] = True

        sid_from_psid = server_node_graph_partition(server_node_graph, iter, args, my_seed=my_seed)
        dist.barrier(args.head_group)


def init_processes(rank: int, args: 'Args') -> None:
    """ Initialize the distributed environment. """
    os.environ['GLOO_SOCKET_IFNAME'] = 'ib0'
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.port)
    
    
    
    
    
    

    dist.init_process_group(args.backend, rank=rank, world_size=args.local_device_cnt*args.total_nodes, timeout=datetime.timedelta(seconds=10800))
    if args.backend == 'nccl':
        torch.cuda.set_device(rank % args.local_device_cnt)
    _set_additional_args_parameters(args)

    _fix_seeds(args)

    

    if args.rank % args.local_device_cnt == 0:  
        _preprocess_partition_graph(args)
    dist.barrier() 

    _local_run(args)


def _local_run(args: 'Args') -> None:
    
    rank, size = dist.get_rank(), dist.get_world_size()
    local_rank, local_size = rank % args.local_device_cnt, args.local_device_cnt

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args, args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g, full_g
        del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)
    
    
    layer_size = get_layer_size(args, args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    model = _create_model(layer_size, args)
    model.cuda()
    model = DDP(model)

    best_model, best_acc = None, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (args.dataset, args.local_device_cnt, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.local_device_cnt, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.local_device_cnt, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.local_device_cnt, int(args.enable_pipeline))
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    

    train_dur, comm_dur, reduce_dur, switch_dur = [], [], [], []
    val_accs = []
    torch.cuda.reset_peak_memory_stats()

    thread = None
    if not args.sequential_eval:  
        pool = ThreadPool(processes=1)

    gpu_masks = list()

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        for iter in range(args.epoch_iter):
            
            if epoch == 0 or args.epoch_iter != 1:
                pre_switch = time.time()
                
                
                
                graph, node_dict, gpb = load_intra_partition(args, local_rank, iter)
                if args.use_flexible:
                    flex_graph, flex_node_dict, flex_gpb = load_partition(args, rank)
                    flex_graph = flex_graph.int().to(torch.device('cuda'))
                graph, node_dict = _move_to_cuda(graph, node_dict)
                if 'part_id' not in node_dict:
                    node_dict['part_id'] = [0] * graph.num_nodes()
                num_in = node_dict['inner_node'].sum().item()
                assert torch.equal(node_dict[dgl.NID][:num_in], gpb.partid2nids(local_rank).cuda())
                assert torch.equal(node_dict['inner_node'].nonzero().flatten(), torch.arange(num_in, device='cuda'))

                """
                Original flex_preload utilized the above flex_graph and mapper class/methods.
                For example, for 3-layer setting, 1-hop preload / 2-hop non-preload,
                We utilize 1-hop preload graph and change the output result with mapper.
                For this implementation, we utilize one baseline_model and one model.
                Using the model we can process 1-hop preload then mapper change the mapping to baseline_model.
                Using the baseline_model, we can generate the final output.
                
                Change Note)
                We found that just using expansion-aware sampling is much better choice than flex_preload.
                It is because the remapping overhead is higher than using expansion-aware sampling.
                Therefore, when using cluster, just using expansion-aware sampling is recommended.
                
                Usage Note)
                When memory size / GPU is enough, just use flex preload with #layers.
                If not, utilize expansion-aware sampling with low hop and fanout hyperparameter.
                Because, expansion-aware sampling is much advantageous due to remapping overhead.
                """
                

                
                boundary = get_intra_boundary(args, node_dict, gpb)

                
                graph, mapper = _get_intra_graph(args, graph, node_dict)

                if args.use_mask:
                    mask_dir = 'masks'
                    masks = torch.load(f'{mask_dir}/{args.node_rank}/{iter}.pt')

                    def _is_in_mask(mask, mapper):
                        return [(x in mask) for x in mapper]
                    
                    n_threads = 30
                    
                    for l_id in range(args.n_layers):
                        agg_results = None
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            
                            futures = [executor.submit(_is_in_mask, masks[l_id], param) for param in torch.split(mapper.cpu(), n_threads)]
                            results = concurrent.futures.wait(futures, timeout=10800)
                            agg_results = list(torch.tensor(res.result()) for res in results.done)
                            failed_agg_results = [res.result() for res in results.not_done]
                            assert len(failed_agg_results) == 0, 'We does not admit failure!'
                        
                        gpu_masks.append(torch.cat(tuple(agg_results)).to('cuda'))

                    
                    
                    
                        
                
                recv_shape = _get_intra_recv_shape(args, node_dict)

                ctx.buffer.init_buffer(args, num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear], \
                                    args.model, use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline, \
                                    corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum, debug=args.debug)
                del boundary

                train_mask = node_dict['train_mask']
                labels = node_dict['label'][train_mask]
                part_train = train_mask.sum().item()

                feat = node_dict['feat']

                
                node_dict.pop('train_mask')
                node_dict.pop('inner_node')
                node_dict.pop('part_id')
                node_dict.pop(dgl.NID)

                if not args.eval:
                    node_dict.pop('val_mask', None)
                    node_dict.pop('test_mask', None)

                switch_time = time.time() - pre_switch
            

            if args.model in ['graphsage', 'deepgcn']:
                in_deg = node_dict['in_degree']
                if args.use_mask:
                    logits = model(graph, feat, in_deg, gpu_masks)
                else:
                    logits = model(graph, feat, in_deg)
            else:
                raise Exception


            if args.inductive: 
                if args.use_mask:
                    logits_shape = logits.shape[0]
                    loss = loss_fcn(logits[~gpu_masks[-1][:logits_shape]], \
                                    labels[~gpu_masks[-1][:logits_shape]])
                else:
                    loss = loss_fcn(logits, labels)
            else:
                loss = loss_fcn(logits[train_mask], labels)
            del logits

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            

            pre_reduce = time.time()
            reduce_time = time.time() - pre_reduce

            optimizer.step()

            if args.epoch_iter > 1:
                torch.cuda.synchronize()
                del graph
                del feat
                torch.cuda.empty_cache()
                gc.collect()

            if epoch >= 5 and epoch % args.log_every != 0:
                if args.debug:
                    torch.cuda.synchronize()
                train_dur.append(time.time() - t0)
                comm_dur.append(ctx.comm_timer.tot_time())
                reduce_dur.append(reduce_time)
                switch_dur.append(switch_time)

            if (epoch + 1) % 10 == 0:
                print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                    rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

            ctx.comm_timer.clear()

            loss_scalar = loss.item()
            del loss
            ctx.buffer.next_epoch()

        
        if args.sequential_eval:
            dist.barrier()

        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            print('Evaluation just started....')
            if args.sequential_eval:
                model_copy = copy.deepcopy(model)
                if not args.inductive:
                    model_copy, val_acc = _evaluate_trans('Epoch %05d' % epoch, model_copy, val_g, result_file_name)
                else:
                    model_copy, val_acc = _evaluate_induc('Epoch %05d' % epoch, model_copy, val_g, 'val', result_file_name)
                val_accs.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    del best_model
                    best_model = model_copy
                else:
                    del model_copy

            else:
                if thread is not None:
                    model_copy, val_acc = thread.get()
                    val_accs.append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = model_copy
                model_copy = copy.deepcopy(model)
                if not args.inductive:
                    thread = pool.apply_async(_evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                    val_g, result_file_name))
                else:
                    thread = pool.apply_async(_evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                    val_g, 'val', result_file_name))
            print('Evaluation finished...!')

        if args.sequential_eval:
            dist.barrier()
        


    if args.create_json:
        info_dict = {}
        test_acc = 0.0

    if rank == 0 and not args.time_calc:
        ckpt_path = 'model/'
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_str = ckpt_path + args.dataset + '_granndis_' + datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '.pth.tar'

        if args.eval:
            if thread is not None:
                model_copy, val_acc = thread.get()
                val_accs.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy

            torch.save(best_model.state_dict(), ckpt_str)
            print('model saved')
            print("Validation accuracy {:.2%}".format(best_acc))
            _, acc = _evaluate_induc('Test Result', best_model, test_g, 'test')
            test_acc = acc

        else:
            model_copy = copy.deepcopy(model)
            model_copy.cpu()

            torch.save(model_copy.state_dict(), ckpt_str)
            print('model saved')


    
    if args.create_json :
        info_dict['best_accuracy'] = best_acc
        info_dict['test_accuracy'] = test_acc
        log_path = args.json_path
        os.makedirs(log_path, exist_ok=True)
        print('json logs will be saved in ', log_path)


    summary_buffer = torch.zeros(5).cuda()
    summary_buffer[0] = np.mean(train_dur)
    summary_buffer[1] = np.mean(comm_dur)
    summary_buffer[2] = np.mean(reduce_dur)
    summary_buffer[3] = loss_scalar/part_train
    summary_buffer[4] = np.mean(switch_dur)

    dist.all_reduce(summary_buffer, op=dist.ReduceOp.SUM)

    summary_buffer = summary_buffer/dist.get_world_size()

    if dist.get_rank() == 0:

        print("="*30, "Speed result summary", "="*30)

        print("train_duration : ", summary_buffer[0].item())
        print("communication_duration : ", summary_buffer[1].item())
        print("reduce duration : ", summary_buffer[2].item())
        print("loss : ", summary_buffer[3].item())
        print("="*60)



    if args.create_json:

        log_path = args.json_path

        info_dict['dataset'] = args.dataset
        info_dict['rank'] = rank
        info_dict['train_dur_mean'] = np.mean(train_dur)
        info_dict['comm_dur_mean'] = np.mean(comm_dur)
        info_dict['reduce_dur_mean'] = np.mean(reduce_dur)
        info_dict['loss'] = loss_scalar/part_train
        info_dict['eval_epoch_interval'] = epoch
        info_dict['train_dur_array'] = train_dur
        info_dict['comm_dur_array'] = comm_dur
        info_dict['reduce_dur_array'] = reduce_dur
        

        for k, v in vars(args).items():
            if 'group' in k :
                continue

            info_dict[k] = v


        info_dict.pop('ssh_username', None)
        info_dict.pop('ssh_pwd', None)

        timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        raw_log_path = log_path + '/json_raws'
        os.makedirs(raw_log_path, exist_ok=True)

        file_name = raw_log_path + ('/') + 'exp_id_' + str(args.exp_id) + '_' + timestr +'_rank_' + str(dist.get_rank())

        info_dict['timestr'] = timestr


        with open(file_name, "w") as outfile:
            json.dump(info_dict, outfile)
        print("Rank ", dist.get_rank(), "successfully created json log file.")

        
        summary_buffer = torch.zeros(5).cuda()
        summary_buffer[0] = np.mean(train_dur)
        summary_buffer[1] = np.mean(comm_dur)
        summary_buffer[2] = np.mean(reduce_dur)
        summary_buffer[3] = loss_scalar/part_train
        summary_buffer[4] = np.mean(switch_dur)

        dist.all_reduce(summary_buffer, op=dist.ReduceOp.SUM)

        summary_buffer = summary_buffer/dist.get_world_size()

        memory_buffer = torch.tensor([torch.cuda.max_memory_reserved('cuda')]).cuda()
        dist.all_reduce(memory_buffer, op=dist.ReduceOp.SUM)


        if dist.get_rank() == 0:
            
            best_acc_tensor = torch.tensor([best_acc]).to('cuda')
            info_dict_summary = {}

            summary_buffer = torch.cat((summary_buffer, best_acc_tensor))
            summary_buffer_list = summary_buffer.tolist()

            info_dict_summary['dataset'] = args.dataset
            info_dict_summary['train_dur_aggregated'] = summary_buffer[0].item()*args.epoch_iter
            info_dict_summary['comm_dur_aggregated'] = summary_buffer[1].item()*args.epoch_iter
            info_dict_summary['reduce_dur_aggregated'] = summary_buffer[2].item()*args.epoch_iter
            info_dict_summary['loss_aggregated'] = summary_buffer[3].item()
            info_dict_summary['switch_dur_aggregated'] = summary_buffer[4].item()
            info_dict_summary['best_accuracy'] = summary_buffer[5].item()
            info_dict_summary['avg_memory_reserved'] = memory_buffer[0].item() / dist.get_world_size() / (1024*1024*1024)
            info_dict_summary['val_accs'] = val_accs


            info_dict_summary['test_accuracy'] = test_acc

            os.makedirs(log_path, exist_ok=True)


            summary_file_name = log_path + ('/') + 'exp_id_' + str(args.exp_id) + '_' + timestr +'_summary'

            with open(summary_file_name, "w") as outfile:
                json.dump(info_dict_summary, outfile)
            print("Rank ", dist.get_rank(), "successfully created summary json log file.")

            for k, v in vars(args).items():
                if 'group' in k :
                    continue

                info_dict_summary[k] = v

            info_dict_summary.pop('ssh_username', None)
            info_dict_summary.pop('ssh_pwd', None)

            if args.send_db :

                try : 

                    db_config['mongo_db'] = args.db_name
                    db_config['project'] = args.project
                    db_config['ssh_username'] = args.ssh_user
                    db_config['ssh_pwd'] = args.ssh_pwd
                    
                    try:    
                        
                        mongo = DBHandler()
                    except Exception as e:
                        mongo = DBHandler()
                        
                    try:
                        mongo.insert_item_one(info_dict_summary)
                    except Exception as e:
                        mongo.insert_item_one(info_dict_summary)

                    print('Rank ', dist.get_rank(), "successfully sended a log to DB.")

                    try:
                        
                        mongo.close_connection()
                    except Exception as e:
                        mongo.close_connection()
                
                except Exception as e :
                    print("Sending logs to DB failed. Tracebacks are as follows.")
                    traceback.print_exc()

            
            print("="*30, "Training result summary", "="*30)
            print("train_duration : ", info_dict_summary['train_dur_aggregated'])
            print("communication_duration : ", info_dict_summary['comm_dur_aggregated'])
            print("reduce duration : ", info_dict_summary['reduce_dur_aggregated'])
            print("loss : ", info_dict_summary['loss_aggregated'])
            print("best accuracy : ", info_dict_summary['best_accuracy'])
            print("avg memory : ", info_dict_summary['avg_memory_reserved'])
            print("="*60)
    
