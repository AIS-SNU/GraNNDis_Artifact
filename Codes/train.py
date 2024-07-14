import torch.nn.functional as F
from module.baseline_model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score

from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback
from helper.MongoManager import *

import datetime

def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict, device):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].to(device)
    graph = graph.int().to(device)
    part = part.int().to(device)

    return graph, part, node_dict


def get_pos(node_dict, gpb, device):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, device= device)
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct(part, graph, pos, one_hops)


def move_train_first(graph, node_dict, boundary, device):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device=device)
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device=device)
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device=device)

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'deepgcn':
        return DeeperGCN(layer_size, nn.ReLU(inplace=True), args.use_pp, args.n_feat, args.n_class, norm=args.norm, dropout=args.dropout,
                        n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def run(graph, node_dict, gpb, device, args):
    
    rank, size = dist.get_rank(), dist.get_world_size()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args, args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    part = create_inner_graph(graph.clone(), node_dict)


    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict, device)

    boundary = get_boundary(node_dict, gpb, device)
    

    layer_size = get_layer_size(args, args.n_feat, args.n_hidden, args.n_class, args.n_layers)

    pos = get_pos(node_dict, gpb, device)
    graph = order_graph(part, graph, gpb, node_dict, pos)
    in_deg = node_dict['in_degree']

    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary, device)

    recv_shape = get_recv_shape(node_dict)

    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           args.model, use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline,
                           corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum, debug= args.debug, check_intra_only = args.check_intra_only)

    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)

    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()


    del boundary
    del part
    del pos

    model = create_model(layer_size, args)
    
    model.to(device)
    
    model = DDP(model)

    
    
    
    
    

    
    

    best_model, best_acc = None, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur = [], [], []
    val_accs = []
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    if not args.eval:
        if 'val_mask' in node_dict.keys():
            node_dict.pop('val_mask')
        if 'test_mask' in node_dict.keys():
            node_dict.pop('test_mask')

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        if args.model in ['graphsage', 'deepgcn'] :
            logits = model(graph, feat, in_deg)
        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        ctx.buffer.next_epoch()

        pre_reduce = time.time()
        
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5 and epoch % args.log_every != 0:
            if args.debug:
                torch.cuda.synchronize()
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)

        if (epoch + 1) % 10 == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        ctx.comm_timer.clear()


        loss_scalar = loss.item()
        del loss

        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                model_copy, val_acc = thread.get()
                val_accs.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, result_file_name))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', result_file_name))


    if args.create_json:
        info_dict = {}
        test_acc = 0.0

    if args.eval and rank == 0 and not args.time_calc:
        if thread is not None:
            model_copy, val_acc = thread.get()
            val_accs.append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        ckpt_path = 'model/'
        os.makedirs(ckpt_path, exist_ok=True)

        args.ckpt_str = ckpt_path + args.dataset + '_fullgraph_' + datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '.pth.tar'

        torch.save(best_model.state_dict(), args.ckpt_str)
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))
        _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')
        test_acc = acc

        if args.create_json :
            info_dict['best_accuracy'] = best_acc
            info_dict['test_accuracy'] = test_acc
            log_path = args.json_path            
            os.makedirs(log_path, exist_ok=True)
            print('json logs will be saved in ', log_path)

    if not args.eval and rank == 0:
        ckpt_path = 'model/'
        os.makedirs(ckpt_path, exist_ok=True)

        model_copy = copy.deepcopy(model)
        model_copy.cpu()

        args.ckpt_str = ckpt_path + args.dataset + '_fullgraph_model_optimizer_' + datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '.pth.tar'

        torch.save({
            'epoch': args.n_epochs,
            'model_state_dict': model_copy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, args.ckpt_str)
        print('model saved')

    summary_buffer = torch.zeros(4).to(device)
    summary_buffer[0] = np.mean(train_dur)
    summary_buffer[1] = np.mean(comm_dur)
    summary_buffer[2] = np.mean(reduce_dur)
    summary_buffer[3] = loss_scalar/part_train

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

        
        summary_buffer = torch.zeros(4).to(device)
        summary_buffer[0] = np.mean(train_dur)
        summary_buffer[1] = np.mean(comm_dur)
        summary_buffer[2] = np.mean(reduce_dur)
        summary_buffer[3] = loss_scalar/part_train

        dist.all_reduce(summary_buffer, op=dist.ReduceOp.SUM)

        summary_buffer = summary_buffer/dist.get_world_size()

        memory_buffer = torch.tensor([torch.cuda.max_memory_reserved(device)]).to(device)
        dist.all_reduce(memory_buffer, op=dist.ReduceOp.SUM)


        if dist.get_rank() == 0:
            
            best_acc_tensor = torch.tensor([best_acc]).to(device)
            info_dict_summary = {}

            summary_buffer = torch.cat((summary_buffer, best_acc_tensor))
            summary_buffer_list = summary_buffer.tolist()

            info_dict_summary['dataset'] = args.dataset
            info_dict_summary['train_dur_aggregated'] = summary_buffer[0].item()
            info_dict_summary['comm_dur_aggregated'] = summary_buffer[1].item()
            info_dict_summary['reduce_dur_aggregated'] = summary_buffer[2].item()
            info_dict_summary['loss_aggregated'] = summary_buffer[3].item()
            info_dict_summary['best_accuracy'] = summary_buffer[4].item()
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



def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['GLOO_SOCKET_IFNAME']='ib0'
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    
    dist.init_process_group(args.backend, rank=rank, world_size=size, timeout=datetime.timedelta(seconds=10800))
    
    if args.backend == 'gloo':
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = str(rank % torch.cuda.device_count())
        device = "cuda:" + device
        torch.cuda.set_device(device)
    

    print("My device:", device)

    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)

    torch.manual_seed(args.seed)

    run(g, node_dict, gpb, device, args)
