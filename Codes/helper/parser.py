import argparse
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch.distributed as dist
    from helper.mapper import Mapper


class Args:
    dataset: str = 'reddit'
    "The input dataset"
    dataset_scale: int = 1
    "Scaling factor for dataset"
    dataset_path: str = "/datasets/atc23/"
    "Dataset path"
    graph_name: str = ''
    
    "sampler type"
    sampler: str = 'sage'

    model: str = 'graphsage'
    "Model for training"
    dropout: float = 0.5
    "Dropout probability"
    lr: float = 1e-2
    "Learning rate"
    n_epochs: int = 200
    "The number of training epochs"
    n_partitions: int = 2
    "The number of partitions"
    n_hidden: int = 16
    "The number of hidden units"
    n_layers: int = 2
    "The number of GCN layers"
    n_linear: int = 0
    "The number of linear layers"
    norm: Literal['layer', 'batch', 'none'] = 'layer'
    "Normalization method"
    weight_decay: float = 0
    "Weight for L2 loss"

    n_feat: int = 0
    n_class: int = 0
    n_train: int = 0
    skip_partition: bool = False
    "Skip graph partition"

    partition_obj: Literal['vol', 'cut'] = 'vol'
    "Partition objective function ('vol' or 'cut')"
    partition_method: Literal['metis', 'random'] = 'metis'
    "The method for graph partition ('metis' or 'random')"

    enable_pipeline: bool = False
    feat_corr: bool = False
    grad_corr: bool = False
    corr_momentum: float = 0.95

    use_mask: bool = False
    bandwidth_aware: bool = False
    "Whether to use intra-inter bandwidth aware method"
    remove_tmp: bool = False
    "Remove intra-partitions..."
    time_calc: bool = False
    "Time calculation..."
    epoch_iter: int = 1
    fanout: int = 5
    subgraph_hop: int = 1
    "Subgraph hyperparameter"
    flexible_hop: int = 0
    "flexible preloading hop"
    use_flexible: bool = False
    check_intra_only: bool = False
    "Check intra only comm time for evaluation...."
    full_neighbor: bool = False
    "Whether to use full neighbor mini-batch generation"
    skip_minibatch_partition: bool = False
    "Skip minibatch generation and partition if already exists"
    
    use_pp: bool = False
    "Whether to use precomputation"
    inductive: bool = False
    "Inductive or transductive learning setting"
    fix_seed: bool = False
    "Fix random seed"
    seed: int = 0
    log_every: int = 10

    backend: Literal['gloo', 'nccl', 'mpi'] = 'nccl'
    master_addr: str = "127.0.0.1"
    port: int = 18118
    "The network port for communication"
    key_value_port: int = 18118
    "The network port for communication"  
    node_rank : int = 0
    total_nodes: int = 2
    parts_per_node: int = 10
    rank: int
    "[Warning] This is not a part of the parser."
    world_size: int
    "[Warning] This is not a part of the parser."
    local_rank: int
    "[Warning] This is not a part of the parser."
    local_device_cnt: int
    "Number of local devices. [Warning] This is not a part of the parser."

    head_group: 'dist.ProcessGroup'
    "Group consists of the head of the nodes. (ex: [0, 4, 8, ...]) [Warning] This is not a part of the parser."
    device_group: 'dist.ProcessGroup'
    "Group consists of the local devices. (ex: [4, 5, 6, 7]) [Warning] This is not a part of the parser."
    mapper: 'Mapper'
    "Mapper class between full-graph, server-graph, gpu-graph. [Warning] This is not a part of the parser."

    eval: bool = True  
    "Enable evaluation"
    sequential_eval: bool = False
    "If true, evaluate sequentially. Else, make another thread and do evaluation while training next epoch."

    create_json: int = 0
    send_db: int = 0
    db_name: str
    "DB name... normally nickname"
    project: str
    "Project(experiment) name"
    ssh_user: str
    "SSH username"
    ssh_pwd: str
    "SSH password"
    json_path: str = "./json_logs"

    debug: bool = False
    exp_id: int = 0


def create_parser() -> Args:

    parser = argparse.ArgumentParser(description='PipeGCN')

    parser.add_argument("--dataset", type=str, default='reddit',
                        help="the input dataset")
    parser.add_argument("--dataset-scale", type=int, default=1, help='scaling factor for dataset')
    parser.add_argument("--graph-name", "--graph_name", type=str, default='')
    
    parser.add_argument("--sampler", type=str, default='sage',
                        help='sampler type')

    parser.add_argument("--model", type=str, default='graphsage',
                        help="model for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=200,
                        help="the number of training epochs")
    parser.add_argument("--n-partitions", "--n_partitions", type=int, default=2,
                        help="the number of partitions")
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=16,
                        help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2,
                        help="the number of GCN layers")
    parser.add_argument("--n-linear", "--n_linear", type=int, default=0,
                        help="the number of linear layers")
    parser.add_argument("--norm", choices=['layer', 'batch'], default='layer',
                        help="normalization method")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0,
                        help="weight for L2 loss")

    parser.add_argument("--n-feat", "--n_feat", type=int, default=0)
    parser.add_argument("--n-class", "--n_class", type=int, default=0)
    parser.add_argument("--n-train", "--n_train", type=int, default=0)
    parser.add_argument('--skip-partition', action='store_true',
                        help="skip graph partition")

    parser.add_argument("--partition-obj", "--partition_obj", choices=['vol', 'cut'], default='vol',
                        help="partition objective function ('vol' or 'cut')")
    parser.add_argument("--partition-method", "--partition_method", choices=['metis', 'random'], default='metis',
                        help="the method for graph partition ('metis' or 'random')")

    parser.add_argument("--enable-pipeline", "--enable_pipeline", action='store_true')
    parser.add_argument("--feat-corr", "--feat_corr", action='store_true')
    parser.add_argument("--grad-corr", "--grad_corr", action='store_true')
    parser.add_argument("--corr-momentum", "--corr_momentum", type=float, default=0.95)

    parser.add_argument("--use-mask", "--use_mask", action='store_true', 
                        help='whether to use mask based unified batching to bandwidth aware method.')
    parser.add_argument("--bandwidth-aware", "--bandwidth_aware", action='store_true',
                        help='whether to use intra-inter bandwidth aware method.')
    parser.add_argument('--remove-tmp', '--remove_tmp', action='store_true',
                        help='remove intra-partitions...')
    parser.add_argument('--time_calc', '--time-calc', action='store_true',
                        help='time calculation...')
    
    parser.add_argument("--epoch-iter", "--epoch-iter", type=int, default=1)
    parser.add_argument("--fanout", type=int, default=5)
    parser.add_argument("--subgraph-hop", "--subgraph_hop", type=int, default=1) 
    parser.add_argument("--use-flexible", "--use_flexible", action='store_true', \
                        help='use flexible preloading')
    parser.add_argument("--flexible-hop", "--flexible_hop", type=int, default=0)
    parser.add_argument("--check-intra-only", "--check_intra_only", action='store_true', \
                        help='check intra only comm time for evaluation....')
    parser.add_argument("--full-neighbor", "--full_neighbor", action='store_true',
                        help="whether use full neighbor mini-batch generation")
    parser.add_argument("--skip-minibatch-partition", "--skip_minibatch_partition", action='store_true',
                        help='skip minibatch generation and partition if already exists.')

    parser.add_argument("--use-pp", "--use_pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--inductive", action='store_true',
                        help="inductive learning setting")
    parser.add_argument("--fix-seed", "--fix_seed", action='store_true',
                        help="fix random seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", "--log_every", type=int, default=10)

    parser.add_argument("--backend", type=str, default='nccl')
    parser.add_argument("--port", type=int, default=18118,
                        help="the network port for communication")
    parser.add_argument("--key-value-port", "--key_value_port", type=int, default=18118,
                        help="the network port for communication")
    parser.add_argument("--master-addr", "--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--node-rank", "--node_rank", type=int, default=0)
    parser.add_argument('--total-nodes', '--total_nodes', type=int, default=2)
    parser.add_argument("--parts-per-node", "--parts_per_node", type=int, default=10)

    parser.add_argument('--eval', action='store_true',
                        help="enable evaluation")
    parser.add_argument('--no-eval', action='store_false', dest='eval',
                        help="disable evaluation")
    parser.add_argument('--sequential-eval', '--sequential_eval', action='store_true',
                        help="if true, evaluate sequentially; else, make another thread and do evaluation while training next epoch.")

    parser.add_argument('--dataset-path', '--dataset_path', default='/datasets/atc23/', type=str, \
                        help='dataset path')

    parser.add_argument("--create-json", "--create_json", type=int, default=0)
    parser.add_argument("--send-db", "--send_db", type=int, default=0)
    parser.add_argument('--db-name', '--db_name', help='db name... normally nickname')
    parser.add_argument('--project', help='project(experiment) name')
    parser.add_argument('--ssh-user', '--ssh_user', help='ssh username')
    parser.add_argument('--ssh-pwd', '--ssh_pwd', help='ssh password')
    parser.add_argument('--json-path', '--json_path', type=str, default='./json_logs')

    parser.add_argument("--debug", action='store_true')    
    parser.add_argument("--exp-id", "--exp_id", type=int, default=0)

    parser.set_defaults(eval=True)
    parser.set_defaults(sequential_eval=False)

    return parser.parse_args()
