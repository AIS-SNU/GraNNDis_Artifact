import torch
import warnings
from module.model import *
from helper.utils import *
from sklearn.metrics import f1_score
import argparse

"""
Tester for Hyper-scale Datasets (e.g., Papers-100M)

e.g.
python test.py --dataset yelp --ckpt ./model/yelp_granndis_2023_05_14__17_34_57.pth.tar --n-hidden 512 --n-layers 15 --n-linear 1
"""

def get_layer_size(args, n_feat, n_hidden, n_class, n_layers):
    
    if args.model in ['deepgcn']:
        layer_size = [n_hidden] 
    else:
        layer_size = [n_feat]
    
    
    layer_size.extend([n_hidden] * (n_layers - 1))
    
    layer_size.append(n_class)
    return layer_size

def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'deepgcn':
        return DeeperGCN(layer_size, nn.ReLU(inplace=True), args.use_pp, args.n_feat, args.n_class, norm=args.norm, dropout=args.dropout,
                        n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError

def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


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
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Code for Hyper-scale Datasets')

    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--inductive', action='store_true', help='inductive learning setting')
    parser.add_argument('--ckpt', type=str, required=True, help='ckpt path')

    parser.add_argument('--model', type=str, default='deepgcn', help='model type')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=16,
                        help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2,
                        help="the number of GCN layers")
    parser.add_argument("--n-linear", "--n_linear", type=int, default=0,
                        help="the number of linear layers")
    parser.add_argument("--norm", choices=['layer', 'batch'], default='layer',
                        help="normalization method")
    parser.add_argument("--use-pp", "--use_pp", action='store_true',
                        help="whether to use precomputation")

    parser.add_argument('--dataset-path', '--dataset_path', default='/datasets/atc23/', type=str, \
                        help='dataset path')

    args = parser.parse_args()

    g, n_feat, n_class = load_data(args, args.dataset)
    args.n_feat = n_feat
    args.n_class = n_class
    args.n_train = g.ndata['train_mask'].int().sum().item()

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        del train_g
        del val_g
        del g
    else:
        test_g = g


    layer_size = get_layer_size(args, args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    model = create_model(layer_size, args)
    
    loaded = torch.load(args.ckpt)

    
    for key in list(loaded.keys()):
        key_split = key.split('.')
        loaded['.'.join(key_split[1:])] = loaded.pop(key)

    model.load_state_dict(loaded)

    _, acc = evaluate_induc('Test Result', model, test_g, 'test')

    print('Testing Finished.... ACC. ' + str(acc*100) + ' (%) !!!')


