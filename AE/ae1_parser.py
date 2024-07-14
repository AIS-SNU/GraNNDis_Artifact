from typing import List
from os import listdir
from os.path import isfile, join

import prettytable
import json

def find_dur(dict_list: List, dataset_name: str, dur_type: str) -> float:
    r""" find proper duration from dictionary list of various datasets
    """
    for element in dict_list:
        if element['dataset'] == dataset_name:
            if dur_type in element:
                return element[dur_type]
            else:
                raise NotImplementedError
    

if __name__ == '__main__':
    # prepare list
    opt_baseline_list = []
    flx_list = []
    cob_list = []
    eas_list = []
    
    # parse opt_baseline
    opt_baseline_basedir = '../Logs/granndis_opt_baseline'
    opt_baseline_files = [f for f in listdir(opt_baseline_basedir) \
        if isfile(join(opt_baseline_basedir, f))]
    assert len(opt_baseline_files) == 3, 'we tested only three datasets.'
    for opt_baseline_file in opt_baseline_files:
        with open(join(opt_baseline_basedir, opt_baseline_file)) as json_file:
            opt_baseline_data = json.load(json_file)
            opt_baseline_list.append(opt_baseline_data)
    
    # parse flx (flexible preloading)
    flx_basedir = '../Logs/granndis_flx'
    flx_files = [f for f in listdir(flx_basedir) \
        if isfile(join(flx_basedir, f))]
    assert len(flx_files) == 3, 'we tested only three datasets.'
    for flx_file in flx_files:
        with open(join(flx_basedir, flx_file)) as json_file:
            flx_data = json.load(json_file)
            flx_list.append(flx_data)
            
    # parse cob (cooperative batching)
    cob_basedir = '../Logs/granndis_cob'
    cob_files = [f for f in listdir(cob_basedir) \
        if isfile(join(cob_basedir, f))]
    assert len(cob_files) == 3, 'we tested only three datasets.'
    for cob_file in cob_files:
        with open(join(cob_basedir, cob_file)) as json_file:
            cob_data = json.load(json_file)
            cob_list.append(cob_data)

    # parse eas (expansion-aware sampling)
    eas_basedir = '../Logs/granndis_eas'
    eas_files = [f for f in listdir(eas_basedir) \
        if isfile(join(eas_basedir, f))]
    assert len(eas_files) == 3, 'we tested only three datasets.'
    for eas_file in eas_files:
        with open(join(eas_basedir, eas_file)) as json_file:
            eas_data = json.load(json_file)
            eas_list.append(eas_data)
            
    ae1_arxiv = prettytable.PrettyTable()
    ae1_arxiv.title = 'Throughput Results for Arxiv'
    ae1_arxiv.field_names = ['Method', 'Total Time (sec)', 'Comm Time (sec)', 'Speedup']
    base_tot = find_dur(opt_baseline_list, 'ogbn-arxiv', 'train_dur_aggregated')*1000
    ae1_arxiv.add_row(['Opt_FB', base_tot,
                       find_dur(opt_baseline_list, 'ogbn-arxiv', 'comm_dur_aggregated')*1000, 1.0])
    cur_tot =  find_dur(flx_list, 'ogbn-arxiv', 'train_dur_aggregated')*1000
    ae1_arxiv.add_row(['FLX', cur_tot,
                       find_dur(flx_list, 'ogbn-arxiv', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(cob_list, 'ogbn-arxiv', 'train_dur_aggregated')*1000
    ae1_arxiv.add_row(['CoB', cur_tot,
                       find_dur(cob_list, 'ogbn-arxiv', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(eas_list, 'ogbn-arxiv', 'train_dur_aggregated')*1000
    ae1_arxiv.add_row(['EAS', cur_tot,
                       find_dur(eas_list, 'ogbn-arxiv', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    ae1_arxiv.float_format = ".2"
    print(ae1_arxiv)
    
    ae1_reddit = prettytable.PrettyTable()
    ae1_reddit.title = 'Throughput Results for Reddit'
    ae1_reddit.field_names = ['Method', 'Total Time (sec)', 'Comm Time (sec)', 'Speedup']
    base_tot = find_dur(opt_baseline_list, 'reddit', 'train_dur_aggregated')*1000
    ae1_reddit.add_row(['Opt_FB', base_tot,
                       find_dur(opt_baseline_list, 'reddit', 'comm_dur_aggregated')*1000, 1.0])
    cur_tot =  find_dur(flx_list, 'reddit', 'train_dur_aggregated')*1000
    ae1_reddit.add_row(['FLX', cur_tot,
                       find_dur(flx_list, 'reddit', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(cob_list, 'reddit', 'train_dur_aggregated')*1000
    ae1_reddit.add_row(['CoB', cur_tot,
                       find_dur(cob_list, 'reddit', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(eas_list, 'reddit', 'train_dur_aggregated')*1000
    ae1_reddit.add_row(['EAS', cur_tot,
                       find_dur(eas_list, 'reddit', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    ae1_reddit.float_format = ".2"
    print(ae1_reddit)
    
    ae1_products = prettytable.PrettyTable()
    ae1_products.title = 'Throughput Results for Products'
    ae1_products.field_names = ['Method', 'Total Time (sec)', 'Comm Time (sec)', 'Speedup']
    base_tot = find_dur(opt_baseline_list, 'ogbn-products', 'train_dur_aggregated')*1000
    ae1_products.add_row(['Opt_FB', base_tot,
                       find_dur(opt_baseline_list, 'ogbn-products', 'comm_dur_aggregated')*1000, 1.0])
    cur_tot =  find_dur(flx_list, 'ogbn-products', 'train_dur_aggregated')*1000
    ae1_products.add_row(['FLX', cur_tot,
                       find_dur(flx_list, 'ogbn-products', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(cob_list, 'ogbn-products', 'train_dur_aggregated')*1000
    ae1_products.add_row(['CoB', cur_tot,
                       find_dur(cob_list, 'ogbn-products', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    cur_tot =  find_dur(eas_list, 'ogbn-products', 'train_dur_aggregated')*1000
    ae1_products.add_row(['EAS', cur_tot,
                       find_dur(eas_list, 'ogbn-products', 'comm_dur_aggregated')*1000, base_tot/cur_tot])
    ae1_products.float_format = ".2"
    print(ae1_products)