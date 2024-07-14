from typing import List
from os import listdir
from os.path import isfile, join

import prettytable
import json

def find_acc(dict_list: List, dataset_name: str, acc_type: str) -> float:
    r""" find proper duration from dictionary list of various datasets
    """
    for element in dict_list:
        if element['dataset'] == dataset_name:
            if acc_type in element:
                return element[acc_type]
            else:
                raise NotImplementedError

if __name__ == '__main__':
    # prepare list
    fb_list = []
    eas_list = []
    
    # parse fb
    fb_basedir = '../Logs/fb_acc'
    fb_files =  [f for f in listdir(fb_basedir) \
        if isfile(join(fb_basedir, f))]
    assert len(fb_files) == 3, 'we tested only three datasets.'
    for fb_file in fb_files:
        with open(join(fb_basedir, fb_file)) as json_file:
            fb_data = json.load(json_file)
            fb_list.append(fb_data)

    # parse eas
    eas_basedir = '../Logs/eas_acc'
    eas_files =  [f for f in listdir(eas_basedir) \
        if isfile(join(eas_basedir, f))]
    assert len(eas_files) == 3, 'we tested only three datasets.'
    for eas_file in eas_files:
        with open(join(eas_basedir, eas_file)) as json_file:
            eas_data = json.load(json_file)
            eas_list.append(eas_data)
            
    acc_table = prettytable.PrettyTable()
    acc_table.title = 'Accuracy Comparison (FB vs. FLX-EAS)'
    acc_table.field_names = ['Method', 'Arxiv', 'Reddit', 'Products']
    acc_table.add_row(['FB', find_acc(fb_list, 'ogbn-arxiv', 'test_accuracy'),
                       find_acc(fb_list, 'reddit', 'test_accuracy'),
                       find_acc(fb_list, 'ogbn-products', 'test_accuracy')])
    acc_table.add_row(['EAS', find_acc(eas_list, 'ogbn-arxiv', 'test_accuracy'),
                       find_acc(eas_list, 'reddit', 'test_accuracy'),
                       find_acc(eas_list, 'ogbn-products', 'test_accuracy')])
    acc_table.float_format = ".2"
    print(acc_table)