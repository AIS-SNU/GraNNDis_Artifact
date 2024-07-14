import os
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist


class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {}
        self._pool = None
        self._handles = []
        self._stream = None

    def init(self, model, world_size):

        num_params = len(list(model.named_parameters())) 
        num_workers = int(os.cpu_count() / world_size)
        
        for i, (name, param) in enumerate(model.named_parameters()):
            cur_group = dist.new_group()  
            self._data_cpu[name] = (torch.zeros_like(param.data, pin_memory=True, device='cpu'), cur_group)

        self._pool = ThreadPool(processes= num_workers)
        self._stream = torch.cuda.Stream()

    def reduce(self, param, name, data, n_train):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data)
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True)

        self._handles.append(self._pool.apply_async(create_stream))

    def synchronize(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
