import torch
from multiprocessing.pool import ThreadPool
from multiprocessing import Event
from helper.timer.timer import *
import queue
import datetime


class IntraBuffer(object):

    def __init__(self):
        super(IntraBuffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._pipeline = False
        self._epoch = 0
        self._feat_cpu, self._grad_cpu = [], []
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        self._f_recv_cpu, self._b_recv_cpu = [], []
        self._f_avg, self._b_avg = [], []
        self._recv_shape = []
        self._pool = None
        self._comm_stream, self._corr_stream = None, None
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        self._corr_momentum = 0
        self._corr_feat, self._corr_grad = False, False
        self._pl, self._pr = [], []

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    def init_buffer(self, args, num_in, num_all, boundary, f_recv_shape, layer_size, model, use_pp=False, backend='gloo',
                    pipeline=False, corr_feat=False, corr_grad=False, corr_momentum=0, debug=False):
        self._rank, self._size = dist.get_rank(), dist.get_world_size()
        self._local_rank, self._local_size = self._rank % args.local_device_cnt, args.local_device_cnt
        self._rank_start_id = args.node_rank * args.local_device_cnt
        self._model = model
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size)

        self._debug = debug
        self._cio = args.check_intra_only

        self._layer_size = layer_size
        self._pipeline = pipeline
        self._epoch = 0
        self._recv_shape = f_recv_shape

        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(self._local_size):
                    if j == self._local_rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3
                if i > 0:
                    self._grad_cpu[i] = tmp2
                    self._b_recv_cpu[i] = tmp4
        elif backend == 'nccl':
            pass
        else:
            raise NotImplementedError

        self._f_buf = [None] * self._n_layers
        self._f_recv, self._b_recv = [], []
        self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []

        self._backend = backend

        self._f_avg, self._b_avg = [None] * self._n_layers, [None] * self._n_layers
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers

        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for j in range(self._local_size):
                if j == self._local_rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)
                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
                    tmp3.append(torch.zeros(s1, device='cuda'))
                    tmp4.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            self._b_recv[i] = tmp2
            if corr_feat:
                self._f_avg[i] = tmp3
            if corr_grad and i > 0:
                self._b_avg[i] = tmp4
            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()

        self._corr_momentum = corr_momentum
        self._corr_feat, self._corr_grad = corr_feat, corr_grad
        self._pool = ThreadPool(processes=2*self._n_layers)
        self._intra_group = args.device_group
        self.__init_pl_pr()

    def next_epoch(self):
        self._epoch += 1


    def __feat_concat(self, layer, feat):
        
        tmp = [feat] 
        for i in range(self._local_size):
            if i != self._local_rank:
                if self._corr_feat:
                    tmp.append(self._f_avg[layer][i])
                else:
                    tmp.append(self._f_recv[layer][i])
        return torch.cat(tmp)


    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        with comm_timer.timer(f'forward_{layer}'):
            
            self.__feat_transfer(self._epoch, layer, feat)
            torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
            if self._debug:
                torch.cuda.synchronize()
            self._f_buf[layer] = self.__feat_concat(layer, feat)
        if self._f_buf[layer].requires_grad:
            """
            DeepGCN enables requires_grad because of node encoder !
            : in detail, it should be requires_grad because
                linear layer should also be updated.
                Therefore, we just skip hook register here!
            """
            
        
            self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
        return self._f_buf[layer]


    def __nccl_all_to_all(self, send_gpu, recv_gpu, tag, corr, avg=None, forward=True, layer=None):
        node_rank = self._rank // self._local_size

        req1, req2 = [], []
        self._comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            for i in range(1, self._local_size):
                left = (self._local_rank - i + self._local_size) % self._local_size
                right = (self._local_rank + i) % self._local_size
               
                if not self._cio or (self._cio and right // self._local_size == node_rank):
                    if self._local_rank < i:
                        if forward:
                            r1 = dist.isend(send_gpu[self._boundary[right]] , tag=tag, dst=self._rank_start_id + right, group=self._intra_group)
                        else:
                            
                            r1 = dist.isend(send_gpu[self._pl[right]:self._pr[right]], tag=tag, dst=self._rank_start_id + right, group=self._intra_group)
                        req1.append(r1)

                    
                    r2 = dist.irecv(recv_gpu[left], tag=tag, src=self._rank_start_id + left, group=self._intra_group)
                    req2.append(r2)
                
                    if self._local_rank >= i:
                        if forward:
                            r1 = dist.isend(send_gpu[self._boundary[right]] , tag=tag, dst=self._rank_start_id + right, group=self._intra_group)
                        else:
                            
                            r1 = dist.isend(send_gpu[self._pl[right]:self._pr[right]], tag=tag, dst=self._rank_start_id + right, group=self._intra_group)
                        req1.append(r1)


            """
            Please refer to pytorch issue 
            """

            i = 0
            while len(req2) != 0:
                r = req2[i]
                r.wait()
                if r.is_completed():
                    del req2[i]
                else:
                    i += 1
                if i >= len(req2):
                    i = 0

            i = 0
            while len(req1) != 0:
                r = req1[i]
                r.wait()
                if r.is_completed():
                    del req1[i]
                else:
                    i += 1
                if i >= len(req1):
                    i = 0
            
            

    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, corr, avg=None, forward=True):
        
        req1, req2 = [], queue.Queue()
        self._comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            for i in range(1, self._local_size):
                left = (self._local_rank - i + self._local_size) % self._local_size
                right = (self._local_rank + i) % self._local_size
                r2 = dist.irecv(recv_cpu[left], tag=tag, src=self._rank_start_id + left)
                req2.put((r2, left))
                if forward:
                    send_cpu[right].copy_(send_gpu[self._boundary[right]])
                else:
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
                r1 = dist.isend(send_cpu[right], tag=tag, dst=self._rank_start_id + right)
                req1.append(r1)
            while not req2.empty():
                r, idx = req2.get()
                
                r.wait()
                recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
                if corr:
                    with torch.cuda.stream(self._corr_stream):
                        self._corr_stream.wait_stream(self._comm_stream)
                        t = avg[idx]
                        t *= self._corr_momentum
                        t += (1 - self._corr_momentum) * recv_gpu[idx]
            
            for r in req1:
                r.wait()

    def __feat_transfer(self, epoch, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                                   tag, self._corr_feat, self._f_avg[layer], forward=True)
            self._f_cuda_event[layer].record(self._comm_stream)
            if self._corr_feat:
                self._f_cuda_event[layer].record(self._corr_stream)
        else:
            self.__nccl_all_to_all(feat, self._f_recv[layer],
                                   tag, self._corr_feat, self._f_avg[layer], forward=True)
        self._f_cuda_event[layer].record(self._comm_stream)
    
    def __update_grad(self, layer, grad):
        
        for i in range(self._local_size):
            if i == self._local_rank:
                continue
            else:
                if self._corr_grad:
                    grad[self._boundary[i]] += self._b_avg[layer][i]
                else:
                    grad[self._boundary[i]] += self._b_recv[layer][i]

    def __grad_hook(self, epoch, layer):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    self.__grad_transfer(epoch, layer, grad)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                    if self._debug:
                        torch.cuda.synchronize()
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        self._b_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                        self._b_cpu_event[layer].clear()
                        torch.cuda.synchronize()
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad))
                return grad
        return fn

    def __grad_transfer(self, epoch, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                                   tag, self._corr_grad, self._b_avg[layer], forward=False)
        elif self._backend == 'nccl':
            
            self.__nccl_all_to_all(grad, self._b_recv[layer],
                                   tag, self._corr_grad, self._b_avg[layer], forward=False, layer=layer)
        else:
            raise NotImplementedError
        self._b_cuda_event[layer].record(self._comm_stream)
        if self._corr_grad:
            self._b_cuda_event[layer].record(self._corr_stream)

        self._b_cpu_event[layer].set()

class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._pipeline = False
        self._epoch = 0
        self._feat_cpu, self._grad_cpu = [], []
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        self._f_recv_cpu, self._b_recv_cpu = [], []
        self._f_avg, self._b_avg = [], []
        self._recv_shape = []
        self._pool = None
        self._comm_stream, self._corr_stream = None, None
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        self._corr_momentum = 0
        self._corr_feat, self._corr_grad = False, False
        self._pl, self._pr = [], []

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, model, use_pp=False, backend='gloo',
                    pipeline=False, corr_feat=False, corr_grad=False, corr_momentum=0, debug=False, check_intra_only= False):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._model = model
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._pipeline = pipeline
        self._epoch = 0
        self._recv_shape = f_recv_shape

        self._debug = debug
        self._cio = check_intra_only

        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3
                if i > 0:
                    self._grad_cpu[i] = tmp2
                    self._b_recv_cpu[i] = tmp4
        elif backend == 'nccl':
            pass
        else:
            raise NotImplementedError

        self._f_buf = [None] * self._n_layers
        self._f_recv, self._b_recv = [], []
        self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []

        self._backend = backend

        self._f_avg, self._b_avg = [None] * self._n_layers, [None] * self._n_layers
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers

        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)
                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
                    tmp3.append(torch.zeros(s1, device='cuda'))
                    tmp4.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2
            if corr_feat:
                self._f_avg[i] = tmp3
            if corr_grad and i > 0:
                self._b_avg[i] = tmp4
            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()
        self._corr_momentum = corr_momentum
        self._corr_feat, self._corr_grad = corr_feat, corr_grad
        self._pool = ThreadPool(processes=2*self._n_layers)
        self.__init_pl_pr()

    def next_epoch(self):
        self._epoch += 1

    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                if self._corr_feat:
                    tmp.append(self._f_avg[layer][i])
                else:
                    tmp.append(self._f_recv[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        with comm_timer.timer(f'forward_{layer}'):
            self.__feat_transfer(self._epoch, layer, feat)
            torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
            if self._debug == True:
                torch.cuda.synchronize()

        self._f_buf[layer] = self.__feat_concat(layer, feat)
        if self._f_buf[layer].requires_grad:
            """
            DeepGCN enables requires_grad because of node encoder !
            : in detail, it should be requirese_grad because
              linear layer should also be updated.
              Therefore, we just skip hook register here!
            """
            if not (self._model in ['deepgcn'] and layer == 0):
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))

        return self._f_buf[layer]

        """
        if self._pipeline is False:
        else:
            if self._epoch > 0:
                with comm_timer.timer(f'forward_{layer}'):
                    self._f_cpu_event[layer].wait()
                    torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                    self._f_cpu_event[layer].clear()
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, layer, feat))
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]
        """
    
    
    def __nccl_all_to_all(self, send_gpu, recv_gpu, tag, corr, avg=None, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        
        local_size = int( torch.cuda.device_count() )
        cur_server = int(rank / local_size)

        req1, req2 = [], queue.Queue()
        self._comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            for i in range(1, size):
                left = (rank - i + size) % size
                right = (rank + i) % size
                
                if not self._cio or (self._cio and int(right / local_size ) == cur_server):
                    if rank < i:
                        if forward:
                            r1 = dist.isend(send_gpu[self._boundary[right]] , tag=tag, dst=right)
                        else:
                            r1 = dist.isend(send_gpu[self._pl[right]:self._pr[right]], tag=tag, dst=right)
                        req1.append(r1)
                else:
                    pass
                
                if not self._cio or (self._cio and int(left / local_size ) == cur_server):
                    r2 = dist.irecv(recv_gpu[left], tag=tag, src=left)
                    req2.put((r2, left))
                else:
                    pass

                if not self._cio or (self._cio and int(right / local_size ) == cur_server):
                    if rank >= i:
                        if forward:
                            r1 = dist.isend(send_gpu[self._boundary[right]] , tag=tag, dst=right)
                        else:
                            r1 = dist.isend(send_gpu[self._pl[right]:self._pr[right]], tag=tag, dst=right)
                        req1.append(r1)
                else:
                    pass
                    
            while not req2.empty():
                r, idx = req2.get()
                
                r.wait()
                

                if corr:
                    with torch.cuda.stream(self._corr_stream):
                        self._corr_stream.wait_stream(self._comm_stream)
                        t = avg[idx]
                        t *= self._corr_momentum
                        t += (1 - self._corr_momentum) * recv_gpu[idx]
            
            for r in req1:
                r.wait()


    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, corr, avg=None, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()

        self._comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            for i in range(1, size):
                left = (rank - i + size) % size
                right = (rank + i) % size

                r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                req2.put((r2, left))

                if forward:
                    send_cpu[right].copy_(send_gpu[self._boundary[right]])
                else:
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])

                r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                req1.append(r1)

            while not req2.empty():
                r, idx = req2.get()
                
                r.wait()
                recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
                if corr:
                    with torch.cuda.stream(self._corr_stream):
                        self._corr_stream.wait_stream(self._comm_stream)
                        t = avg[idx]
                        t *= self._corr_momentum
                        t += (1 - self._corr_momentum) * recv_gpu[idx]
            
            for r in req1:
                r.wait()

    def __feat_transfer(self, epoch, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                                   tag, self._corr_feat, self._f_avg[layer], forward=True)
        elif self._backend == 'nccl':
            self.__nccl_all_to_all(feat, self._f_recv[layer],
                                   tag, self._corr_feat, self._f_avg[layer], forward=True)
        else:
            raise NotImplementedError
        self._f_cuda_event[layer].record(self._comm_stream)
        if self._corr_feat:
            self._f_cuda_event[layer].record(self._corr_stream)

        self._f_cpu_event[layer].set()

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                if self._corr_grad:
                    grad[self._boundary[i]] += self._b_avg[layer][i]
                else:
                    grad[self._boundary[i]] += self._b_recv[layer][i]

    def __grad_hook(self, epoch, layer):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    self.__grad_transfer(epoch, layer, grad)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                    if self._debug == True:
                        torch.cuda.synchronize()

                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        self._b_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                        self._b_cpu_event[layer].clear()
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad))
                return grad
        return fn

    def __grad_transfer(self, epoch, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                                   tag, self._corr_grad, self._b_avg[layer], forward=False)
        elif self._backend == 'nccl':
            self.__nccl_all_to_all(grad, self._b_recv[layer],
                                   tag, self._corr_grad, self._b_avg[layer], forward=False)
        else:
            raise NotImplementedError
        self._b_cuda_event[layer].record(self._comm_stream)
        if self._corr_grad:
            self._b_cuda_event[layer].record(self._corr_stream)

        self._b_cpu_event[layer].set()
