from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import dgl

if TYPE_CHECKING:
    from helper.parser import Args

class Mapper:
    """Original full-graph id <-> Server-wise macrobatch graph id <-> GPU-wise minibatch graph id
    Once you register the mapping, you can find full_id or gpu_id in O(1) time.

    This scheme has a total of 5 id types - full-id, server-id, psuedo-server-id, gpu-id, and remapped gpu-id.
    I'll refer full-id as fid, server-id and pseudo-server-id as sid, gpu-id and remapped-gpu-id as gid.
    When a sampler samples the full-graph, then full-id is translated into server-id.
    When the server-graph partitioned by METIS, first server-id is translated into psuedo-server-id,
    then the pseudo-server-graph is partitioned into many gpu-graphs.
    Finally, when the gpu-graph is transformed into the intra_graph, the gpu-graph remaps itself.

    To communicate which vertex should be shared(to implement historical embeddings), we have to know the full-ids of
    given boundary nodes, then send/recv to/from corresponding rank,
    then revert the full-ids into its corresponding gpu's (remapped)gpu_ids.
    """
    def __init__(self, args: 'Args') -> None:
        self._full_graph_size: int = -1
        self.node_rank = args.node_rank
        self.local_rank = args.local_rank
        self.parts_per_node = args.parts_per_node
        
        self.args = args

        self._sid_from_fid: list[torch.Tensor] = [None] * args.epoch_iter
        self._fid_from_sid: list[torch.Tensor] = [None] * args.epoch_iter 
        self._gid_from_fid: list[torch.Tensor] = [None] * args.epoch_iter
        self._fid_from_gid: list[torch.Tensor] = [None] * args.epoch_iter

        self._rank_from_fid: torch.Tensor = None

        if __debug__:
            if self.local_rank == 0:
                self.seeds: list[torch.Tensor]


    def synchronize(self):
        """Broadcast 0-th process's full graph size, server_from_full, full_from_server.
        """
        args = self.args
        full_graph_size = torch.tensor(self._full_graph_size, device='cuda')
        dist.broadcast(full_graph_size, src=0, group=args.device_group)
        self._full_graph_size = full_graph_size.item()
        self._rank_from_fid = torch.zeros(self._full_graph_size, dtype=torch.int32, device='cuda') - 1  

        for iter in range(args.epoch_iter):
            sid_from_fid_size = torch.tensor(self._sid_from_fid[iter].shape[0] if self.local_rank == 0 else -1, device='cuda')
            dist.broadcast(sid_from_fid_size, src=0, group=args.device_group)
            print(self.local_rank, sid_from_fid_size)
            if self.local_rank != 0:
                self._sid_from_fid[iter] = torch.empty(sid_from_fid_size, dtype=torch.int32, device='cuda')
            print(self.local_rank, self._sid_from_fid[iter].shape, self._sid_from_fid[iter])
            dist.broadcast(self._sid_from_fid[iter], src=0, group=args.device_group)
            dist.barrier(args.device_group)

            fid_from_sid_size = torch.tensor(self._fid_from_sid[iter].shape[0] if self.local_rank == 0 else -1, device='cuda')
            dist.broadcast(fid_from_sid_size, src=0, group=args.device_group)
            print(self.local_rank, fid_from_sid_size)
            if self.local_rank != 0:
                self._fid_from_sid[iter] = torch.empty(fid_from_sid_size, dtype=torch.int32, device='cuda')
            print(self.local_rank, self._fid_from_sid[iter].shape, self._fid_from_sid[iter])
            dist.broadcast(self._fid_from_sid[iter], src=0, group=args.device_group)
            dist.barrier(args.device_group)


    def register_full_graph_size(self, size: int):
        self._full_graph_size = size


    def register_fid_sid(self, iter: int, fid_from_sid: torch.Tensor, sid_from_psid: torch.Tensor) -> None:
        """Register mapping between full_ids and server_ids. This roughly takes O(V) time.

        Parameters
        ----------
        iter : int
            Current iter value.
        fid_from_sid : Tensor
            Ids of the original full-graph. You can find this tensor by g.ndata[dgl.NID].
        sid_from_psid : Tensor
        """
        fid_from_sid = fid_from_sid.to(torch.int32)
        fid_from_psid = _chain(fid_from_sid, sid_from_psid)
        self._fid_from_sid[iter] = fid_from_psid
        self._sid_from_fid[iter] = _invert(fid_from_psid)

        self._fid_from_sid[iter] = self._fid_from_sid[iter].cuda()
        self._sid_from_fid[iter] = self._sid_from_fid[iter].cuda()


    def register_sid_gid(self, iter: int, node_dict: dict[str, torch.Tensor]) -> None:
        """Register mapping between server_ids and current gpu's gpu_ids.
        Before running this method, please first register a mapping between full_ids and server_ids.

        Parameters
        ----------
        iter : int
            Current iter value.
        node_dict : dict[str, Tensor]
        """
        assert self._sid_from_fid[iter] is not None
        assert self._fid_from_sid[iter] is not None

        sid_from_gid = node_dict[dgl.NID]
        self._fid_from_gid[iter] = _chain(self._fid_from_sid[iter], sid_from_gid)

        gid_from_sid = _invert(sid_from_gid)
        self._gid_from_fid[iter] = _chain(gid_from_sid, self._sid_from_fid[iter], naive=True)

        self._fid_from_gid[iter] = self._fid_from_gid[iter].cuda()
        self._gid_from_fid[iter] = self._gid_from_fid[iter].cuda()


    def register_and_share_core_nodes(self, iter: int, seed_node: torch.Tensor) -> None:
        """Register core nodes and all-gather to all workers (including other server's gpus).
        GPU's core nodes are (seed nodes of the GPU's partition) ∩ (inner nodes of the GPU).
        In other words, every full nodes are uniquely assigned as core to each GPUs.
        """
        args = self.args
        rank, size = dist.get_rank(), dist.get_world_size()
        
        core_gids = seed_node.nonzero().flatten()
        core_fids = self._fid_from_gid[iter][core_gids]
        assert torch.all(core_fids != -1), f"{sum(core_fids == -1)=}"  
        core_nodes_num = torch.tensor(len(core_fids), dtype=torch.int32, device='cuda')

        core_nodes_num_list = [torch.empty(1, dtype=torch.int32, device='cuda') for _ in range(size)]
        dist.all_gather(core_nodes_num_list, core_nodes_num)
        dist.barrier()

        core_full_ids_list = [torch.empty(core_nodes_num_list[r], dtype=torch.int32, device='cuda') for r in range(size)]
        dist.all_gather(core_full_ids_list, core_fids)
        dist.barrier()
        

        if __debug__:
            print(self.local_rank, f"{core_nodes_num=}", f"{core_fids=}")
            

        if self.local_rank == 0:
            assert self.seeds[iter] == set(torch.cat(core_full_ids_list).tolist()), f"{len(self.seeds[iter] & set(torch.cat(core_full_ids_list).tolist()))}"  

        for r in range(size):
            assert torch.all(self._rank_from_fid[core_full_ids_list[r]] == -1), f"{r} {sum(self._rank_from_fid[core_full_ids_list[r]] != -1)=}"  
            self._rank_from_fid[core_full_ids_list[r]] = r

        if self.local_rank == 0 and iter == args.epoch_iter - 1:
            assert sum(self._rank_from_fid != -1) == self.train_size, f"{sum(self._rank_from_fid != -1)=}, {self.train_size=}"  


    def remap_gpu_and_gpu(self, iter: int, orig_from_new: torch.Tensor) -> None:
        assert self._gid_from_fid[iter] is not None
        assert self._fid_from_gid[iter] is not None

        self._fid_from_gid[iter] = _chain(self._fid_from_gid[iter], orig_from_new)

        new_from_orig = _invert(orig_from_new)
        self._gid_from_fid[iter] = _chain(new_from_orig, self._gid_from_fid[iter], naive=True)


    def get_full_ids(self, iter: int, gpu_ids: torch.Tensor) -> torch.Tensor:
        """Translates gpu_ids into full_ids.
        This method takes O(1) time.
        Make sure to register and remap the mapping between full_ids and gpu_ids.

        Parameters
        ----------
        iter : int
            Current iter value.
        gpu_ids : int | list[int] | IntTensor

        Returns
        -------
        full_ids : Tensor
        """
        return self._fid_from_gid[iter][gpu_ids]


    def get_gpu_ids(self, iter: int, full_ids: torch.Tensor):
        """Translates full_ids into gpu_ids.
        This method takes O(1) time.
        Translating non-gpu full_ids into gpu_ids is undefined behavior (this could fire IndexError or return -1 tensor).
        So make sure full_ids are valid ids in the current gpu.
        Also make sure to register and remap the mapping between full_ids and gpu_ids.

        Parameters
        ----------
        iter : int
            Current iter value.
        full_ids : int | list[int] | IntTensor
            full_ids must be valid ids in the current gpu.

        Returns
        -------
        gpu_ids : Tensor
        """
        return self._gid_from_fid[iter][full_ids]


    def get_rank(self, full_ids: torch.Tensor):
        return self._rank_from_fid[full_ids]


def _chain(c_from_b: torch.Tensor, b_from_a: torch.Tensor, naive: bool = False) -> torch.Tensor:
    """Basically same as c_from_b[b_from_a], but this method ignores -1 from b_from_a.

    Parameters
    ----------
    c_from_b : Tensor
    b_from_a : Tensor
    naive : bool
        If naive is true, then error does not occur when b_from_a is out of bounds from c_from_b.
        This option is used when the domain of b_from_a is bigger than the domain of c_from_b.

    Returns
    -------
    c_from_a : Tensor

    Examples
    --------
    When |B| > |A|: (ex: full <- server <- gpu)
    
    >>> c_b = torch.tensor([3, 2, 5, 4, 0])
    >>> b_a = torch.tensor([1, 3, 0])
    >>> c_a = _chain(c_b, b_a)
    >>> c_a
    tensor([2, 4, 3])

    When |B| < |A|: (like gpu <- server <- full)
    
    >>> c_b = torch.tensor([1, 3, 0])
    >>> b_a = torch.tensor([3, 2, -1, 4, 0])
    >>> c_a = _chain(c_b, b_a, naive=True)
    >>> c_a
    tensor([-1,  0, -1, -1,  1])
    
    In this case, c_a[0] is undefined because b_a[0] is 3 but 3 is not in the domain of c_b.
    c_a[1] is 0 because b_a[1] is 2 and c_b[2] is 0.
    c_a[2] is undefined because b_a[2] is undefined.
    """
    len_a = len(b_from_a)
    dtype = c_from_b.dtype
    c_from_a = torch.zeros(len_a, dtype=dtype, device=c_from_b.device) - 1  

    b_mask = b_from_a != -1  
    if naive:
        safe_b_mask = b_from_a < len(c_from_b)
        b_mask &= safe_b_mask
    bs = b_from_a[b_mask]  
    c_from_a[b_mask] = c_from_b[bs]

    """Above codes are equivalent to:
    for a, b in enumerate(b_from_a):
        if b != -1 (and b < len(c_from_b)):
            c_from_a[a] = c_from_b[b]
    """
    
    
    
    
    
    
    
    
    return c_from_a


def _map_one_by_one(in_ids: torch.Tensor, out_ids: torch.Tensor) -> torch.Tensor:
    """Returns mapper tensor that behaves like mapper[in_ids] = out_ids.
    If in_ids has a hole(e.g. [0, 1, 2, 4, 5]), then out_id corresponds to the hole(e.g. mapper[3]) becomes -1.

    Parameters
    ----------
    in_ids : Tensor
        Each ids must be unique.
    out_ids : Tensor
        Each ids must be unique.

    Returns
    -------
    mapper : Tensor

    Examples
    --------
    >>> t = torch.tensor([3, 2, 6, 4, 0])
    >>> s = torch.tensor([4, 2, 1, 0, 5])
    >>> mapper = _map_one_by_one(t, s)
    >>> mapper
    tensor([ 5, -1,  2,  4,  0, -1,  1])
    >>> mapper[t]
    tensor([4, 2, 1, 0, 5])
    """
    assert in_ids.ndim == out_ids.ndim == 1
    assert in_ids.dtype == out_ids.dtype
    assert len(in_ids) == len(out_ids)
    l = in_ids.max().item() + 1
    dtype = out_ids.dtype
    mapper = torch.zeros(l, dtype=dtype, device=in_ids.device) - 1  
    for i, o in zip(in_ids, out_ids):
        if i != -1:
            mapper[i] = o
    return mapper


def _invert(mapper: torch.Tensor) -> torch.Tensor:
    """Returns inverted mapper.

    Parameters
    ----------
    mapper : Tensor
        Each ids must be unique.

    Returns
    -------
    mapper : Tensor

    Examples
    --------
    >>> t = torch.tensor([1, 3, 4, 0])
    >>> s = _invert(t)
    >>> s
    tensor([3, 0, -1, 1, 2])
    >>> _invert(s)
    tensor([1, 3, 4, 0])
    """
    in_ids = torch.arange(mapper.shape[0], dtype=mapper.dtype, device=mapper.device)
    return _map_one_by_one(mapper, in_ids)
