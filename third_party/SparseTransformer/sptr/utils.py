import numbers
import torch
import numpy as np
from torch_scatter import segment_csr, gather_csr
from torch_geometric.nn import voxel_grid
from . import precompute_all


def to_3d_numpy(size):
    if isinstance(size, numbers.Number):
        size = np.array([size, size, size]).astype(np.float32)
    elif isinstance(size, list):
        size = np.array(size)
    elif isinstance(size, np.ndarray):
        size = size
    else:
        raise ValueError("size is either a number, or a list, or a np.ndarray")
    return size

def grid_sample(pos, batch, size, start, return_p2v=True, return_counts=True, return_unique=False):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None

    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False and return_counts == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    if return_p2v == False and return_counts == True:
        return cluster, counts.max().item(), counts

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)

    if return_unique:
        return cluster, p2v_map, counts, unique

    return cluster, p2v_map, counts

def point_partition(xyz, window_size, radial_partition, delta, batch = None, debug=False):
        indices1 = xyz[:, 2] <= radial_partition[0]
        indices2 = (xyz[:, 2] > radial_partition[0]) & (xyz[:, 2] <= radial_partition[1])
        indices3 = xyz[:, 2] > radial_partition[1]

        if debug:
            print(window_size)
        window_size1 = window_size.clone().detach()
        window_size1[:-1] -= delta
        window_size2 = window_size.clone().detach()
        window_size3 = window_size.clone().detach()
        window_size3[:-1] += delta  

        window_size1, window_size2, window_size3 =  window_size1.to(xyz.device), \
                                                    window_size2.to(xyz.device), \
                                                    window_size3.to(xyz.device)
        if debug:
            print(f"window size1 {window_size1}, window size2 {window_size2}, window size3 {window_size3}")

        xyz1 = xyz[indices1]
        xyz2 = xyz[indices2]
        xyz3 = xyz[indices3]

        if batch is not None:
            batch1 = batch[indices1]
            batch2 = batch[indices2]
            batch3 = batch[indices3]

        if batch is not None:
            return xyz1, xyz2, xyz3, batch1, batch2, batch3, window_size1, window_size2, window_size3
        else:
            return xyz1, xyz2, xyz3, window_size1, window_size2, window_size3

def get_indices_params(xyz, batch, window_size, shift_win: bool, radial_partition = None, delta = None, debug:bool=False):
    
    if isinstance(window_size, list) or isinstance(window_size, np.ndarray):
        window_size = torch.from_numpy(window_size).type_as(xyz).to(xyz.device)
    else:
        window_size = torch.tensor([window_size]*3).type_as(xyz).to(xyz.device)
    
    if radial_partition is not None:
        xyz1, xyz2, xyz3, batch1, batch2, batch3, window_size1, window_size2, window_size3 = point_partition(xyz, window_size, radial_partition, delta, batch, debug=debug)

        counts1, counts2, counts3 = torch.empty(0, dtype=torch.int32).to(xyz.device), \
                                    torch.empty(0, dtype=torch.int32).to(xyz.device), \
                                    torch.empty(0, dtype=torch.int32).to(xyz.device)
        v2p_map1, v2p_map2, v2p_map3 = torch.empty(0, dtype=torch.int32).to(xyz.device), \
                                       torch.empty(0, dtype=torch.int32).to(xyz.device), \
                                       torch.empty(0).to(xyz.device)
        k1, k2, k3 = 0, 0, 0
        if xyz1.numel() != 0:
            v2p_map1, k1, counts1 = grid_sample(xyz1, batch1, window_size1, start=None, return_p2v=False, return_counts=True)
        if xyz2.numel() != 0:
            v2p_map2, k2, counts2 = grid_sample(xyz2, batch2, window_size2, start=None, return_p2v=False, return_counts=True)
        if xyz3.numel() != 0:
            v2p_map3, k3, counts3 = grid_sample(xyz3, batch3, window_size3, start=None, return_p2v=False, return_counts=True)

        if debug: 
            print(f'xyz1 {xyz1}')
            print(f'v2p_map1 {v2p_map1}, k1 {k1}, counts1 {counts1} tpye {counts1.dtype}\n\n')
            print(f'xyz2 {xyz2}')
            print(f'v2p_map2 {v2p_map2}, k2 {k2}, counts2 {counts2} type {counts2.dtype}\n\n')
            print(f'xyz3 {xyz3}')
            print(f'v2p_map3 {v2p_map3}, k3 {k3}, counts3 {counts3} tpye {counts3.dtype}\n\n')
        map2_offset, map3_offset = len(counts1), len(counts1) + len(counts2)
        v2p_map2, v2p_map3 = v2p_map2 + map2_offset, v2p_map3 + map3_offset

        if debug:
            print(f'After v2p_map3 {v2p_map3}, k3 {k3}, counts3 {map3_offset}\n\n')
            print(f'After v2p_map3 {v2p_map2}, k3 {k2}, counts3 {map2_offset}\n\n')
        counts = torch.cat((counts1, counts2, counts3), dim=0)
        v2p_map = torch.cat((v2p_map1, v2p_map2, v2p_map3), dim=0)
        k = max(k1, k2, k3)

        if debug: 
            print(f'v2p_map {v2p_map}, k {k}, counts {counts}\n\n')
    elif shift_win:
        v2p_map, k, counts = grid_sample(xyz+1/2*window_size, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    else:
        v2p_map, k, counts = grid_sample(xyz, batch, window_size, start=None, return_p2v=False, return_counts=True)

    v2p_map, sort_idx = v2p_map.sort()
    
    n = counts.shape[0]
    N = v2p_map.shape[0]

    if debug:
        print(f"Points , clusters {N}, {n}")
        print(f"v2pshape, counts, sort_idx - {v2p_map}, {counts}, {sort_idx}")

    n_max = k
    index_0_offsets, index_1_offsets, index_0, index_1 = precompute_all(N, n, n_max, counts)
    index_0 = index_0.long()
    index_1 = index_1.long()

    return index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx

def scatter_softmax_csr(src: torch.Tensor, indptr: torch.Tensor, dim: int = -1):
    ''' src: (N, C),
        index: (Ni+1, ), [0, n0^2, n0^2+n1^2, ...]
    '''
    max_value_per_index = segment_csr(src, indptr, reduce='max')
    max_per_src_element = gather_csr(max_value_per_index, indptr)
    
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = segment_csr(
        recentered_scores_exp, indptr, reduce='sum')
    
    normalizing_constants = gather_csr(sum_per_index, indptr)

    return recentered_scores_exp.div(normalizing_constants)
