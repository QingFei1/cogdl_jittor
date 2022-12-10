import random
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import jittor
from cogdl.operators.sample import coo2csr_cpu, coo2csr_cpu_index


def get_degrees(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    b = jittor.ones(col.shape[0])
    out = jittor.zeros(num_nodes)
    degrees = out.scatter_(dim=0, index=row, src=b,reduce='add')
    return degrees.float()


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    row, col = edge_index
    if edge_weight is None:
        edge_weight = jittor.ones(edge_index[0].shape[0])
    if num_nodes is None:
        num_nodes = jittor.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    self_weight = jittor.full((num_nodes,), fill_value, dtype=edge_weight.dtype)
    loop_index = jittor.arange(0, N, dtype=row.dtype)
    row = jittor.concat([row, loop_index])
    col = jittor.concat([col, loop_index])
    edge_index = jittor.stack([row, col])
    edge_weight = jittor.concat([edge_weight, self_weight])
    return edge_index, edge_weight


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    row, col = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = jittor.ones(row.shape[0])
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    mask = row != col

    loop_index = jittor.arange(0, N, dtype=row.dtype)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    _row = jittor.concat([row[mask], loop_index[0]])
    _col = jittor.concat([col[mask], loop_index[1]])
    # edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    inv_mask = jittor.logical_not(mask)

    loop_weight = jittor.full((N,), fill_value, dtype=edge_weight.dtype)
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = jittor.concat([edge_weight[mask], loop_weight], dim=0)

    return (_row, _col), edge_weight


def row_normalization(num_nodes, row, col, val=None):
    if val is None:
        val = jittor.ones(row.shape[0])
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv = row_sum.pow(-1).view(-1)
    row_sum_inv[jittor.isinf(row_sum_inv)] = 0
    return val * row_sum_inv[row]


def symmetric_normalization(num_nodes, row, col, val=None):
    if val is None:
        val = jittor.ones(row.shape[0])
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float("inf")] = 0
    return row_sum_inv_sqrt[col] * val * row_sum_inv_sqrt[row]


def _coo2csr(edge_index, data, num_nodes=None, ordered=False, return_index=False):
    if ordered:
        return sorted_coo2csr(edge_index[0], edge_index[1], data, return_index=return_index)
    if num_nodes is None:
        num_nodes = jittor.max(edge_index) + 1
    sorted_index = jittor.argsort(edge_index[0])[0]
    sorted_index = sorted_index.long()
    edge_index = edge_index[:, sorted_index]
    indices = edge_index[1]

    row = edge_index[0]
    indptr = jittor.zeros(num_nodes + 1, dtype=jittor.int32)
    elements, counts = jittor.unique(row, return_counts=True)
    elements = elements.long() + 1
    indptr[elements] = counts.to(indptr.dtype)
    indptr = indptr.cumsum(dim=0)

    if return_index:
        return indptr, sorted_index
    if data is not None:
        data = data[sorted_index]
    return indptr, indices, data


def coo2csr(row, col, data, num_nodes=None, ordered=False):
    if ordered:
        indptr, indices, data = sorted_coo2csr(row, col, data)
        return indptr, indices, data
    if num_nodes is None:
        num_nodes = jittor.max(jittor.stack(row, col)).item() + 1
    if coo2csr_cpu is None:
        return _coo2csr(jittor.stack([row, col]), data, num_nodes)
    row = row.long()
    col = col.long()
    data = data.float()
    indptr, indices, data = coo2csr_cpu(row, col, data, num_nodes)
    return indptr, indices, data


def coo2csr_index(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = jittor.max(jittor.stack([row, col])).item() + 1
    if coo2csr_cpu_index is None:
        return _coo2csr(jittor.stack([row, col]), None, num_nodes=num_nodes, return_index=True)
    row = row.long()
    col = col.long()
    # indptr, reindex = coo2csr_cpu_index(row, col, num_nodes)
    print("-------graph_utils_135用的是torch的sample的coo2csr--------")
    row=torch.from_numpy(row.numpy()).long()
    col=torch.from_numpy(col.numpy()).long()
    num_nodes=torch.tensor(num_nodes).long()
    indptr, reindex = coo2csr_cpu_index(row, col, num_nodes)  #这里用的是torch的sample
    indptr=jittor.array(indptr.numpy())
    reindex=jittor.array(reindex.numpy())
    print("-------#这里用的是torch的sample的coo2csr--------")
    return indptr, reindex


def sorted_coo2csr(row, col, data, num_nodes=None, return_index=False):
    indptr = jittor.array(np.bincount(row.numpy())) #jittor无bincount
    indptr = jittor.misc.cumsum(indptr,dim=0) #替换
    zero = jittor.zeros(1)
    indptr = jittor.concat([zero, indptr])
    if return_index:
        return indptr, jittor.arange(0, row.shape[0])
    return indptr, col, data


def coo2csc(row, col, data, num_nodes=None, sorted=False):
    return coo2csr(col, row, data, num_nodes, sorted)


def csr2csc(indptr, indices, data=None):
    indptr = indptr.numpy()
    indices = indices.numpy()
    num_nodes = indptr.shape[0] - 1
    if data is None:
        data = np.ones(indices.shape[0])
    else:
        data = data.numpy()
    adj = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    adj = adj.tocsc()
    data = jittor.array(adj.data)
    col_indptr = jittor.array(adj.indptr)
    row_indices = jittor.array(adj.indices)
    return col_indptr, row_indices, data


def csr2coo(indptr, indices, data):
    num_nodes = indptr.size(0) - 1
    row = jittor.arange(num_nodes).numpy()
    repeat=(indptr[1:] - indptr[:-1]).numpy()
    #row = row.repeat_interleave(row_count)
    row = jittor.array([item for n,s in zip(repeat, row) for item in [s]*n])
    return row, indices, data


def remove_self_loops(indices, values=None):
    row, col = indices
    mask = indices[0] != indices[1]
    row = row[mask]
    col = col[mask]
    if values is not None:
        values = values[mask]
    return (row, col), values


def coalesce(row, col, value=None):
    if isinstance(row,jittor.Var):
        row = row.numpy()
    if isinstance(col,jittor.Var):
        col = col.numpy()
    indices = np.lexsort((col, row))
    row = jittor.array(row[indices]).int64()
    col = jittor.array(col[indices]).int64()

    num = col.shape[0] + 1
    idx = jittor.full((num,), -1).int64()
    max_num = max(row.max(), col.max()) + 100
    idx[1:] = (row + 1) * max_num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col, value
    row = row[mask]
    if value is not None:
        _value = jittor.zeros(row.shape[0], dtype=jittor.float)
        value = _value.scatter_(dim=0, src=value, index=col,reduce='add')
    col = col[mask]
    return row, col, value


def to_undirected(edge_index, num_nodes=None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """

    row, col = edge_index
    row, col = jittor.concat([row, col], dim=0), jittor.concat([col, row], dim=0)
    row, col, _ = coalesce(row, col, None)
    edge_index = jittor.stack([row, col])
    return edge_index


def negative_edge_sampling(
    edge_index: Union[Tuple, jittor.Var],
    num_nodes: Optional[int] = None,
    num_neg_samples: Optional[int] = None,
    undirected: bool = False,
):
    if num_nodes is None:
        num_nodes = len(jittor.unique(edge_index))
    if num_neg_samples is None:
        num_neg_samples = edge_index[0].shape[0]

    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index[1].shape[0])

    row, col = edge_index
    unique_pair = row * num_nodes + col

    num_samples = int(num_neg_samples * abs(1 / (1 - 1.1 * row.size(0) / size)))
    sample_result = jittor.array(random.sample(range(size), min(num_samples, num_samples))).long()
    mask = jittor.array(np.isin(sample_result, unique_pair)).bool()
    selected = sample_result[~mask][:num_neg_samples]

    row = selected // num_nodes
    col = selected % num_nodes
    return jittor.stack([row, col]).long()
