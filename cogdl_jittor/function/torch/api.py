import torch
import numpy as np

def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

def ones(shape, dtype=None, device=None):
    return torch.ones(shape, dtype=dtype, device=device)

def zeros(shape, dtype=None, device=None):
    return torch.zeros(shape, dtype=dtype, device=device)

def zeros_like(input, dtype=None, device=None):
    return torch.zeros_like(input, dtype=dtype, device=device)

def ones_like(input):
    return torch.ones_like(input)

def arange(start=0, end=None, step=1, dtype=None, device=None):
    if end is None:
        end,start = start,0
    return torch.arange(start, end, step, dtype=dtype, device=device)


def cat(input, dim=0):
    return torch.cat(input, dim)

def pow(x,y):
    return torch.pow(x,y)

def isinf(x):
    return torch.isinf(x)

def stack(x, dim=0):
    return torch.stack(x, dim=dim)

def where(input):  #torch.bool
    return torch.where(input)

def max(input, dim=0):
    # NOTE: return value,argmax array is not returned,torch.amx return value,index
    return torch.max(input, dim=dim)[0]

def argmax(input, dim=0):
    # return indices
    return torch.argmax(input, dim=dim)[0]

def tensor(input, dtype=None):
    return torch.tensor(input, dtype=dtype)

def full(shape, val, dtype=None, device=None):
    return torch.full(shape, val, dtype=dtype, device=device)

def rand(*size, dtype=None, requires_grad=True, device=None):
    return torch.rand(size, dtype=dtype, requires_grad=requires_grad, device=device)

def randint(low, high, shape, device=None):
    return torch.randint(low, high, shape, device=device)

def from_numpy(input):
    return torch.from_numpy(input)

def as_tensor(data,dtype=None, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def unique(input, dim=None, return_inverse=False, return_counts=False): 
    #return output，counts
    output, inverse, counts=torch.unique(input, dim=None, return_inverse=True, return_counts=True)
    if return_counts == True and return_inverse == False:
        return output, counts
    elif return_inverse == True and  return_counts == False:
        return output, inverse
    elif return_inverse == False and return_counts == False:
        return output
    else:
        return output, inverse, counts

def cpu(input):
    return input.cpu()

def to(input,to_device):
    #Update Device,tensor.to(device). to_device:data or data.device
    if torch.is_tensor(to_device):
        device=to_device.device
    else:
        device=to_device
    return input.to(device)

def device(input):
    return input.device

def dtype_dict(dtype):
    type={'float16' : torch.float16,
            'float' : torch.float32,
            'float32' : torch.float32,
            'float64' : torch.float64,
            'int8'    : torch.int8,
            'int16'   : torch.int16,
            'int32'   : torch.int32,
            'int64'   : torch.int64,
            'long'   : torch.int64,
            'bool'    : torch.bool,
            'tensor' : torch.Tensor}
    return(type[dtype])

def dim(input):
    return(input.dim())

def load(path):
    return torch.load(path)

def save(obj,path):
    return torch.save(obj,path)

def sort(data, dim=-1, descending=False):
    # return value,index
    return torch.sort(data, dim=dim, descending=descending)

def argsort(data, dim=-1, descending=False):
    # return index
    return torch.argsort(data, dim=dim, descending=descending)

def sum(input, dim, keepdims=False):
    return torch.sum(input, dim=dim, keepdim=keepdims)

def isnan(input):
    return torch.isnan(input)
    
def scatter_add_(x, dim, index, src, reduce=None):
    return x.scatter_add_(dim=dim,index=index,src=src)    

def bincount(input, weights=None, minlength=0):
    return torch.bincount(input, weights=weights, minlength=minlength)

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def cuda_is_available():
    return torch.cuda.is_available()

def index_select(x, dim, index):
    return x.index_select(dim=dim,index=index)

def type_as(input, type):
    return input.type_as(type)

def logical_not(input):
    return ~input

def squeeze(input, dim=None):
    return torch.squeeze(input,dim=dim)

def eq(input, other):
    return torch.eq(input, other)

def repeat_interleave(input, repeats, dim=None):
    output = torch.repeat_interleave(input, repeats=repeats, dim=dim)
    return output

if __name__ == '__main__':

    a=torch.tensor([1,4,9,2])
    b=torch.tensor([1,4],device="cuda")
    print(sort(a),sort(a)[0],unique(a, return_inverse=True, return_counts=True))
    print(to(a,"cuda"))