import jittor as jt
import numpy as np

def is_tensor(obj):    # r isinstance():
    return isinstance(obj, jt.Var)

def ones(shape, dtype="float32", device=None):
    if 	isinstance(shape, jt.Var):
        shape=shape.item()
    return jt.ones(shape, dtype=dtype)

def zeros(shape, dtype="float32", device=None):
    if 	isinstance(shape, jt.Var):
        shape=shape.item()
    return jt.zeros(shape, dtype=dtype)

def zeros_like(input, dtype=None, device=None):
    return jt.zeros_like(input, dtype=dtype)

def ones_like(input):
    if 	isinstance(shape, jt.Var):
        shape=shape.item()
    return jt.ones_like(input)


def arange(start=0, end=None, step=1, dtype=None, device=None):
    if end is None:
        end,start = start,0
    return jt.arange(start, end, step,  dtype)

def cat(input,dim=0):
    return jt.concat(input,dim)

def pow(x,y):
    return jt.pow(x,y)

def isinf(x):
    return x==float('inf')
    
def stack(x, dim=0):
    return jt.stack(x, dim=dim)

def where(input):  #jittor.bool
    return jt.where(input)

def max(input, dim=0):
    # return value
    return jt.max(input, dim=dim)

def argmax(input, dim=0):
    # NOTE: return indices
    return jt.argmax(input, dim=dim)[0]

def tensor(input, dtype=None):
    return jt.array(input, dtype=dtype)

def full(shape,val,dtype="float32", device=None):
    return jt.full(shape,val,dtype)

def rand(*size, dtype="float32", requires_grad=True, device=None):
    return jt.rand(*size, dtype="float32", requires_grad=True)

def randn(*size, dtype="float32", requires_grad=True, device=None):
    return jt.randn(*size, dtype="float32", requires_grad=True)

def randint(low, high, shape, device=None):
    return jt.randint(low, high, shape)

def from_numpy(input):
    if isinstance(input,np.matrix):
        input = input.getA()
    return jt.array(input)

def as_tensor(data,dtype=None, device=None):
    return jt.array(data, dtype=dtype)

def unique(input, dim= None, return_inverse=False, return_counts=False): 
    #retyrn output，counts
    output, inverse, counts=jt.unique(input, dim=None, return_inverse=True, return_counts=True)
    if return_counts == True and return_inverse == False:
        return output, counts
    elif return_inverse == True and  return_counts == False:
        return output, inverse
    elif return_inverse == False and return_counts == False:
        return output
    else:
        return output, inverse, counts

def cpu(input):
    return input

def to(input,to_device):
    return input

def device(input):
    if jt.flags.use_cuda ==0:
        return "cpu"
    else:
        return "cuda"

def is_list(input):
    return isinstance(input,list)

def is_tensor(obj):
    return isinstance(obj,jt.Var)

def is_number(input):
    return isinstance(input,int) or isinstance(input,float)

def is_string(input):
    return isinstance(input,str)

def is_number_tensor(input):
    return isinstance(input,jt.Tensor) and is_number(input.data)

def is_tensor_or_number(input):
    return is_tensor(input) or is_number(input)

def is_gpu(input):
    return input.is_cuda

def jt_is_gpu(input):
    return input.is_cuda

def zeros(*size, dtype="float32", device=None):
    return jt.zeros(*size, dtype="float32")

def dtype_dict(dtype):
    type={'float16' : "float16",
            'float' : "float32",
            'float32' : "float32",
            'float64' : "float64",
            'int8'    : "int8",
            'int16'   : "int16",
            'int32'   : "int32",
            'int64'   : "int64",
            'long'    : "int64",
            'bool'    : "bool",
            'tensor'  : jt.Var}
    return type[dtype]

def dim(input):
    return(len(input.shape))

def load(path):
    return jt.load(path)

def save(obj,path):
    return jt.save(obj,path)
  

def sort(data, dim=-1, descending=False):
    # return value, index
    index, value= jt.argsort(data, dim=dim, descending=descending)
    return value, index

def argsort(data, dim=-1, descending=False):
    # return index
    index, value= jt.argsort(data, dim=dim, descending=descending)
    return index

def sum(input, dim, keepdims=False):
    return jt.sum(input, dim=dim, keepdims=keepdims)

def isnan(input):
    return jt.isnan(input)

def scatter_add_(x, dim, index, src, reduce='add'):
    return x.scatter_(dim=dim,index=index,src=src,reduce=reduce)

def bincount(input, weights=None, minlength=0):
    if isinstance(weights, jt.Var):
        weights=weights.numpy()
    return jt.array(np.bincount(input.numpy(), weights=weights, minlength=minlength))

def set_random_seed(seed):
    jt.misc.set_global_seed(seed)

def cuda_is_available():
    if jt.flags.use_cuda == 0:
        return False
    else:
        return True

def index_select(x, dim, index):
    return x.getitem(((slice(None),)*dim)+(index,))

def type_as(input, tp):
    return input.astype(tp.dtype)

def logical_not(input):
    return jt.logical_not(input)

def squeeze(input, dim=None):
    #jt.squeeze,Add dim is NOne
    if dim is None:
        shape=list(input.shape)
        for i in shape:
            if i ==1:
                shape.remove(1)
        return input.reshape(shape)
    return jt.squeeze(input, dim=dim)

def eq(input, other):
    return jt.equal(input, other)

def repeat_interleave(input, repeat, dim=None):
    # TODO Add dim, repeadt is int
    output = jt.array([item for n,s in zip(repeat.numpy(), input.numpy()) for item in [s]*n])
    return output

if __name__ == '__main__':

    a=jt.array([1,4,9,2])
    print(sort(a),sort(a)[0])
    print('111',to(a,device(a)))