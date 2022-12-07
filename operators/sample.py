import os
from torch.utils.cpp_extension import load
from jittor.compiler import compile_torch_extensions

path = os.path.join(os.path.dirname(__file__))

compile_torch_extensions("spmm", [os.path.join(path,"spmm/spmm.cpp")], [], [], [],1, 1)
# print("2222",spmm)
compile_torch_extensions("sampler", [os.path.join(path,"sample/sample.cpp")], [], [], [],1, 1)
# from sampler import subgraph
# print("----------------",sampler)
# subgraph and sample_adj
try:
    print("------------path",os.path.join(path,"sample/sample.cpp"))
    compile_torch_extensions("sampler", os.path.join(path,"sample/sample.cpp"), [], [], [],1, 1)
    #sample = load(name="sampler", sources=[os.path.join(path, "sample/sample.cpp")], verbose=False)
    from sample import subgraph
    print("----------------",sample)
    subgraph_c = sample.subgraph
    sample_adj_c = sample.sample_adj
    coo2csr_cpu = sample.coo2csr_cpu
    coo2csr_cpu_index = sample.coo2csr_cpu_index
    print("----------------",coo2csr_cpu)
except Exception:
    subgraph_c = None
    sample_adj_c = None
    coo2csr_cpu_index = None
    coo2csr_cpu = None
