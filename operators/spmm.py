# import os
# import jittor as jt
# from jittor import Function
# from jittor.compiler import compile_torch_extensions
# cached_op = {"csr_spmm": None}


# def init_spmm_ops():
#     if cached_op["csr_spmm"] is None:
#         op_path = os.path.abspath(__file__)
#         spmm_path = os.path.join(os.path.dirname(op_path), "spmm/spmm.cpp")
#         spmm_cu_path = os.path.join(os.path.dirname(op_path), "spmm/spmm_kernel.cu")
#         sample_path = os.path.join(os.path.dirname(op_path), "sample/sample.cpp")
#         compile_torch_extensions("spmm", [spmm_path, spmm_cu_path], [], [], [],1, 1)
#         # compile_torch_extensions("spmm", ["/home/qingfei/app/anaconda3/envs/jittor/lib/python3.7/site-packages/cogdl_jittor/operators/spmm/spmm.cpp","/home/qingfei/app/anaconda3/envs/jittor/lib/python3.7/site-packages/cogdl_jittor/operators/spmm/spmm_kernel.cu"], [], [], [],1, 1)
#         # compile_torch_extensions("sddmm", ["/home/qingfei/fei/cogdl_back/cogdl/cogdl/operators/spmm/sddmm.cpp","/home/qingfei/fei/cogdl_back/cogdl/cogdl/operators/spmm/sddmm_kernel.cu"], [], [], [],1, 1)
#         # compile_torch_extensions("sampler", ["/home/qingfei/fei/cogdl_back/cogdl/cogdl/operators/sample/sample.cpp"], [], [], [],1, 1)
#         from spmm import csr_spmm
#         # from sddmm import csr_sddmm
#         # from spmm import csr2csc

#         cached_op["csr_spmm"] = csr_spmm


# def spmm(graph, x):
#     row_ptr, col_indices = graph.row_indptr, graph.col_indices
#     csr_data = graph.edge_weight
#     spmm1 = SPMM()
#     x = spmm1(row_ptr.int(), col_indices.int(), x,csr_data)
#     return x


# class SPMM(Function):
#     def execute(self, rowptr, colind, feat, edge_weight_csr=None):
#         init_spmm_ops()
#         self.csr_spmm = cached_op["csr_spmm"]

#         out = self.csr_spmm(rowptr, colind, edge_weight_csr, feat)
#         self.backward_csc = (rowptr, colind, edge_weight_csr)
#         return out

#     def grad(self, grad_out):
#         rowptr, colind, edge_weight_csr = self.backward_csc
#         colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
#         grad_feat = self.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)

#         return None, None, grad_feat, None



import os
import numpy as np
import jittor 
import torch
from jittor import Function
from jittor.compiler import compile_torch_extensions
path = os.path.join(os.path.dirname(__file__))

# SPMM


try:
    compile_torch_extensions("spmm",[os.path.join(path, "spmm/spmm.cpp"), os.path.join(path, "spmm/spmm_kernel.cu")], [], [], [],1, 1)
    # compile_torch_extensions("sddmm",[os.path.join(path, "spmm/sddmm.cpp"), os.path.join(path, "spmm/sddmm_kernel.cu")], [], [], [],1, 1)
    import spmm
    def csrspmm(rowptr, colind, x, csr_data, sym=False, actnn=False):
        if actnn:
            return ActSPMMFunction.apply(rowptr, colind, x, csr_data, sym)
        return SPMMFunction.apply(rowptr, colind, x, csr_data, sym)


except Exception:
    csrspmm = None


# try:
#     spmm_cpu = load(
#         name="spmm_cpu", extra_cflags=["-fopenmp"], sources=[os.path.join(path, "spmm/spmm_cpu.cpp")], verbose=False,
#     )
#     spmm_cpu = spmm_cpu.csr_spmm_cpu
# except Exception:
#     spmm_cpu = None


class SPMMFunction(Function):
    def execute(self, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            self.backward_csc = (rowptr, colind, feat, edge_weight_csr, sym)
        else:
            self.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    def grad(self, grad_out):
        if len(self.backward_csc) == 5:
            rowptr, colind, feat, edge_weight_csr, sym = self.backward_csc
        else:
            rowptr, colind, edge_weight_csr, sym = self.backward_csc
        if edge_weight_csr is not None:
            if sym:
                colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
            else:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
            grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
            if edge_weight_csr.requires_grad:
                grad_edge_weight = sddmm.csr_sddmm(rowptr, colind, grad_out, feat)
            else:
                grad_edge_weight = None
        else:
            if sym is False:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
                grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            else:
                grad_feat = spmm.csr_spmm_no_edge_value(rowptr, colind, grad_out)
            grad_edge_weight = None
        return None, None, grad_feat, grad_edge_weight, None
    

    # def grad(self, grad_out):
    #     rowptr, colind, edge_weight_csr ,edge_weight_csr ,sym= self.backward_csc
    #     colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
    #     grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)

    #     return None, None, grad_feat, None

# try:
#     from actnn.ops import quantize_activation, dequantize_activation
# except Exception:
#     pass


class ActSPMMFunction(Function):
    @staticmethod
    def execute(self, rowptr, colind, feat, edge_weight_csr=None, sym=False):
        if edge_weight_csr is None:
            out = spmm.csr_spmm_no_edge_value(rowptr, colind, feat)
        else:
            out = spmm.csr_spmm(rowptr, colind, edge_weight_csr, feat)
        if edge_weight_csr is not None and edge_weight_csr.requires_grad:
            quantized = quantize_activation(feat, None)
            self.backward_csc = (rowptr, colind, quantized, edge_weight_csr, sym)
            self.other_args = feat.shape
        else:
            self.backward_csc = (rowptr, colind, edge_weight_csr, sym)
        return out

    @staticmethod
    def grad(self, grad_out):
        if len(self.backward_csc) == 5:
            rowptr, colind, quantized, edge_weight_csr, sym = self.backward_csc
            q_input_shape = self.other_args
            feat = dequantize_activation(quantized, q_input_shape)
            del quantized
        else:
            rowptr, colind, edge_weight_csr, sym = self.backward_csc
        del self.backward_csc

        if edge_weight_csr is not None:
            if sym:
                colptr, rowind, edge_weight_csc = rowptr, colind, edge_weight_csr
            else:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
            grad_feat = spmm.csr_spmm(colptr, rowind, edge_weight_csc, grad_out)
            if edge_weight_csr.requires_grad:
                grad_edge_weight = sddmm.csr_sddmm(rowptr, colind, grad_out, feat)
            else:
                grad_edge_weight = None
        else:
            if sym is False:
                colptr, rowind, edge_weight_csc = spmm.csr2csc(rowptr, colind, edge_weight_csr)
                grad_feat = spmm.csr_spmm_no_edge_value(colptr, rowind, grad_out)
            else:
                grad_feat = spmm.csr_spmm_no_edge_value(rowptr, colind, grad_out)
            grad_edge_weight = None
        return None, None, grad_feat, grad_edge_weight, None
