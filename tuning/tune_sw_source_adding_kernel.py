#!/usr/bin/env python
import os
from collections import OrderedDict
import numpy as np

import kernel_tuner
from kernel_tuner.observers import BenchmarkObserver

file_path = os.path.dirname(os.path.abspath(__file__))[:-6]


def tune_sw_source_adding_kernel(type_float):
    """template<typename TF>__global__
    void sw_source_adding_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                                 const TF* __restrict__ sfc_alb_dir, const TF* __restrict__ sfc_alb_dif,
                                 TF* __restrict__ r_dif, TF* __restrict__ t_dif,
                                 TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                                 TF* __restrict__ flux_up, TF* __restrict__ flux_dn, TF* __restrict__ flux_dir,
                                 TF* __restrict__ source_up, TF* __restrict__ source_dn, TF* __restrict__ source_sfc,
                                 TF* __restrict__ albedo, TF* __restrict__ src, TF* __restrict__ denom)
    """

    if type_float == "float":
        use_type = np.float32
    else:
        use_type = np.float64

    #can we vary ncols?
    ncol, nlay, ngpt = np.int32(16), np.int32(256), np.int32(256)

    top_at_1 = np.int32(0)

    r_dif, t_dif, r_dir, t_dir, t_noscat, source_up, source_dn, denom = np.zeros((8, ncol, nlay, ngpt), dtype=use_type)

    source_sfc = np.zeros((256, 256), dtype=use_type) #guess for alb_size
    albedo, src = np.zeros((2, 16, 256, 256), dtype=use_type) #guess flx_size

    sfc_alb_dir, sfc_alb_dif = np.random.random((2, 256, 256)).astype(use_type) #guessed size
    flux_up, flux_dn, flux_dir = np.random.random((3, 16, 256, 256)).astype(use_type) #guessed size

    args = [ncol, nlay, ngpt, top_at_1, sfc_alb_dir, sfc_alb_dif, r_dif, t_dif, r_dir, t_dir, t_noscat, flux_up, flux_dn, flux_dir, source_up, source_dn, source_sfc, albedo, src, denom]


    """ const int block_col2d = 32;  //block_size_x
        const int block_gpt2d = 32;  //block_size_y

        const int grid_col2d  = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d  = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);
        sw_source_adding_kernel<<<grid_gpu2d, block_gpu2d>>>(
    """
    problem_size = (ncol, ngpt)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16]
    tune_params["block_size_y"] = [8, 16, 32, 64]

    tune_params2 = OrderedDict()
    tune_params2.update(tune_params)
    tune_params2["loop_unroll_factor_n"] = [0, 1, 16, 32]

    tune_params["loop_unroll_factor_n"] = [0]

    kernel_name = f"sw_source_adding_kernel<{type_float}>"
    source_file = file_path + "src_cuda/rte_solver_kernels_only.cu"
    include_path = file_path + "include"

    cp = ["-O3", "--std=c++11", "-I"+include_path]
    #cp = cp + ["--maxrregcount=64"]

    #get number of registers
    class RegisterObserver(BenchmarkObserver):
        def get_results(self):
            return {"num_regs": self.dev.func.num_regs}
    reg_observer = RegisterObserver()

    metrics = OrderedDict()
    metrics["registers"] = lambda p: p["num_regs"] #not strictly necessary, but this forces KT to print registers while tuning

    print(kernel_name)
    results, env = kernel_tuner.tune_kernel(kernel_name, source_file, problem_size, args, tune_params,
                                            compiler_options=cp, observers=[reg_observer], metrics=metrics)




    kernel_name = f"sw_source_adding_kernel<{type_float}, 0>"
    source_file = file_path + "src_cuda/rte_solver_kernels_only_temp.cu"

    print(kernel_name)
    results, env = kernel_tuner.tune_kernel(kernel_name, source_file, problem_size, args, tune_params,
                                            compiler_options=cp, observers=[reg_observer], metrics=metrics)

    kernel_name = f"sw_source_adding_kernel<{type_float}, 1>"
    source_file = file_path + "src_cuda/rte_solver_kernels_only_temp.cu"

    print(kernel_name)
    results, env = kernel_tuner.tune_kernel(kernel_name, source_file, problem_size, args, tune_params2,
                                            compiler_options=cp, observers=[reg_observer], metrics=metrics)





if __name__ == "__main__":
    tune_sw_source_adding_kernel("float")
    tune_sw_source_adding_kernel("double")











