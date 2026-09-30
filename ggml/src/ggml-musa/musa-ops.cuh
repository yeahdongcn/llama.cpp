#pragma once

// Vectorized fast paths for the memory-bound MUSA ops whose shared ggml-cuda kernels are limited by
// per-element instruction cost rather than by DRAM: softmax, the gated GLU, and copies that leave
// dim 0 in place. The implementations live in ggml/src/ggml-musa/ so that the shared ggml-cuda
// sources keep tracking upstream with only a guarded hook, the same shape as the pre-existing muDNN
// COPY support in cpy.cu.
//
// GGML_MUSA_VEC_OPS is OFF by default. With it off nothing in this file is compiled and the shared
// files behave exactly as upstream.
//
// Split between this header and musa-ops.cu: kernels for which the including translation unit owns a
// type the kernel has to see cannot be compiled once in musa-ops.cu, because the softmax parameter
// struct is local to ggml-cuda/softmax.cu and the GLU gate is a file-static device function of
// ggml-cuda/unary.cu. Those two are function templates instantiated where the caller is compiled; a
// translation unit that includes this header without using them emits nothing. The row copy, whose
// kernel needs no such type, is compiled once in musa-ops.cu.

#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_VEC_OPS)

#include <climits>
#include <cstdint>
#include <type_traits>

#include "ggml.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/unary.cuh"

// True when the copy is the one musa_cpy_rows_vec() handles: both sides contiguous in dim 0, i.e. a
// permutation that leaves dim 0 in place (for example the (0, 2, 1, 3) transpose of the attention
// scores), with every row starting on a 16 B boundary. Pure predicate: no stream, no launch and no
// state, so it is safe to call it from the dispatch chain; the exact preconditions are documented
// next to its implementation in musa-ops.cu.
bool musa_cpy_rows_vec_supported(const struct ggml_tensor * src0, const struct ggml_tensor * src1);

// Streams such a copy with 16 B accesses. Returns false when musa_cpy_rows_vec_supported() is not
// true, so the caller falls back to the existing kernels. Bit-copy, no numerics.
// Implemented in musa-ops.cu.
bool musa_cpy_rows_vec(const struct ggml_tensor * src0, const struct ggml_tensor * src1,
                       const char * src0_data, char * src1_data, cudaStream_t stream);

// Fields of the softmax parameter struct that the vectorized kernel needs. The struct itself is
// local to ggml-cuda/softmax.cu, so it is copied here by musa_soft_max_vec_params_from().
struct musa_soft_max_vec_params {
    int64_t  ncols;
    int64_t  ne12;
    int64_t  ne13;
    int64_t  nb11;
    int64_t  nb12;
    int64_t  nb13;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    uint32_t n_head_log2;
};

template <typename params_t>
static musa_soft_max_vec_params musa_soft_max_vec_params_from(const params_t & p) {
    musa_soft_max_vec_params r;
    r.ncols       = p.ncols;
    r.ne12        = p.ne12;
    r.ne13        = p.ne13;
    r.nb11        = p.nb11;
    r.nb12        = p.nb12;
    r.nb13        = p.nb13;
    r.scale       = p.scale;
    r.max_bias    = p.max_bias;
    r.m0          = p.m0;
    r.m1          = p.m1;
    r.n_head_log2 = p.n_head_log2;
    return r;
}

// Launches musa_soft_max_f32_vec_kernel. Implemented in musa-ops.cu.
bool musa_soft_max_vec_launch(const float * x, const void * mask, bool mask_is_f32, float * dst,
                              const musa_soft_max_vec_params & p, const dim3 & block_nums, cudaStream_t stream);

// Vectorized softmax: one block per row and 4 consecutive columns per thread, with the row kept in
// registers across both reductions. The row is therefore read from global memory exactly once and
// written once, instead of one element per thread with a shared-memory round trip, which is
// instruction bound for the small rows produced by attention.
// Preconditions mirroring the existing kernels: no sinks, 4 | ncols, 128 <= ncols <= 4*CUDA_SOFT_MAX_BLOCK_SIZE,
// 16 B aligned x/dst, and a mask that is contiguous in dim 0 (the same assumption soft_max_f32 makes
// when it indexes the mask). A f32 mask row is additionally read with a 16 B load here, so its rows
// must be 16 B aligned; a half mask is read element-wise and has no such requirement.
// The column -> thread mapping and therefore the sum reduction order differ from soft_max_f32, so
// this is tolerance-equal to the CPU reference, not bit-identical; the max is unaffected.
template <typename params_t, typename T>
static bool musa_soft_max_f32_vec(const float * x, const T * mask, const float * sinks, float * dst,
                                  const params_t & params, const dim3 & block_nums, cudaStream_t stream) {
    const bool mask_ok = mask == nullptr || !std::is_same_v<T, float> ||
                         (((uintptr_t) mask % 16) == 0 && params.nb11 % 16 == 0 &&
                          params.nb12 % 16 == 0 && params.nb13 % 16 == 0);

    const bool vec_ok = mask_ok && sinks == nullptr &&
                        params.ncols % 4 == 0 && params.ncols >= 128 &&
                        params.ncols <= 4*CUDA_SOFT_MAX_BLOCK_SIZE &&
                        ((uintptr_t) x   % 16) == 0 && ((uintptr_t) dst % 16) == 0;

    if (!vec_ok) {
        return false;
    }

    return musa_soft_max_vec_launch(x, mask, std::is_same_v<T, float>, dst,
                                    musa_soft_max_vec_params_from(params), block_nums, stream);
}

// Vectorized GLU for f32 operands whose rows are contiguous: a 2 D grid (column quads x rows) keeps
// the 64 bit div/mod that unary_gated_op_kernel needs for the row index out of the inner loop and
// lets every thread move 16 B per operand. This is the layout of the split (fused gate/up) form and
// of separate gate/up tensors, i.e. what transformer FFNs feed in. The gate is the caller's own
// device function passed as a template argument, so the numerics are the caller's, bit for bit.
template <float (*op)(float)>
static __global__ void musa_glu_gated_vec_f32_kernel(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const int64_t nrows, const int64_t n, const int64_t o0, const int64_t o1) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    for (int64_t r = blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t c = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*4;

        if (c >= n) {
            return;
        }

        const float4 xv = *reinterpret_cast<const float4 *>(x + r*o0 + c);
        const float4 gv = *reinterpret_cast<const float4 *>(g + r*o1 + c);

        *reinterpret_cast<float4 *>(dst + r*n + c) = make_float4(
            op(xv.x)*gv.x, op(xv.y)*gv.y, op(xv.z)*gv.z, op(xv.w)*gv.w);
    }
}

// Returns true when the vectorized form handled the launch; false keeps the scalar kernel.
template <float (*op)(float), typename T>
static bool musa_glu_gated_vec_f32(const T * x, const T * g, T * dst,
                                   const int64_t k, const int64_t n,
                                   const int64_t o0, const int64_t o1, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>) {
        // n is the row length and o0/o1 are the row strides in elements; when both rows are
        // contiguous every row starts at a 16 B boundary and the vectorized form can be used
        const bool vec_ok = n > 0 && n % 4 == 0 && o0 % 4 == 0 && o1 % 4 == 0 &&
                            ((uintptr_t) x   % 16) == 0 &&
                            ((uintptr_t) g   % 16) == 0 &&
                            ((uintptr_t) dst % 16) == 0;

        if (vec_ok) {
            const int64_t nrows = k / n;
            const int64_t nquad = (n + 3)/4;
            const int64_t gx    = (nquad + CUDA_GLU_BLOCK_SIZE - 1)/CUDA_GLU_BLOCK_SIZE;
            const int64_t gy    = nrows < 65535 ? nrows : 65535;

            if (gx <= INT_MAX && gy > 0) {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
                    dim3((unsigned int) gx, (unsigned int) gy, 1), CUDA_GLU_BLOCK_SIZE, 0, stream);
                ggml_cuda_kernel_launch(musa_glu_gated_vec_f32_kernel<op>, launch_params, x, g, dst, nrows, n, o0, o1);
                return true;
            }
        }
    }

    return false;
}

#endif // GGML_USE_MUSA && GGML_MUSA_VEC_OPS
