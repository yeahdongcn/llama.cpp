#include "musa-ops.cuh"

#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_VEC_OPS)

// ---------------------------------------------------------------------------------------------
// Copies that leave dim 0 in place
// ---------------------------------------------------------------------------------------------

// Copy of rows that are contiguous in their first dimension on both sides, i.e. a permutation that
// leaves dim 0 in place (for example the (0, 2, 1, 3) transpose of the attention scores). Both the
// read and the write of a row are then sequential, so each thread can move a full 16 B vector and
// the generic per-element index computation (which costs several 64 bit divisions per element) is
// replaced by one division per row.
template <typename T, int vec>
static __global__ void musa_cpy_rows_vec_kernel(
        const char * __restrict__ cx, char * __restrict__ cdst,
        const int64_t ne0, const int64_t nrows,
        const int64_t ne1, const int64_t ne2,
        const int64_t nb1s, const int64_t nb2s, const int64_t nb3s,
        const int64_t nb1d, const int64_t nb2d, const int64_t nb3d) {
    static_assert(vec*sizeof(T) == 16, "vector must be 16 bytes wide");

    const int64_t i0 = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*vec;

    if (i0 >= ne0) {
        return;
    }

    for (int64_t r = blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t i1  = r % ne1;
        const int64_t i23 = r / ne1;
        const int64_t i2  = i23 % ne2;
        const int64_t i3  = i23 / ne2;

        const int64_t off_s = i1*nb1s + i2*nb2s + i3*nb3s;
        const int64_t off_d = i1*nb1d + i2*nb2d + i3*nb3d;

        *reinterpret_cast<uint4 *>(cdst + off_d + i0*sizeof(T)) =
            *reinterpret_cast<const uint4 *>(cx + off_s + i0*sizeof(T));
    }
}

template <typename T, int vec>
static void musa_cpy_rows_vec_launch(const char * cx, char * cdst,
        const int64_t ne0, const int64_t nrows, const int64_t ne1, const int64_t ne2,
        const int64_t nb1s, const int64_t nb2s, const int64_t nb3s,
        const int64_t nb1d, const int64_t nb2d, const int64_t nb3d, cudaStream_t stream) {
    const int64_t nquad = (ne0 + vec - 1)/vec;
    const int64_t gx    = (nquad + CUDA_CPY_BLOCK_SIZE - 1)/CUDA_CPY_BLOCK_SIZE;
    const int64_t gy    = nrows < 65535 ? nrows : 65535;

    dim3 dimGrid((unsigned int) gx, (unsigned int) gy, 1);
    dim3 dimBlock(CUDA_CPY_BLOCK_SIZE, 1, 1);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dimGrid, dimBlock, 0, stream);
    ggml_cuda_kernel_launch(musa_cpy_rows_vec_kernel<T, vec>, launch_params, cx, cdst, ne0, nrows, ne1, ne2,
        nb1s, nb2s, nb3s, nb1d, nb2d, nb3d);
}

bool musa_cpy_rows_vec_supported(const ggml_tensor * src0, const ggml_tensor * src1) {
    const size_t  esz = ggml_element_size(src0);
    const int64_t vec = esz == 4 ? 4 : (esz == 2 ? 8 : 0);

    // Both sides are contiguous in dim 0, so rows can be copied with 16 B vectors when every row
    // starts on a 16 B boundary. Every test below is a precondition of the *generic* scalar copy,
    // i.e. this path only ever replaces copies that ggml_cuda_cpy would have mapped element by
    // element, one flat index to the same flat index:
    //   - same type: no converting pair (F32 -> BF16 and friends) can enter here,
    //   - equal ne[] on all four dims: the mapping is then the identity on (i0, i1, i2, i3). With
    //     equal element counts but different ne[], as in the KV-cache writes (3 D source, 2 D
    //     destination view), the destination's rows are not the source's rows and a row-wise copy
    //     would map the wrong elements,
    //   - nb[0] == element size on both sides: dim 0 is contiguous, which also keeps the block
    //     types out (for those ggml_element_size() != ggml_type_size()),
    //   - nb[1] is a 16 B multiple, so the transposed layout that the caller handles separately
    //     (can_be_transposed, i.e. nb[1] == element size) can never reach this path: for the two
    //     element sizes handled here nb[1] would be 2 or 4 and fail that test.
    // It is deliberately a subset of those copies, not an exact cover.
    // The 16 B alignment of the two data pointers belongs to the predicate because the caller passes
    // the tensors' own data pointers, and the vectorized kernel would fault on a misaligned one.
    const bool rows_vec_ok = vec > 0 && src0->type == src1->type &&
        src0->ne[0] == src1->ne[0] && src0->ne[1] == src1->ne[1] &&
        src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3] &&
        src0->nb[0] == esz && src1->nb[0] == esz && (src0->ne[0] % vec) == 0 &&
        ((uintptr_t) src0->data % 16) == 0 && ((uintptr_t) src1->data % 16) == 0 &&
        (src0->nb[1] % 16) == 0 && (src0->nb[2] % 16) == 0 && (src0->nb[3] % 16) == 0 &&
        (src1->nb[1] % 16) == 0 && (src1->nb[2] % 16) == 0 && (src1->nb[3] % 16) == 0;

    return rows_vec_ok;
}

bool musa_cpy_rows_vec(const ggml_tensor * src0, const ggml_tensor * src1,
                       const char * src0_data, char * src1_data, cudaStream_t stream) {
    if (!musa_cpy_rows_vec_supported(src0, src1)) {
        return false;
    }

    const int64_t ne0   = src0->ne[0];
    const int64_t nrows = src0->ne[1]*src0->ne[2]*src0->ne[3];

    if (ggml_element_size(src0) == 4) {
        musa_cpy_rows_vec_launch<float, 4>(src0_data, src1_data, ne0, nrows, src0->ne[1], src0->ne[2],
            src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3], stream);
    } else {
        musa_cpy_rows_vec_launch<half, 8>(src0_data, src1_data, ne0, nrows, src0->ne[1], src0->ne[2],
            src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3], stream);
    }

    return true;
}

// ---------------------------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------------------------

// Vector of 4 consecutive mask elements -> float. The mask row is assumed to be contiguous in its
// first dimension by the callers of both this kernel and the shared soft_max_f32; a half mask is
// therefore loaded element-wise and a float mask as one 16 B vector.
template <typename T>
static __device__ __forceinline__ void musa_soft_max_load_f32(const T * __restrict__ p, float v[4]) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 f = *reinterpret_cast<const float4 *>(p);
        v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
    } else {
        static_assert(std::is_same_v<T, half>, "unsupported mask type");
        v[0] = __half2float(p[0]);
        v[1] = __half2float(p[1]);
        v[2] = __half2float(p[2]);
        v[3] = __half2float(p[3]);
    }
}

// Vectorized softmax: one block per row and 4 consecutive columns per thread, with the row kept in
// registers across both reductions. The row is therefore read from global memory exactly once and
// written once, instead of one element per thread with a shared-memory round trip, which is
// instruction bound for the small rows produced by attention.
// The column -> thread mapping differs from soft_max_f32, so the sum reduction order differs too and
// the result is tolerance-equal to the CPU reference rather than bit-identical; the max is not
// affected. The column loop is guarded, so ncols need not be a multiple of 4*blockDim.
template <typename T, bool has_mask>
static __global__ void musa_soft_max_f32_vec_kernel(
        const float * __restrict__ x, const T * __restrict__ mask, float * __restrict__ dst,
        const int64_t ncols_p, const int64_t ne12, const int64_t ne13,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const float scale, const float max_bias, const uint32_t n_head_log2,
        const float m0, const float m1) {
    const int ncols = (int) ncols_p;
    const int tid   = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    const int64_t rowx = i01 + i02 * (int64_t) gridDim.x + i03 * (int64_t) gridDim.x * gridDim.y;

    const float * __restrict__ xr = x + rowx * ncols;
    float * __restrict__ dr = dst + rowx * ncols;

    const int col0 = tid * 4;

    float vals[4];
    if (has_mask) {
        const int64_t i12 = i02 % ne12;
        const int64_t i13 = i03 % ne13;
        const T * __restrict__ mr = mask + (i01 * nb11 + i12 * nb12 + i13 * nb13) / (int64_t) sizeof(T);
        const float slope = get_alibi_slope(max_bias, (uint32_t) i02, n_head_log2, m0, m1);

        if (col0 < ncols) {
            const float4 xv = *reinterpret_cast<const float4 *>(xr + col0);
            float mv[4];
            musa_soft_max_load_f32<T>(mr + col0, mv);
            vals[0] = xv.x * scale + slope * mv[0];
            vals[1] = xv.y * scale + slope * mv[1];
            vals[2] = xv.z * scale + slope * mv[2];
            vals[3] = xv.w * scale + slope * mv[3];
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vals[i] = -INFINITY;
            }
        }
    } else {
        if (col0 < ncols) {
            const float4 xv = *reinterpret_cast<const float4 *>(xr + col0);
            vals[0] = xv.x * scale;
            vals[1] = xv.y * scale;
            vals[2] = xv.z * scale;
            vals[3] = xv.w * scale;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vals[i] = -INFINITY;
            }
        }
    }

    extern __shared__ float data_musa_soft_max_f32_vec[];
    float * buf_iw = data_musa_soft_max_f32_vec;

    float max_val = fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[2], vals[3]));
    max_val = block_reduce<block_reduce_method::MAX, 0>(max_val, buf_iw);

    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = expf(vals[i] - max_val);
        tmp += vals[i];
    }

    if (blockDim.x > WARP_SIZE) {
        // sync is needed as we reuse buf_iw across block_reduce invocations
        __syncthreads();
    }
    tmp = block_reduce<block_reduce_method::SUM, 0>(tmp, buf_iw);

    if (col0 < ncols) {
        const float inv_sum = 1.0f / tmp;
        *reinterpret_cast<float4 *>(dr + col0) =
            make_float4(vals[0] * inv_sum, vals[1] * inv_sum, vals[2] * inv_sum, vals[3] * inv_sum);
    }
}

bool musa_soft_max_vec_launch(const float * x, const void * mask, bool mask_is_f32, float * dst,
                              const musa_soft_max_vec_params & p, const dim3 & block_nums, cudaStream_t stream) {
    const int nthread = (int) (((p.ncols/4 + WARP_SIZE - 1)/WARP_SIZE)*WARP_SIZE);
    const dim3 block_dims(nthread, 1, 1);
    const size_t nbytes_shared = WARP_SIZE*sizeof(float);

    if (mask != nullptr) {
        if (mask_is_f32) {
            musa_soft_max_f32_vec_kernel<float, true><<<block_nums, block_dims, nbytes_shared, stream>>>(
                x, (const float *) mask, dst, p.ncols, p.ne12, p.ne13, p.nb11, p.nb12, p.nb13,
                p.scale, p.max_bias, p.n_head_log2, p.m0, p.m1);
        } else {
            musa_soft_max_f32_vec_kernel<half, true><<<block_nums, block_dims, nbytes_shared, stream>>>(
                x, (const half *) mask, dst, p.ncols, p.ne12, p.ne13, p.nb11, p.nb12, p.nb13,
                p.scale, p.max_bias, p.n_head_log2, p.m0, p.m1);
        }
    } else {
        musa_soft_max_f32_vec_kernel<float, false><<<block_nums, block_dims, nbytes_shared, stream>>>(
            x, (const float *) nullptr, dst, p.ncols, p.ne12, p.ne13, p.nb11, p.nb12, p.nb13,
            p.scale, p.max_bias, p.n_head_log2, p.m0, p.m1);
    }

    return true;
}

#endif // GGML_USE_MUSA && GGML_MUSA_VEC_OPS
