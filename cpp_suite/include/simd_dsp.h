#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <algorithm>
#include <vector>
#include <immintrin.h>

#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define SIMD_HAS_FMA 1
#else
#define SIMD_HAS_FMA 0
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace SimdDsp {

enum class CpuSimdLevel {
    Scalar = 0,
    SSE2 = 1,
    SSE4_1 = 2,
    AVX2 = 3
};

inline CpuSimdLevel detect_cpu_simd() {
    static const CpuSimdLevel cached_level = []() -> CpuSimdLevel {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            return CpuSimdLevel::AVX2;
        }
        if (__builtin_cpu_supports("sse4.1")) {
            return CpuSimdLevel::SSE4_1;
        }
        if (__builtin_cpu_supports("sse2")) {
            return CpuSimdLevel::SSE2;
        }
#elif defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        int nIds = info[0];
        bool has_sse2 = false;
        bool has_sse41 = false;
        bool has_fma = false;
        bool has_osxsave = false;
        bool has_avx = false;
        if (nIds >= 1) {
            __cpuid(info, 1);
            has_sse2    = (info[3] & (1 << 26)) != 0;
            has_sse41   = (info[2] & (1 << 19)) != 0;
            has_fma     = (info[2] & (1 << 12)) != 0;
            has_osxsave = (info[2] & (1 << 27)) != 0;
            has_avx     = (info[2] & (1 << 28)) != 0;
        }
        if (nIds >= 7 && has_osxsave && has_avx && has_fma) {
            unsigned long long xcr0 = _xgetbv(0);
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                __cpuidex(info, 7, 0);
                if ((info[1] & (1 << 5)) != 0) {
                    return CpuSimdLevel::AVX2;
                }
            }
        }
        if (has_sse41) return CpuSimdLevel::SSE4_1;
        if (has_sse2) return CpuSimdLevel::SSE2;
#endif
        return CpuSimdLevel::Scalar;
    }();
    return cached_level;
}

inline const char* get_simd_level_name() {
    switch (detect_cpu_simd()) {
        case CpuSimdLevel::AVX2: return "AVX2 + FMA";
        case CpuSimdLevel::SSE4_1: return "SSE4.1";
        case CpuSimdLevel::SSE2: return "SSE2";
        default: return "Scalar (Fallback)";
    }
}

// -----------------------------------------------------------------------
// 64-byte Cache-Aligned Memory Allocators
// -----------------------------------------------------------------------
inline void* aligned_alloc_64(size_t bytes) {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    return _aligned_malloc(bytes, 64);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0) return nullptr;
    return ptr;
#endif
}

inline void aligned_free_64(void* ptr) {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// -----------------------------------------------------------------------
// 1. Vectorized CS16 Conversion: uint8_t [0..255] -> int16_t [-32768..32512]
// RTL-SDR formula: (sample - 128) * 256  <==>  ((int16_t)sample - 128) << 8
// -----------------------------------------------------------------------
#if defined(__AVX2__)
inline void convert_u8_to_cs16_avx2(const uint8_t* __restrict src, int16_t* __restrict dst, size_t count) {
    size_t i = 0;
    const __m256i offset128 = _mm256_set1_epi16(128);

    // Process 32 uint8 samples per iteration -> 32 int16 samples (64 bytes)
    for (; i + 32 <= count; i += 32) {
        __m128i raw_lo16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m128i raw_hi16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 16));

        // Zero-extend 16 x uint8 to 16 x int16
        __m256i v_lo = _mm256_cvtepu8_epi16(raw_lo16);
        __m256i v_hi = _mm256_cvtepu8_epi16(raw_hi16);

        // Subtract 128
        v_lo = _mm256_sub_epi16(v_lo, offset128);
        v_hi = _mm256_sub_epi16(v_hi, offset128);

        // Shift left by 8 bits (multiply by 256)
        v_lo = _mm256_slli_epi16(v_lo, 8);
        v_hi = _mm256_slli_epi16(v_hi, 8);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v_lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 16), v_hi);
    }

    // Remainder
    for (; i < count; ++i) {
        dst[i] = static_cast<int16_t>((static_cast<int>(src[i]) - 128) * 256);
    }
}
#endif

inline void convert_u8_to_cs16_sse2(const uint8_t* __restrict src, int16_t* __restrict dst, size_t count) {
    size_t i = 0;
    const __m128i zero = _mm_setzero_si128();
    const __m128i offset128 = _mm_set1_epi16(128);

    // Process 16 uint8 samples per iteration
    for (; i + 16 <= count; i += 16) {
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));

        __m128i lo = _mm_unpacklo_epi8(raw, zero);
        __m128i hi = _mm_unpackhi_epi8(raw, zero);

        lo = _mm_sub_epi16(lo, offset128);
        hi = _mm_sub_epi16(hi, offset128);

        lo = _mm_slli_epi16(lo, 8);
        hi = _mm_slli_epi16(hi, 8);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), lo);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 8), hi);
    }

    for (; i < count; ++i) {
        dst[i] = static_cast<int16_t>((static_cast<int>(src[i]) - 128) * 256);
    }
}

inline void convert_u8_to_cs16_fast(const uint8_t* __restrict src, int16_t* __restrict dst, size_t count) {
#if defined(__AVX2__)
    convert_u8_to_cs16_avx2(src, dst, count);
#else
    convert_u8_to_cs16_sse2(src, dst, count);
#endif
}

// -----------------------------------------------------------------------
// 2. Vectorized Normalization & Hann Windowing
// Computes complex float I/Q from raw uint8:
//   I = (u8[2*n]   - 127.5f) * (window[n] / 128.0f)
//   Q = (u8[2*n+1] - 127.5f) * (window[n] / 128.0f)
// -----------------------------------------------------------------------
#if defined(__AVX2__)
inline void apply_hann_window_complex_avx2(
    const uint8_t* __restrict src_u8,
    const float* __restrict window_scaled, // Precalculated: window[n] / 128.0f
    float* __restrict dst_cpx,             // Interleaved float array [r0, i0, r1, i1, ...]
    size_t n_pts
) {
    size_t n = 0;
    const __m256 c127_5 = _mm256_set1_ps(127.5f);

    // Process 8 complex points (16 uint8 -> 16 floats) per AVX2 vector iteration
    for (; n + 8 <= n_pts; n += 8) {
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_u8 + 2 * n));
        __m256i i32_lo = _mm256_cvtepu8_epi32(raw16);
        __m256i i32_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(raw16, 8));
        __m256 centered_lo = _mm256_sub_ps(_mm256_cvtepi32_ps(i32_lo), c127_5);
        __m256 centered_hi = _mm256_sub_ps(_mm256_cvtepi32_ps(i32_hi), c127_5);

        __m256 w8 = _mm256_loadu_ps(window_scaled + n);
        // [w0, w0, w1, w1 | w4, w4, w5, w5] and [w2, w2, w3, w3 | w6, w6, w7, w7]
        __m256 w_dup_lo = _mm256_unpacklo_ps(w8, w8);
        __m256 w_dup_hi = _mm256_unpackhi_ps(w8, w8);
        __m256 win03 = _mm256_permute2f128_ps(w_dup_lo, w_dup_hi, 0x20);
        __m256 win47 = _mm256_permute2f128_ps(w_dup_lo, w_dup_hi, 0x31);

        _mm256_storeu_ps(dst_cpx + 2 * n,     _mm256_mul_ps(centered_lo, win03));
        _mm256_storeu_ps(dst_cpx + 2 * n + 8, _mm256_mul_ps(centered_hi, win47));
    }

    // 4-point cleanup
    for (; n + 4 <= n_pts; n += 4) {
        __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_u8 + 2 * n));
        __m256 centered = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw8)), c127_5);
        __m128 w4 = _mm_loadu_ps(window_scaled + n);
        __m128 w_lo = _mm_unpacklo_ps(w4, w4);
        __m128 w_hi = _mm_unpackhi_ps(w4, w4);
        __m256 win8 = _mm256_set_m128(w_hi, w_lo);
        _mm256_storeu_ps(dst_cpx + 2 * n, _mm256_mul_ps(centered, win8));
    }

    // Remainder
    for (; n < n_pts; ++n) {
        float w = window_scaled[n];
        dst_cpx[2 * n]     = (static_cast<float>(src_u8[2 * n])     - 127.5f) * w;
        dst_cpx[2 * n + 1] = (static_cast<float>(src_u8[2 * n + 1]) - 127.5f) * w;
    }
}
#endif

inline void apply_hann_window_complex_scalar(
    const uint8_t* __restrict src_u8,
    const float* __restrict window_scaled,
    float* __restrict dst_cpx,
    size_t n_pts
) {
    for (size_t n = 0; n < n_pts; ++n) {
        float w = window_scaled[n];
        dst_cpx[2 * n]     = (static_cast<float>(src_u8[2 * n])     - 127.5f) * w;
        dst_cpx[2 * n + 1] = (static_cast<float>(src_u8[2 * n + 1]) - 127.5f) * w;
    }
}

inline void apply_hann_window_complex_fast(
    const uint8_t* __restrict src_u8,
    const float* __restrict window_scaled,
    float* __restrict dst_cpx,
    size_t n_pts
) {
#if defined(__AVX2__)
    apply_hann_window_complex_avx2(src_u8, window_scaled, dst_cpx, n_pts);
#else
    apply_hann_window_complex_scalar(src_u8, window_scaled, dst_cpx, n_pts);
#endif
}

// -----------------------------------------------------------------------
// 3. Fast Vectorized Math: log10 and Power in dBFS
// Max error < 0.05 dB across entire dynamic range -140 dBFS to +40 dBFS.
// Clamped to valid dBFS bounds [-140.0, +40.0], handling <= 0, subnormals, NaN/Inf.
// Evaluates at > 120 MSPS with full AVX2 / FMA.
// -----------------------------------------------------------------------
#if defined(__AVX2__)
// Fast AVX2 log2(x) using IEEE-754 exponent extraction & minimax polynomial
inline __m256 fast_log2_avx2(__m256 x) {
    __m256i ix = _mm256_castps_si256(x);
    // Exponent extraction: ((ix >> 23) - 127)
    __m256i exp_i = _mm256_sub_epi32(_mm256_srli_epi32(ix, 23), _mm256_set1_epi32(127));
    __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

    // Mantissa in [1.0, 2.0): (ix & 0x007FFFFF) | 0x3F800000
    __m256i mant_i = _mm256_or_si256(_mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF)),
                                      _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_sub_ps(_mm256_castsi256_ps(mant_i), _mm256_set1_ps(1.0f)); // m in [0.0, 1.0)

    // 4th-degree minimax polynomial for log2(1 + m) on [0, 1] with p(0)=0, p(1)=1:
    // p(m) = m * (c1 + m * (c2 + m * (c3 + m * c4)))
    const __m256 c1 = _mm256_set1_ps(1.439991293f);
    const __m256 c2 = _mm256_set1_ps(-0.681706836f);
    const __m256 c3 = _mm256_set1_ps(0.324036323f);
    const __m256 c4 = _mm256_set1_ps(-0.082320779f);

#if SIMD_HAS_FMA
    __m256 p = _mm256_fmadd_ps(c4, m, c3);
    p = _mm256_fmadd_ps(p, m, c2);
    p = _mm256_fmadd_ps(p, m, c1);
    p = _mm256_mul_ps(p, m);
    return _mm256_add_ps(exp_f, p);
#else
    __m256 p = _mm256_add_ps(_mm256_mul_ps(c4, m), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, m), c2);
    p = _mm256_add_ps(_mm256_mul_ps(p, m), c1);
    p = _mm256_mul_ps(p, m);
    return _mm256_add_ps(exp_f, p);
#endif
}

// Computes 10 * log10(x) = 10 * log2(x) * log10(2) = 3.01029995664 * log2(x)
// Clamped to valid dynamic range [-140.0 dBFS, +40.0 dBFS], handling <= 0, subnormals, and NaN/Inf
inline __m256 fast_10log10_avx2(__m256 x) {
    const __m256 k_min_input = _mm256_set1_ps(1e-14f); // 10^(-140/10) = -140 dBFS
    const __m256 k_max_input = _mm256_set1_ps(1e4f);   // 10^(+40/10)  = +40 dBFS
    const __m256 k_min_db = _mm256_set1_ps(-140.0f);
    const __m256 k_max_db = _mm256_set1_ps(40.0f);
    const __m256 k10_log10_2 = _mm256_set1_ps(3.01029995664f);

    // Sanitize input: detect NaN or subnormal/negative and clamp to [1e-14f, 1e4f]
    __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256 is_huge = _mm256_cmp_ps(x, k_max_input, _CMP_GE_OQ);
    __m256 safe_x = _mm256_blendv_ps(x, k_min_input, is_nan);
    safe_x = _mm256_max_ps(safe_x, k_min_input);
    safe_x = _mm256_min_ps(safe_x, k_max_input);

    __m256 dbfs = _mm256_mul_ps(k10_log10_2, fast_log2_avx2(safe_x));

    // Clamp output strictly to valid bounds [-140.0f, +40.0f]
    dbfs = _mm256_max_ps(dbfs, k_min_db);
    dbfs = _mm256_min_ps(dbfs, k_max_db);
    dbfs = _mm256_blendv_ps(dbfs, k_max_db, is_huge);
    return dbfs;
}

// High-precision (~1e-14) double-precision AVX2+FMA log10(x) for positive normal doubles (Estrin form)
inline __m256d fast_log10_pd_avx2(__m256d x) {
    const __m256d k_sqrt2    = _mm256_set1_pd(1.4142135623730950488);
    const __m256d k_one      = _mm256_set1_pd(1.0);
    const __m256d k_half     = _mm256_set1_pd(0.5);
    const __m256d k_bias     = _mm256_set1_pd(4503599627371519.0); // 2^52 + 1023
    const __m256d k_log10_2  = _mm256_set1_pd(0.30102999566398119521);
    const __m256d k_2log10_e = _mm256_set1_pd(0.86858896380650365530); // 2 * log10(e)
    const __m256i k_exp_zero = _mm256_set1_epi64x(0x3FF0000000000000ULL);
    const __m256i k_mant_msk = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
    const __m256i k_p2_52    = _mm256_set1_epi64x(0x4330000000000000ULL);

    __m256i ix = _mm256_castpd_si256(x);
    __m256i e_raw = _mm256_srli_epi64(ix, 52);
    __m256d k = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(e_raw, k_p2_52)), k_bias);
    __m256d m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(ix, k_mant_msk), k_exp_zero));

    // Shift mantissa to [sqrt(2)/2, sqrt(2)) so |u| = |(m-1)/(m+1)| <= 0.171573
    __m256d mask = _mm256_cmp_pd(m, k_sqrt2, _CMP_GE_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, k_half), mask);
    k = _mm256_blendv_pd(k, _mm256_add_pd(k, k_one), mask);

    __m256d u = _mm256_div_pd(_mm256_sub_pd(m, k_one), _mm256_add_pd(m, k_one));
    __m256d u2 = _mm256_mul_pd(u, u);
    __m256d u4 = _mm256_mul_pd(u2, u2);
    __m256d u8 = _mm256_mul_pd(u4, u4);

    const __m256d c15 = _mm256_set1_pd(1.0 / 15.0);
    const __m256d c13 = _mm256_set1_pd(1.0 / 13.0);
    const __m256d c11 = _mm256_set1_pd(1.0 / 11.0);
    const __m256d c9  = _mm256_set1_pd(1.0 / 9.0);
    const __m256d c7  = _mm256_set1_pd(1.0 / 7.0);
    const __m256d c5  = _mm256_set1_pd(1.0 / 5.0);
    const __m256d c3  = _mm256_set1_pd(1.0 / 3.0);

#if SIMD_HAS_FMA
    __m256d p01 = _mm256_fmadd_pd(c3,  u2, k_one);
    __m256d p23 = _mm256_fmadd_pd(c7,  u2, c5);
    __m256d p45 = _mm256_fmadd_pd(c11, u2, c9);
    __m256d p67 = _mm256_fmadd_pd(c15, u2, c13);
    __m256d q03 = _mm256_fmadd_pd(p23, u4, p01);
    __m256d q47 = _mm256_fmadd_pd(p67, u4, p45);
    __m256d poly = _mm256_fmadd_pd(q47, u8, q03);
    __m256d log10_m = _mm256_mul_pd(_mm256_mul_pd(k_2log10_e, u), poly);
    __m256d res = _mm256_fmadd_pd(k, k_log10_2, log10_m);
#else
    __m256d p01 = _mm256_add_pd(_mm256_mul_pd(c3,  u2), k_one);
    __m256d p23 = _mm256_add_pd(_mm256_mul_pd(c7,  u2), c5);
    __m256d p45 = _mm256_add_pd(_mm256_mul_pd(c11, u2), c9);
    __m256d p67 = _mm256_add_pd(_mm256_mul_pd(c15, u2), c13);
    __m256d q03 = _mm256_add_pd(_mm256_mul_pd(p23, u4), p01);
    __m256d q47 = _mm256_add_pd(_mm256_mul_pd(p67, u4), p45);
    __m256d poly = _mm256_add_pd(_mm256_mul_pd(q47, u8), q03);
    __m256d log10_m = _mm256_mul_pd(_mm256_mul_pd(k_2log10_e, u), poly);
    __m256d res = _mm256_add_pd(_mm256_mul_pd(k, k_log10_2), log10_m);
#endif

    // Match IEEE-754 / std::log10 special-value semantics: NaN->NaN, <0->NaN, 0->-Inf, +Inf->+Inf
    const __m256d k_zero    = _mm256_setzero_pd();
    const __m256d k_pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d k_neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d k_qnan    = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());

    __m256d is_nan  = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    __m256d is_neg  = _mm256_cmp_pd(x, k_zero, _CMP_LT_OQ);
    __m256d is_zero = _mm256_cmp_pd(x, k_zero, _CMP_EQ_OQ);
    __m256d is_pinf = _mm256_cmp_pd(x, k_pos_inf, _CMP_EQ_OQ);

    res = _mm256_blendv_pd(res, k_pos_inf, is_pinf);
    res = _mm256_blendv_pd(res, k_neg_inf, is_zero);
    res = _mm256_blendv_pd(res, k_qnan, _mm256_or_pd(is_nan, is_neg));
    return res;
}

// Shared core: evaluates 2^z = 2^k * exp((z - k) * ln2) in double precision (Estrin deg-11)
// Caller guarantees z is bounded so (k + 1023) is a valid positive normal IEEE-754 exponent.
inline __m256d exp2_pd_core_avx2(__m256d z) {
    const __m256d k_ln2  = _mm256_set1_pd(0.69314718055994530942);
    const __m256d k_bias = _mm256_set1_pd(4503599627371519.0); // 2^52 + 1023
    const __m256d k_one  = _mm256_set1_pd(1.0);

    __m256d k = _mm256_round_pd(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d f = _mm256_sub_pd(z, k);      // |f| <= 0.5
    __m256d t = _mm256_mul_pd(f, k_ln2);  // |t| <= 0.346574
    __m256d t2 = _mm256_mul_pd(t, t);
    __m256d t4 = _mm256_mul_pd(t2, t2);
    __m256d t8 = _mm256_mul_pd(t4, t4);

    const __m256d c11 = _mm256_set1_pd(1.0 / 39916800.0);
    const __m256d c10 = _mm256_set1_pd(1.0 / 3628800.0);
    const __m256d c9  = _mm256_set1_pd(1.0 / 362880.0);
    const __m256d c8  = _mm256_set1_pd(1.0 / 40320.0);
    const __m256d c7  = _mm256_set1_pd(1.0 / 5040.0);
    const __m256d c6  = _mm256_set1_pd(1.0 / 720.0);
    const __m256d c5  = _mm256_set1_pd(1.0 / 120.0);
    const __m256d c4  = _mm256_set1_pd(1.0 / 24.0);
    const __m256d c3  = _mm256_set1_pd(1.0 / 6.0);
    const __m256d c2  = _mm256_set1_pd(0.5);

    __m256d q0 = _mm256_add_pd(k_one, t);
#if SIMD_HAS_FMA
    __m256d q1 = _mm256_fmadd_pd(c3,  t, c2);
    __m256d q2 = _mm256_fmadd_pd(c5,  t, c4);
    __m256d q3 = _mm256_fmadd_pd(c7,  t, c6);
    __m256d q4 = _mm256_fmadd_pd(c9,  t, c8);
    __m256d q5 = _mm256_fmadd_pd(c11, t, c10);
    __m256d r0 = _mm256_fmadd_pd(q1, t2, q0);
    __m256d r1 = _mm256_fmadd_pd(q3, t2, q2);
    __m256d r2 = _mm256_fmadd_pd(q5, t2, q4);
    __m256d s0 = _mm256_fmadd_pd(r1, t4, r0);
    __m256d p  = _mm256_fmadd_pd(r2, t8, s0);
#else
    __m256d q1 = _mm256_add_pd(_mm256_mul_pd(c3,  t), c2);
    __m256d q2 = _mm256_add_pd(_mm256_mul_pd(c5,  t), c4);
    __m256d q3 = _mm256_add_pd(_mm256_mul_pd(c7,  t), c6);
    __m256d q4 = _mm256_add_pd(_mm256_mul_pd(c9,  t), c8);
    __m256d q5 = _mm256_add_pd(_mm256_mul_pd(c11, t), c10);
    __m256d r0 = _mm256_add_pd(_mm256_mul_pd(q1, t2), q0);
    __m256d r1 = _mm256_add_pd(_mm256_mul_pd(q3, t2), q2);
    __m256d r2 = _mm256_add_pd(_mm256_mul_pd(q5, t2), q4);
    __m256d s0 = _mm256_add_pd(_mm256_mul_pd(r1, t4), r0);
    __m256d p  = _mm256_add_pd(_mm256_mul_pd(r2, t8), s0);
#endif

    __m256i pow2k_i = _mm256_slli_epi64(_mm256_castpd_si256(_mm256_add_pd(k, k_bias)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(pow2k_i));
}

// High-precision (~1e-14) double-precision AVX2+FMA exp(x); flushes underflow <= -708.0 safely to +0.0,
// preserves NaN and maps overflow / +Inf to +Inf
inline __m256d fast_exp_pd_avx2(__m256d x) {
    const __m256d k_min_x   = _mm256_set1_pd(-708.0);
    const __m256d k_max_x   = _mm256_set1_pd(709.0);
    const __m256d k_over_x  = _mm256_set1_pd(709.782712893384);
    const __m256d k_log2_e  = _mm256_set1_pd(1.44269504088896340736);
    const __m256d k_pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

    __m256d is_nan  = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    __m256d is_over = _mm256_cmp_pd(x, k_over_x, _CMP_GT_OQ);
    __m256d mask_gt = _mm256_cmp_pd(x, k_min_x, _CMP_GT_OQ);
    __m256d safe_x = _mm256_min_pd(_mm256_max_pd(x, k_min_x), k_max_x);
    __m256d res = _mm256_and_pd(exp2_pd_core_avx2(_mm256_mul_pd(safe_x, k_log2_e)), mask_gt);
    res = _mm256_blendv_pd(res, k_pos_inf, is_over);
    res = _mm256_blendv_pd(res, x, is_nan);
    return res;
}

// High-precision (~1e-14) double-precision AVX2+FMA pow(10, x);
// preserves NaN, flushes underflow <= -308.0 to +0.0, and maps overflow / +Inf to +Inf
inline __m256d fast_exp10_pd_avx2(__m256d x) {
    const __m256d k_under   = _mm256_set1_pd(-308.0);
    const __m256d k_min_x   = _mm256_set1_pd(-307.5);
    const __m256d k_max_x   = _mm256_set1_pd(308.0);
    const __m256d k_over    = _mm256_set1_pd(308.25471555991675);
    const __m256d k_log2_10 = _mm256_set1_pd(3.32192809488736234787);
    const __m256d k_pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d k_zero    = _mm256_setzero_pd();

    __m256d is_nan   = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    __m256d is_over  = _mm256_cmp_pd(x, k_over, _CMP_GT_OQ);
    __m256d is_under = _mm256_cmp_pd(x, k_under, _CMP_LT_OQ);

    __m256d safe_x = _mm256_min_pd(_mm256_max_pd(x, k_min_x), k_max_x);
    __m256d res = exp2_pd_core_avx2(_mm256_mul_pd(safe_x, k_log2_10));
    res = _mm256_blendv_pd(res, k_zero, is_under);
    res = _mm256_blendv_pd(res, k_pos_inf, is_over);
    res = _mm256_blendv_pd(res, x, is_nan);
    return res;
}
#endif

// Vectorized 10 * log10(max(in[i], floor_val)) over double array (2x unrolled AVX2)
inline void power_to_db_f64(const double* __restrict in, double* __restrict out, size_t n, double floor_val = 1e-12) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256d v_floor = _mm256_set1_pd(floor_val);
    const __m256d v_ten   = _mm256_set1_pd(10.0);
    for (; i + 8 <= n; i += 8) {
        __m256d x0 = _mm256_loadu_pd(in + i);
        __m256d x1 = _mm256_loadu_pd(in + i + 4);
        __m256d nan0 = _mm256_cmp_pd(x0, x0, _CMP_UNORD_Q);
        __m256d nan1 = _mm256_cmp_pd(x1, x1, _CMP_UNORD_Q);
        __m256d safe0 = _mm256_max_pd(_mm256_blendv_pd(x0, v_floor, nan0), v_floor);
        __m256d safe1 = _mm256_max_pd(_mm256_blendv_pd(x1, v_floor, nan1), v_floor);
        _mm256_storeu_pd(out + i,     _mm256_mul_pd(v_ten, fast_log10_pd_avx2(safe0)));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(v_ten, fast_log10_pd_avx2(safe1)));
    }
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(in + i);
        __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        __m256d safe_x = _mm256_max_pd(_mm256_blendv_pd(x, v_floor, is_nan), v_floor);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(v_ten, fast_log10_pd_avx2(safe_x)));
    }
#endif
    for (; i < n; ++i) {
        double v = (std::isnan(in[i]) || in[i] < floor_val) ? floor_val : in[i];
        out[i] = 10.0 * std::log10(v);
    }
}

// Vectorized 10^(in_db[i] / 10.0) over double array (2x unrolled AVX2)
inline void db_to_power_f64(const double* __restrict in_db, double* __restrict out, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256d v_inv10 = _mm256_set1_pd(0.1);
    for (; i + 8 <= n; i += 8) {
        __m256d db0 = _mm256_loadu_pd(in_db + i);
        __m256d db1 = _mm256_loadu_pd(in_db + i + 4);
        _mm256_storeu_pd(out + i,     fast_exp10_pd_avx2(_mm256_mul_pd(db0, v_inv10)));
        _mm256_storeu_pd(out + i + 4, fast_exp10_pd_avx2(_mm256_mul_pd(db1, v_inv10)));
    }
    for (; i + 4 <= n; i += 4) {
        __m256d db = _mm256_loadu_pd(in_db + i);
        _mm256_storeu_pd(out + i, fast_exp10_pd_avx2(_mm256_mul_pd(db, v_inv10)));
    }
#endif
    for (; i < n; ++i) {
        out[i] = std::pow(10.0, in_db[i] * 0.1);
    }
}

// Vectorized CS16 -> windowed interleaved float conversion:
// src_i16 has 2 * n_pts int16_t values [I0, Q0, I1, Q1, ...]
// win_interleaved has 2 * n_pts floats [w0, w0, w1, w1, ...]
inline void convert_cs16_window_float_fast(
    const int16_t* __restrict src_i16,
    const float* __restrict win_interleaved,
    float* __restrict dst_cpx,
    size_t n_pts
) {
    size_t k = 0;
    const size_t total_scalars = n_pts * 2;
#if defined(__AVX2__)
    for (; k + 8 <= total_scalars; k += 8) {
        __m128i raw8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_i16 + k));
        __m256i i32 = _mm256_cvtepi16_epi32(raw8);
        __m256 f = _mm256_cvtepi32_ps(i32);
        __m256 w = _mm256_loadu_ps(win_interleaved + k);
        _mm256_storeu_ps(dst_cpx + k, _mm256_mul_ps(f, w));
    }
#endif
    for (; k < total_scalars; ++k) {
        dst_cpx[k] = static_cast<float>(src_i16[k]) * win_interleaved[k];
    }
}

inline float fast_10log10_scalar(float x) {
    // Handle edge cases: NaN, <= 0, subnormal
    if (std::isnan(x) || x <= 1e-14f) {
        return -140.0f;
    }
    if (x >= 1e4f) {
        return 40.0f;
    }

    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    int exp = static_cast<int>(bits >> 23) - 127;
    bits = (bits & 0x007FFFFFu) | 0x3F800000u;
    float mant = 1.0f;
    std::memcpy(&mant, &bits, sizeof(mant));
    float m = mant - 1.0f;

    // Endpoint-continuous 4th-degree minimax polynomial matching fast_log2_avx2
    float p = -0.082320779f * m + 0.324036323f;
    p = p * m - 0.681706836f;
    p = p * m + 1.439991293f;
    p = p * m;

    float val = 3.01029995664f * (static_cast<float>(exp) + p);
    return std::clamp(val, -140.0f, 40.0f);
}

// Vectorized Power (dBFS) & optional Linear Normalized Magnitude for FFT output with FFT-shift
// out_power_db[k] = 10 * log10((r^2 + i^2) * inv_norm + eps)
// If out_mag_linear != nullptr, out_mag_linear[k] = sqrt((r^2 + i^2) * inv_norm)
inline void compute_power_dbfs_avx2(
    const float* __restrict fft_out_cpx,
    float* __restrict out_power_db,
    size_t n_pts,
    float inv_norm,
    float* __restrict out_mag_linear = nullptr
) {
    if (n_pts == 0) return;
    // Standard fftshift (matches numpy.fft.fftshift & StreamingProcessor):
    // Negative freqs: input [pos_len .. n_pts) (count = neg_len = floor(n/2)) -> out [0 .. neg_len)
    // DC + pos freqs: input [0 .. pos_len)     (count = pos_len = ceil(n/2))  -> out [neg_len .. n_pts)
    // So DC (input bin 0) always lands at output bin n_pts / 2 for both even and odd n_pts.
    const size_t neg_len = n_pts / 2;
    const size_t pos_len = n_pts - neg_len;
#if defined(__AVX2__)
    const __m256 v_inv_n = _mm256_set1_ps(inv_norm);
    const __m256 v_eps = _mm256_set1_ps(1e-14f);

    // Helper lambda to process a contiguous complex range [src_base .. src_base + len) -> out_base
    auto process_range = [&](const float* __restrict src_cpx, float* __restrict dst_db,
                             float* __restrict dst_mag, size_t len) {
        size_t i = 0;
        for (; i + 8 <= len; i += 8) {
            __m256 v0 = _mm256_loadu_ps(src_cpx + 2 * i);
            __m256 v1 = _mm256_loadu_ps(src_cpx + 2 * i + 8);
            // Deinterleave within 128-bit lanes -> [r0, r1, r4, r5 | r2, r3, r6, r7] and [i0, i1, i4, i5 | i2, i3, i6, i7]
            __m256 re_p = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));
            __m256 im_p = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1));
#if SIMD_HAS_FMA
            __m256 mag2_p = _mm256_fmadd_ps(re_p, re_p, _mm256_mul_ps(im_p, im_p));
#else
            __m256 mag2_p = _mm256_add_ps(_mm256_mul_ps(re_p, re_p), _mm256_mul_ps(im_p, im_p));
#endif
            __m256 mag2 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(mag2_p), 0xD8));

            __m256 norm_pwr = _mm256_mul_ps(mag2, v_inv_n);
            if (dst_mag) {
                _mm256_storeu_ps(dst_mag + i, _mm256_sqrt_ps(norm_pwr));
            }
            __m256 scaled = _mm256_add_ps(norm_pwr, v_eps);
            __m256 dbfs = fast_10log10_avx2(scaled);
            _mm256_storeu_ps(dst_db + i, dbfs);
        }
        for (; i < len; ++i) {
            float r = src_cpx[2 * i];
            float im = src_cpx[2 * i + 1];
            float norm_pwr = (r * r + im * im) * inv_norm;
            if (dst_mag) {
                dst_mag[i] = std::sqrt(norm_pwr);
            }
            dst_db[i] = fast_10log10_scalar(norm_pwr + 1e-14f);
        }
    };

    process_range(fft_out_cpx + 2 * pos_len, out_power_db, out_mag_linear, neg_len);
    process_range(fft_out_cpx, out_power_db + neg_len,
                  out_mag_linear ? (out_mag_linear + neg_len) : nullptr, pos_len);
#else
    for (size_t k = 0; k < n_pts; ++k) {
        size_t src_idx = (k < neg_len) ? (k + pos_len) : (k - neg_len);
        float r = fft_out_cpx[2 * src_idx];
        float i = fft_out_cpx[2 * src_idx + 1];
        float norm_pwr = (r * r + i * i) * inv_norm;
        if (out_mag_linear) {
            out_mag_linear[k] = std::sqrt(norm_pwr);
        }
        out_power_db[k] = fast_10log10_scalar(norm_pwr + 1e-14f);
    }
#endif
}

// -----------------------------------------------------------------------
// 4. Vectorized Exponential Smoothing
// smooth[k] = avg * smooth[k] + (1 - avg) * new[k]
// -----------------------------------------------------------------------
inline void apply_smoothing_avx2(
    float* __restrict smooth,
    const float* __restrict current,
    size_t count,
    float avg_factor
) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256 v_avg = _mm256_set1_ps(avg_factor);
    const __m256 v_inv_avg = _mm256_set1_ps(1.0f - avg_factor);

    for (; i + 8 <= count; i += 8) {
        __m256 sm = _mm256_loadu_ps(smooth + i);
        __m256 cur = _mm256_loadu_ps(current + i);

#if SIMD_HAS_FMA
        __m256 res = _mm256_fmadd_ps(cur, v_inv_avg, _mm256_mul_ps(sm, v_avg));
#else
        __m256 res = _mm256_add_ps(_mm256_mul_ps(sm, v_avg), _mm256_mul_ps(cur, v_inv_avg));
#endif
        _mm256_storeu_ps(smooth + i, res);
    }

    for (; i < count; ++i) {
        smooth[i] = avg_factor * smooth[i] + (1.0f - avg_factor) * current[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        smooth[i] = avg_factor * smooth[i] + (1.0f - avg_factor) * current[i];
    }
#endif
}

// -----------------------------------------------------------------------
// 5. Peak-Preserving Vectorized Waterfall Palette Mapping
// Max-pools across frequency bins per pixel column so narrow spikes remain visible,
// then quantizes 8 columns at a time with AVX2.
// -----------------------------------------------------------------------
inline void map_waterfall_line_fast(
    const float* __restrict power_db,
    size_t n_pwr,
    float min_db,
    float max_db,
    const uint32_t* __restrict palette_lut,
    uint32_t* __restrict out_pixels,
    int width
) {
    if (width <= 0 || n_pwr == 0) return;

    const float range_db = std::max(max_db - min_db, 1.0f);
    const float scale = 255.0f / range_db;

    thread_local std::vector<float> col_pwr;
    col_pwr.resize(static_cast<size_t>(width));

    if (n_pwr >= static_cast<size_t>(width)) {
        for (int x = 0; x < width; ++x) {
            size_t b0 = static_cast<size_t>(x) * n_pwr / static_cast<size_t>(width);
            size_t b1 = static_cast<size_t>(x + 1) * n_pwr / static_cast<size_t>(width);
            if (b1 <= b0) b1 = std::min(b0 + 1, n_pwr);
            float peak = -FLT_MAX;
            bool has_finite = false;
            for (size_t b = b0; b < b1; ++b) {
                const float v = power_db[b];
                if (std::isfinite(v) && (!has_finite || v > peak)) {
                    peak = v;
                    has_finite = true;
                }
            }
            col_pwr[x] = has_finite ? peak : min_db;
        }
    } else {
        for (int x = 0; x < width; ++x) {
            float pos = (width > 1) ? (static_cast<float>(x) / static_cast<float>(width - 1)) * static_cast<float>(n_pwr - 1) : 0.0f;
            size_t b0 = static_cast<size_t>(pos);
            size_t b1 = std::min(b0 + 1, n_pwr - 1);
            float frac = pos - static_cast<float>(b0);
            const float v0 = power_db[b0];
            const float v1 = power_db[b1];
            if (std::isfinite(v0) && std::isfinite(v1)) {
                col_pwr[x] = v0 + frac * (v1 - v0);
            } else if (std::isfinite(v0)) {
                col_pwr[x] = v0;
            } else if (std::isfinite(v1)) {
                col_pwr[x] = v1;
            } else {
                col_pwr[x] = min_db;
            }
        }
    }

    int x = 0;
#if defined(__AVX2__)
    const __m256 v_min   = _mm256_set1_ps(min_db);
    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_255   = _mm256_set1_ps(255.0f);
    for (; x + 8 <= width; x += 8) {
        __m256 p = _mm256_loadu_ps(col_pwr.data() + x);
        __m256 scaled = _mm256_mul_ps(_mm256_sub_ps(p, v_min), v_scale);
        // Clamp NaNs safely via min(max(scaled, 0), 255): cmp/blend or max(0, x) where NaN becomes 0
        __m256 valid = _mm256_cmp_ps(scaled, scaled, _CMP_ORD_Q);
        scaled = _mm256_and_ps(scaled, valid);
        scaled = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255);
        alignas(32) int32_t idx[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx), _mm256_cvttps_epi32(scaled));
        out_pixels[x + 0] = palette_lut[idx[0]];
        out_pixels[x + 1] = palette_lut[idx[1]];
        out_pixels[x + 2] = palette_lut[idx[2]];
        out_pixels[x + 3] = palette_lut[idx[3]];
        out_pixels[x + 4] = palette_lut[idx[4]];
        out_pixels[x + 5] = palette_lut[idx[5]];
        out_pixels[x + 6] = palette_lut[idx[6]];
        out_pixels[x + 7] = palette_lut[idx[7]];
    }
#endif
    for (; x < width; ++x) {
        float norm = (col_pwr[x] - min_db) * scale;
        if (!(norm >= 0.0f)) norm = 0.0f;
        else if (norm > 255.0f) norm = 255.0f;
        out_pixels[x] = palette_lut[static_cast<uint8_t>(norm)];
    }
}

// -----------------------------------------------------------------------
// 6. Fast In-Place AVX2+FMA Forward Complex FFT (Power-of-2 N >= 8)
// Stockham Auto-Sort Radix-8 (first stage) + Radix-4 + trailing Radix-2.
// Computes standard forward DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-i*2*pi*k*n/N).
// Returns false if AVX2 is unavailable or N is not a power of 2 >= 8.
// -----------------------------------------------------------------------
inline bool fft_cpx_f32_avx2(float* __restrict cpx, int nfft) {
#if defined(__AVX2__)
    if (!cpx || nfft < 8 || (nfft & (nfft - 1)) != 0 || detect_cpu_simd() < CpuSimdLevel::AVX2) {
        return false;
    }

    struct ScratchF32 {
        std::vector<float> storage;
        float* ptr = nullptr;

        ScratchF32() = default;
        ScratchF32(const ScratchF32&) = delete;
        ScratchF32& operator=(const ScratchF32&) = delete;
        ScratchF32(ScratchF32&& other) noexcept
            : storage(std::move(other.storage)), ptr(other.ptr) {
            other.ptr = nullptr;
        }
        ScratchF32& operator=(ScratchF32&& other) noexcept {
            if (this != &other) {
                storage = std::move(other.storage);
                ptr = other.ptr;
                other.ptr = nullptr;
            }
            return *this;
        }

        void resize(size_t count) {
            storage.resize(count + 16);
            uintptr_t raw = reinterpret_cast<uintptr_t>(storage.data());
            ptr = reinterpret_cast<float*>((raw + 31u) & ~uintptr_t(31u));
        }
    };

    struct StageTwiddles {
        int s = 0;
        int radix = 0; // 4 or 2
        ScratchF32 wr;
        ScratchF32 wi;
    };

    struct FftPlan {
        int n = 0;
        ScratchF32 re_a, im_a, re_b, im_b;
        std::vector<StageTwiddles> stages;
    };

    thread_local FftPlan plan;
    if (plan.n != nfft) {
        plan.n = nfft;
        plan.re_a.resize(static_cast<size_t>(nfft));
        plan.im_a.resize(static_cast<size_t>(nfft));
        plan.re_b.resize(static_cast<size_t>(nfft));
        plan.im_b.resize(static_cast<size_t>(nfft));
        plan.stages.clear();

        constexpr double kTwoPi = 6.283185307179586476925286766559;
        int s = 8;
        int rem = nfft / 8;
        while (rem >= 4) {
            StageTwiddles st;
            st.s = s;
            st.radix = 4;
            st.wr.resize(static_cast<size_t>(s) * 3);
            st.wi.resize(static_cast<size_t>(s) * 3);
            const double base_ang = -kTwoPi / static_cast<double>(4 * s);
            for (int j = 0; j < s; ++j) {
                double a1 = base_ang * static_cast<double>(j);
                double a2 = 2.0 * a1;
                double a3 = 3.0 * a1;
                st.wr.ptr[j]         = static_cast<float>(std::cos(a1));
                st.wi.ptr[j]         = static_cast<float>(std::sin(a1));
                st.wr.ptr[s + j]     = static_cast<float>(std::cos(a2));
                st.wi.ptr[s + j]     = static_cast<float>(std::sin(a2));
                st.wr.ptr[2 * s + j] = static_cast<float>(std::cos(a3));
                st.wi.ptr[2 * s + j] = static_cast<float>(std::sin(a3));
            }
            plan.stages.push_back(std::move(st));
            s *= 4;
            rem /= 4;
        }
        if (rem == 2) {
            StageTwiddles st;
            st.s = s;
            st.radix = 2;
            st.wr.resize(static_cast<size_t>(s));
            st.wi.resize(static_cast<size_t>(s));
            const double base_ang = -kTwoPi / static_cast<double>(2 * s);
            for (int j = 0; j < s; ++j) {
                double a1 = base_ang * static_cast<double>(j);
                st.wr.ptr[j] = static_cast<float>(std::cos(a1));
                st.wi.ptr[j] = static_cast<float>(std::sin(a1));
            }
            plan.stages.push_back(std::move(st));
        }
    }

    auto transpose_8x8 = [](__m256& r0, __m256& r1, __m256& r2, __m256& r3,
                            __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
        __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        __m256 t7 = _mm256_unpackhi_ps(r6, r7);

        __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

        r0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        r2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        r4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        r6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        r7 = _mm256_permute2f128_ps(u3, u7, 0x31);
    };

    float* re_b = plan.re_b.ptr;
    float* im_b = plan.im_b.ptr;
    const int M = nfft / 8;

    // Stage 1: Fused deinterleave + Radix-8 Stockham DIT
    if (M >= 8) {
        const __m256 c_sqrt2_2 = _mm256_set1_ps(0.7071067811865475244f);
        for (int k = 0; k < M; k += 8) {
            __m256 xr[8], xi[8];
            for (int r = 0; r < 8; ++r) {
                const float* src = cpx + 2 * (r * M + k);
                __m256 v0 = _mm256_loadu_ps(src);
                __m256 v1 = _mm256_loadu_ps(src + 8);
                xr[r] = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                    _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0))), 0xD8));
                xi[r] = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                    _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1))), 0xD8));
            }

            // 8-point forward DFT across xr[0..7], xi[0..7]
            __m256 sr0 = _mm256_add_ps(xr[0], xr[4]), si0 = _mm256_add_ps(xi[0], xi[4]);
            __m256 dr0 = _mm256_sub_ps(xr[0], xr[4]), di0 = _mm256_sub_ps(xi[0], xi[4]);
            __m256 sr1 = _mm256_add_ps(xr[1], xr[5]), si1 = _mm256_add_ps(xi[1], xi[5]);
            __m256 dr1 = _mm256_sub_ps(xr[1], xr[5]), di1 = _mm256_sub_ps(xi[1], xi[5]);
            __m256 sr2 = _mm256_add_ps(xr[2], xr[6]), si2 = _mm256_add_ps(xi[2], xi[6]);
            __m256 dr2 = _mm256_sub_ps(xr[2], xr[6]), di2 = _mm256_sub_ps(xi[2], xi[6]);
            __m256 sr3 = _mm256_add_ps(xr[3], xr[7]), si3 = _mm256_add_ps(xi[3], xi[7]);
            __m256 dr3 = _mm256_sub_ps(xr[3], xr[7]), di3 = _mm256_sub_ps(xi[3], xi[7]);

            __m256 er0 = _mm256_add_ps(sr0, sr2), ei0 = _mm256_add_ps(si0, si2);
            __m256 er1 = _mm256_sub_ps(sr0, sr2), ei1 = _mm256_sub_ps(si0, si2);
            __m256 er2 = _mm256_add_ps(sr1, sr3), ei2 = _mm256_add_ps(si1, si3);
            __m256 er3 = _mm256_sub_ps(sr1, sr3), ei3 = _mm256_sub_ps(si1, si3);

            __m256 yr0 = _mm256_add_ps(er0, er2), yi0 = _mm256_add_ps(ei0, ei2);
            __m256 yr4 = _mm256_sub_ps(er0, er2), yi4 = _mm256_sub_ps(ei0, ei2);
            __m256 yr2 = _mm256_add_ps(er1, ei3), yi2 = _mm256_sub_ps(ei1, er3);
            __m256 yr6 = _mm256_sub_ps(er1, ei3), yi6 = _mm256_add_ps(ei1, er3);

            // Odd half twiddles: W_8^0 = 1, W_8^1 = c(1-i), W_8^2 = -i, W_8^3 = c(-1-i)
            __m256 tr0 = dr0, ti0 = di0;
            __m256 tr1 = _mm256_mul_ps(c_sqrt2_2, _mm256_add_ps(dr1, di1));
            __m256 ti1 = _mm256_mul_ps(c_sqrt2_2, _mm256_sub_ps(di1, dr1));
            __m256 tr2 = di2;
            __m256 ti2 = _mm256_sub_ps(_mm256_setzero_ps(), dr2);
            __m256 tr3 = _mm256_mul_ps(c_sqrt2_2, _mm256_sub_ps(di3, dr3));
            __m256 ti3 = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(c_sqrt2_2, _mm256_add_ps(dr3, di3)));

            __m256 or0 = _mm256_add_ps(tr0, tr2), oi0 = _mm256_add_ps(ti0, ti2);
            __m256 or1 = _mm256_sub_ps(tr0, tr2), oi1 = _mm256_sub_ps(ti0, ti2);
            __m256 or2 = _mm256_add_ps(tr1, tr3), oi2 = _mm256_add_ps(ti1, ti3);
            __m256 or3 = _mm256_sub_ps(tr1, tr3), oi3 = _mm256_sub_ps(ti1, ti3);

            __m256 yr1 = _mm256_add_ps(or0, or2), yi1 = _mm256_add_ps(oi0, oi2);
            __m256 yr5 = _mm256_sub_ps(or0, or2), yi5 = _mm256_sub_ps(oi0, oi2);
            __m256 yr3 = _mm256_add_ps(or1, oi3), yi3 = _mm256_sub_ps(oi1, or3);
            __m256 yr7 = _mm256_sub_ps(or1, oi3), yi7 = _mm256_add_ps(oi1, or3);

            transpose_8x8(yr0, yr1, yr2, yr3, yr4, yr5, yr6, yr7);
            transpose_8x8(yi0, yi1, yi2, yi3, yi4, yi5, yi6, yi7);

            __m256* oy_r[8] = { &yr0, &yr1, &yr2, &yr3, &yr4, &yr5, &yr6, &yr7 };
            __m256* oy_i[8] = { &yi0, &yi1, &yi2, &yi3, &yi4, &yi5, &yi6, &yi7 };
            for (int m = 0; m < 8; ++m) {
                _mm256_store_ps(re_b + 8 * (k + m), *oy_r[m]);
                _mm256_store_ps(im_b + 8 * (k + m), *oy_i[m]);
            }
        }
    } else {
        constexpr float kSqrt2_2 = 0.7071067811865475244f;
        for (int k = 0; k < M; ++k) {
            float xr[8], xi[8];
            for (int r = 0; r < 8; ++r) {
                xr[r] = cpx[2 * (r * M + k)];
                xi[r] = cpx[2 * (r * M + k) + 1];
            }
            float sr0 = xr[0] + xr[4], si0 = xi[0] + xi[4];
            float dr0 = xr[0] - xr[4], di0 = xi[0] - xi[4];
            float sr1 = xr[1] + xr[5], si1 = xi[1] + xi[5];
            float dr1 = xr[1] - xr[5], di1 = xi[1] - xi[5];
            float sr2 = xr[2] + xr[6], si2 = xi[2] + xi[6];
            float dr2 = xr[2] - xr[6], di2 = xi[2] - xi[6];
            float sr3 = xr[3] + xr[7], si3 = xi[3] + xi[7];
            float dr3 = xr[3] - xr[7], di3 = xi[3] - xi[7];

            float er0 = sr0 + sr2, ei0 = si0 + si2;
            float er1 = sr0 - sr2, ei1 = si0 - si2;
            float er2 = sr1 + sr3, ei2 = si1 + si3;
            float er3 = sr1 - sr3, ei3 = si1 - si3;

            float* ob_r = re_b + 8 * k;
            float* ob_i = im_b + 8 * k;
            ob_r[0] = er0 + er2; ob_i[0] = ei0 + ei2;
            ob_r[4] = er0 - er2; ob_i[4] = ei0 - ei2;
            ob_r[2] = er1 + ei3; ob_i[2] = ei1 - er3;
            ob_r[6] = er1 - ei3; ob_i[6] = ei1 + er3;

            float tr0 = dr0, ti0 = di0;
            float tr1 = kSqrt2_2 * (dr1 + di1), ti1 = kSqrt2_2 * (di1 - dr1);
            float tr2 = di2, ti2 = -dr2;
            float tr3 = kSqrt2_2 * (di3 - dr3), ti3 = -kSqrt2_2 * (dr3 + di3);

            float or0 = tr0 + tr2, oi0 = ti0 + ti2;
            float or1 = tr0 - tr2, oi1 = ti0 - ti2;
            float or2 = tr1 + tr3, oi2 = ti1 + ti3;
            float or3 = tr1 - tr3, oi3 = ti1 - ti3;

            ob_r[1] = or0 + or2; ob_i[1] = oi0 + oi2;
            ob_r[5] = or0 - or2; ob_i[5] = oi0 - oi2;
            ob_r[3] = or1 + oi3; ob_i[3] = oi1 - or3;
            ob_r[7] = or1 - oi3; ob_i[7] = oi1 + or3;
        }
    }

    float* cur_re = re_b;
    float* cur_im = im_b;
    float* nxt_re = plan.re_a.ptr;
    float* nxt_im = plan.im_a.ptr;

    const int q1 = nfft / 4;
    const int q2 = nfft / 2;
    const int q3 = 3 * q1;

    for (const auto& st : plan.stages) {
        const int s = st.s;
        if (st.radix == 4) {
            const int rem = nfft / (4 * s);
            const float* wr1_base = st.wr.ptr;
            const float* wi1_base = st.wi.ptr;
            const float* wr2_base = wr1_base + s;
            const float* wi2_base = wi1_base + s;
            const float* wr3_base = wr2_base + s;
            const float* wi3_base = wi2_base + s;

            for (int k = 0; k < rem; ++k) {
                const int in_off = k * s;
                const int out_off = k * (4 * s);
                for (int j = 0; j < s; j += 8) {
                    __m256 ar = _mm256_load_ps(cur_re + in_off + j);
                    __m256 ai = _mm256_load_ps(cur_im + in_off + j);

                    __m256 br_raw = _mm256_load_ps(cur_re + in_off + q1 + j);
                    __m256 bi_raw = _mm256_load_ps(cur_im + in_off + q1 + j);
                    __m256 cr_raw = _mm256_load_ps(cur_re + in_off + q2 + j);
                    __m256 ci_raw = _mm256_load_ps(cur_im + in_off + q2 + j);
                    __m256 dr_raw = _mm256_load_ps(cur_re + in_off + q3 + j);
                    __m256 di_raw = _mm256_load_ps(cur_im + in_off + q3 + j);

                    __m256 wr1 = _mm256_load_ps(wr1_base + j), wi1 = _mm256_load_ps(wi1_base + j);
                    __m256 wr2 = _mm256_load_ps(wr2_base + j), wi2 = _mm256_load_ps(wi2_base + j);
                    __m256 wr3 = _mm256_load_ps(wr3_base + j), wi3 = _mm256_load_ps(wi3_base + j);

#if SIMD_HAS_FMA
                    __m256 br = _mm256_fmsub_ps(br_raw, wr1, _mm256_mul_ps(bi_raw, wi1));
                    __m256 bi = _mm256_fmadd_ps(br_raw, wi1, _mm256_mul_ps(bi_raw, wr1));
                    __m256 cr = _mm256_fmsub_ps(cr_raw, wr2, _mm256_mul_ps(ci_raw, wi2));
                    __m256 ci = _mm256_fmadd_ps(cr_raw, wi2, _mm256_mul_ps(ci_raw, wr2));
                    __m256 dr = _mm256_fmsub_ps(dr_raw, wr3, _mm256_mul_ps(di_raw, wi3));
                    __m256 di = _mm256_fmadd_ps(dr_raw, wi3, _mm256_mul_ps(di_raw, wr3));
#else
                    __m256 br = _mm256_sub_ps(_mm256_mul_ps(br_raw, wr1), _mm256_mul_ps(bi_raw, wi1));
                    __m256 bi = _mm256_add_ps(_mm256_mul_ps(br_raw, wi1), _mm256_mul_ps(bi_raw, wr1));
                    __m256 cr = _mm256_sub_ps(_mm256_mul_ps(cr_raw, wr2), _mm256_mul_ps(ci_raw, wi2));
                    __m256 ci = _mm256_add_ps(_mm256_mul_ps(cr_raw, wi2), _mm256_mul_ps(ci_raw, wr2));
                    __m256 dr = _mm256_sub_ps(_mm256_mul_ps(dr_raw, wr3), _mm256_mul_ps(di_raw, wi3));
                    __m256 di = _mm256_add_ps(_mm256_mul_ps(dr_raw, wi3), _mm256_mul_ps(di_raw, wr3));
#endif

                    __m256 t0r = _mm256_add_ps(ar, cr), t0i = _mm256_add_ps(ai, ci);
                    __m256 t1r = _mm256_sub_ps(ar, cr), t1i = _mm256_sub_ps(ai, ci);
                    __m256 t2r = _mm256_add_ps(br, dr), t2i = _mm256_add_ps(bi, di);
                    __m256 t3r = _mm256_sub_ps(br, dr), t3i = _mm256_sub_ps(bi, di);

                    _mm256_store_ps(nxt_re + out_off + j,         _mm256_add_ps(t0r, t2r));
                    _mm256_store_ps(nxt_im + out_off + j,         _mm256_add_ps(t0i, t2i));
                    _mm256_store_ps(nxt_re + out_off + s + j,     _mm256_add_ps(t1r, t3i));
                    _mm256_store_ps(nxt_im + out_off + s + j,     _mm256_sub_ps(t1i, t3r));
                    _mm256_store_ps(nxt_re + out_off + 2 * s + j, _mm256_sub_ps(t0r, t2r));
                    _mm256_store_ps(nxt_im + out_off + 2 * s + j, _mm256_sub_ps(t0i, t2i));
                    _mm256_store_ps(nxt_re + out_off + 3 * s + j, _mm256_sub_ps(t1r, t3i));
                    _mm256_store_ps(nxt_im + out_off + 3 * s + j, _mm256_add_ps(t1i, t3r));
                }
            }
        } else {
            // Trailing Radix-2 stage (s == nfft / 2)
            const float* wr_base = st.wr.ptr;
            const float* wi_base = st.wi.ptr;
            for (int j = 0; j < s; j += 8) {
                __m256 ar = _mm256_load_ps(cur_re + j);
                __m256 ai = _mm256_load_ps(cur_im + j);
                __m256 br_raw = _mm256_load_ps(cur_re + s + j);
                __m256 bi_raw = _mm256_load_ps(cur_im + s + j);
                __m256 wr = _mm256_load_ps(wr_base + j);
                __m256 wi = _mm256_load_ps(wi_base + j);
#if SIMD_HAS_FMA
                __m256 br = _mm256_fmsub_ps(br_raw, wr, _mm256_mul_ps(bi_raw, wi));
                __m256 bi = _mm256_fmadd_ps(br_raw, wi, _mm256_mul_ps(bi_raw, wr));
#else
                __m256 br = _mm256_sub_ps(_mm256_mul_ps(br_raw, wr), _mm256_mul_ps(bi_raw, wi));
                __m256 bi = _mm256_add_ps(_mm256_mul_ps(br_raw, wi), _mm256_mul_ps(bi_raw, wr));
#endif
                _mm256_store_ps(nxt_re + j,     _mm256_add_ps(ar, br));
                _mm256_store_ps(nxt_im + j,     _mm256_add_ps(ai, bi));
                _mm256_store_ps(nxt_re + s + j, _mm256_sub_ps(ar, br));
                _mm256_store_ps(nxt_im + s + j, _mm256_sub_ps(ai, bi));
            }
        }
        std::swap(cur_re, nxt_re);
        std::swap(cur_im, nxt_im);
    }

    // Re-interleave into interleaved complex buffer
    for (int i = 0; i + 8 <= nfft; i += 8) {
        __m256 vr = _mm256_load_ps(cur_re + i);
        __m256 vi = _mm256_load_ps(cur_im + i);
        __m256 lo = _mm256_unpacklo_ps(vr, vi);
        __m256 hi = _mm256_unpackhi_ps(vr, vi);
        _mm256_storeu_ps(cpx + 2 * i,     _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(cpx + 2 * i + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
    return true;
#else
    (void)cpx;
    (void)nfft;
    return false;
#endif
}

} // namespace SimdDsp
