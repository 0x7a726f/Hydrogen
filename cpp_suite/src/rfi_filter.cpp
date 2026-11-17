#include "rfi_filter.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace astro {

static double calc_median_inplace(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (n % 2 != 0) {
        return v[mid];
    } else {
        auto left_max = std::max_element(v.begin(), v.begin() + mid);
        return (*left_max + v[mid]) * 0.5;
    }
}

static inline double calc_median_15(const double* src) {
    double win[15];
    std::memcpy(win, src, 15 * sizeof(double));
    std::nth_element(win, win + 7, win + 15);
    return win[7];
}

// Unreplicated boundary window median over [max(0, center_idx - 7), min(nfft - 1, center_idx + 7)]
// Avoids edge replication where a bin-0 or bin-(nfft-1) spike would replicate itself 8 times and become its own median.
static inline double calc_median_boundary(const double* power, int nfft, int center_idx) {
    double win[15] = {};
    int lo = std::max(0, center_idx - 7);
    int hi = std::min(nfft - 1, center_idx + 7);
    int count = 0;
    for (int idx = lo; idx <= hi; ++idx) {
        win[count++] = power[idx];
    }
    int mid = count / 2;
    std::nth_element(win, win + mid, win + count);
    if (count % 2 != 0) return win[mid];
    double left_max = *std::max_element(win, win + mid);
    return 0.5 * (left_max + win[mid]);
}

#ifdef __AVX2__
static inline void avx2_cas(__m256d& a, __m256d& b) {
    __m256d lo = _mm256_min_pd(a, b);
    __m256d hi = _mm256_max_pd(a, b);
    a = lo;
    b = hi;
}

// 4-wide AVX2 Batcher Odd-Even Merge sorting network for 15-element median (output in lane index 7).
// Treats implicit 16th lane as +infinity (all compare-swaps with index 15 omitted).
static inline __m256d calc_median_15_avx2_4x(const double* src_minus_7) {
    __m256d w0  = _mm256_loadu_pd(src_minus_7 + 0);
    __m256d w1  = _mm256_loadu_pd(src_minus_7 + 1);
    __m256d w2  = _mm256_loadu_pd(src_minus_7 + 2);
    __m256d w3  = _mm256_loadu_pd(src_minus_7 + 3);
    __m256d w4  = _mm256_loadu_pd(src_minus_7 + 4);
    __m256d w5  = _mm256_loadu_pd(src_minus_7 + 5);
    __m256d w6  = _mm256_loadu_pd(src_minus_7 + 6);
    __m256d w7  = _mm256_loadu_pd(src_minus_7 + 7);
    __m256d w8  = _mm256_loadu_pd(src_minus_7 + 8);
    __m256d w9  = _mm256_loadu_pd(src_minus_7 + 9);
    __m256d w10 = _mm256_loadu_pd(src_minus_7 + 10);
    __m256d w11 = _mm256_loadu_pd(src_minus_7 + 11);
    __m256d w12 = _mm256_loadu_pd(src_minus_7 + 12);
    __m256d w13 = _mm256_loadu_pd(src_minus_7 + 13);
    __m256d w14 = _mm256_loadu_pd(src_minus_7 + 14);

    // Stage p=1, k=1
    avx2_cas(w0, w1);   avx2_cas(w2, w3);   avx2_cas(w4, w5);   avx2_cas(w6, w7);
    avx2_cas(w8, w9);   avx2_cas(w10, w11); avx2_cas(w12, w13);

    // Stage p=2, k=2 then k=1
    avx2_cas(w0, w2);   avx2_cas(w1, w3);   avx2_cas(w4, w6);   avx2_cas(w5, w7);
    avx2_cas(w8, w10);  avx2_cas(w9, w11);  avx2_cas(w12, w14);
    avx2_cas(w1, w2);   avx2_cas(w5, w6);   avx2_cas(w9, w10);  avx2_cas(w13, w14);

    // Stage p=4, k=4, k=2, k=1
    avx2_cas(w0, w4);   avx2_cas(w1, w5);   avx2_cas(w2, w6);   avx2_cas(w3, w7);
    avx2_cas(w8, w12);  avx2_cas(w9, w13);  avx2_cas(w10, w14);
    avx2_cas(w2, w4);   avx2_cas(w3, w5);   avx2_cas(w10, w12); avx2_cas(w11, w13);
    avx2_cas(w1, w2);   avx2_cas(w3, w4);   avx2_cas(w5, w6);
    avx2_cas(w9, w10);  avx2_cas(w11, w12); avx2_cas(w13, w14);

    // Final merge p=8 (pruned to wires affecting index 7)
    avx2_cas(w0, w8);   avx2_cas(w1, w9);   avx2_cas(w2, w10);  avx2_cas(w3, w11);
    avx2_cas(w4, w12);  avx2_cas(w5, w13);  avx2_cas(w6, w14);
    avx2_cas(w4, w8);   avx2_cas(w5, w9);   avx2_cas(w6, w10);  avx2_cas(w7, w11);
    avx2_cas(w2, w4);   avx2_cas(w3, w5);   avx2_cas(w6, w8);   avx2_cas(w7, w9);
    avx2_cas(w10, w12); avx2_cas(w11, w13);
    // For k=1 in final stage, only (7, 8) modifies element 7
    __m256d lo7 = _mm256_min_pd(w7, w8);
    return lo7;
}
#endif

std::vector<double> RFIFilter::mitigate(
    const std::vector<double>& power,
    const std::vector<double>& freqs_mhz,
    const AstroConfig& config,
    int& out_num_rfi
) {
    int nfft = static_cast<int>(power.size());
    std::vector<double> clean_power = power;
    out_num_rfi = 0;
    if (nfft < 32) return clean_power;

    double hi_half_width_mhz = config.hi_protection_half_width_khz / 1e3;
    int edge_bins = static_cast<int>(nfft * std::clamp(config.passband_edge_fraction, 0.0, 0.45));

    std::vector<uint8_t> valid_mask(nfft, 1);
    for (int i = 0; i < edge_bins; ++i) valid_mask[i] = 0;
    for (int i = nfft - edge_bins; i < nfft; ++i) valid_mask[i] = 0;

    for (int i = 0; i < nfft; ++i) {
        if (freqs_mhz[i] >= (HYDROGEN_REST_FREQ_MHZ - hi_half_width_mhz) &&
            freqs_mhz[i] <= (HYDROGEN_REST_FREQ_MHZ + hi_half_width_mhz)) {
            valid_mask[i] = 0; // Protected HI window
        }
    }

    // Rolling median filter (kernel size = 15) directly from clean_power (zero full-array pad copy, zero per-call scratch allocs)
    thread_local std::vector<double> tl_residuals;
    thread_local std::vector<double> tl_residuals_valid;
    thread_local std::vector<int> tl_rfi_indices;

    tl_residuals.resize(static_cast<size_t>(nfft));
    double* res_ptr = tl_residuals.data();
    const double* pwr_ptr = clean_power.data();

    // Left boundary [0..6]
    for (int i = 0; i < 7; ++i) {
        res_ptr[i] = pwr_ptr[i] - calc_median_boundary(pwr_ptr, nfft, i);
    }
    int i = 7;
#ifdef __AVX2__
    // Interior [7 .. nfft - 8] processed 4 bins at a time using Batcher Odd-Even Merge sorting network
    for (; i + 11 <= nfft; i += 4) {
        __m256d med = calc_median_15_avx2_4x(pwr_ptr + i - 7);
        __m256d val = _mm256_loadu_pd(pwr_ptr + i);
        _mm256_storeu_pd(res_ptr + i, _mm256_sub_pd(val, med));
    }
#endif
    for (; i <= nfft - 8; ++i) {
        res_ptr[i] = pwr_ptr[i] - calc_median_15(pwr_ptr + i - 7);
    }
    // Right boundary [nfft - 7 .. nfft - 1]
    for (; i < nfft; ++i) {
        res_ptr[i] = pwr_ptr[i] - calc_median_boundary(pwr_ptr, nfft, i);
    }

    tl_residuals_valid.clear();
    if (tl_residuals_valid.capacity() < static_cast<size_t>(nfft)) {
        tl_residuals_valid.reserve(static_cast<size_t>(nfft));
    }
    for (int i = 0; i < nfft; ++i) {
        if (valid_mask[i]) {
            tl_residuals_valid.push_back(res_ptr[i]);
        }
    }

    if (tl_residuals_valid.empty()) {
        out_num_rfi = 0;
        return clean_power;
    }

    double med_res = calc_median_inplace(tl_residuals_valid);
    for (double& r : tl_residuals_valid) {
        r = std::abs(r - med_res);
    }
    double mad = calc_median_inplace(tl_residuals_valid);
    double sigma_est = std::max(1e-12, 1.4826 * mad);

    double threshold = config.rfi_threshold_mad * sigma_est;
    thread_local std::vector<uint8_t> is_rfi;
    is_rfi.assign(nfft, 0);
    tl_rfi_indices.clear();
    for (int i = 0; i < nfft; ++i) {
        if (valid_mask[i] && (res_ptr[i] - med_res) > threshold) {
            is_rfi[i] = 1;
            tl_rfi_indices.push_back(i);
        }
    }

    out_num_rfi = static_cast<int>(tl_rfi_indices.size());
    for (size_t k = 0; k < tl_rfi_indices.size();) {
        const int run_start = tl_rfi_indices[k];
        int run_end = run_start;
        while (k + 1 < tl_rfi_indices.size() && tl_rfi_indices[k + 1] == run_end + 1) {
            run_end = tl_rfi_indices[++k];
        }
        ++k;

        int l_idx = run_start - 1;
        while (l_idx >= 0 && is_rfi[l_idx]) --l_idx;
        int r_idx = run_end + 1;
        while (r_idx < nfft && is_rfi[r_idx]) ++r_idx;

        const bool has_left = (l_idx >= 0);
        const bool has_right = (r_idx < nfft);
        if (!has_left && !has_right) continue;

        const double v_l = has_left ? power[l_idx] : power[r_idx];
        const double v_r = has_right ? power[r_idx] : power[l_idx];
        const int span_l = has_left ? l_idx : run_start - 1;
        const int span_r = has_right ? r_idx : run_end + 1;
        const int span = std::max(1, span_r - span_l);
        for (int idx = run_start; idx <= run_end; ++idx) {
            const double t = static_cast<double>(idx - span_l) / static_cast<double>(span);
            clean_power[idx] = v_l + t * (v_r - v_l);
        }
    }

    return clean_power;
}

} // namespace astro
