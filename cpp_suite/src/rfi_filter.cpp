#include "rfi_filter.h"
#include "simd_dsp.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace astro {

static double calc_median_inplace(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    if (n % 2 != 0) {
        return v[mid];
    }
    auto left_max = std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (*left_max + v[mid]);
}

static inline double calc_median_15(const double* src) {
    double win[15];
    std::memcpy(win, src, 15 * sizeof(double));
    std::nth_element(win, win + 7, win + 15);
    return win[7];
}

// Unreplicated boundary window median over [max(0, center_idx - 7), min(nfft - 1, center_idx + 7)]
// Avoids edge replication where a bin-0 or bin-(nfft-1) spike would replicate itself 8 times and become its own median.
static inline double calc_median_boundary(const double* src, int nfft, int center_idx) {
    double win[15] = {};
    const int lo = std::max(0, center_idx - 7);
    const int hi = std::min(nfft - 1, center_idx + 7);
    int count = 0;
    for (int idx = lo; idx <= hi; ++idx) {
        win[count++] = src[idx];
    }
    const int mid = count / 2;
    std::nth_element(win, win + mid, win + count);
    if (count % 2 != 0) return win[mid];
    const double left_max = *std::max_element(win, win + mid);
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
    const int nfft = static_cast<int>(power.size());
    std::vector<double> clean_power = power;
    out_num_rfi = 0;
    if (nfft < 32 || static_cast<int>(freqs_mhz.size()) != nfft) return clean_power;

    const double hi_half_width_mhz = config.hi_protection_half_width_khz / 1e3;
    const int edge_bins = static_cast<int>(nfft * std::clamp(config.passband_edge_fraction, 0.0, 0.45));

    thread_local std::vector<uint8_t> passband_mask;
    thread_local std::vector<uint8_t> hi_window;
    thread_local std::vector<uint8_t> valid_mask;
    passband_mask.assign(static_cast<size_t>(nfft), 1);
    hi_window.assign(static_cast<size_t>(nfft), 0);
    valid_mask.assign(static_cast<size_t>(nfft), 1);

    for (int i = 0; i < edge_bins; ++i) {
        passband_mask[static_cast<size_t>(i)] = 0;
        valid_mask[static_cast<size_t>(i)] = 0;
    }
    for (int i = nfft - edge_bins; i < nfft; ++i) {
        passband_mask[static_cast<size_t>(i)] = 0;
        valid_mask[static_cast<size_t>(i)] = 0;
    }

    const double hi_lo = HYDROGEN_REST_FREQ_MHZ - hi_half_width_mhz;
    const double hi_hi = HYDROGEN_REST_FREQ_MHZ + hi_half_width_mhz;
    for (int i = 0; i < nfft; ++i) {
        if (freqs_mhz[static_cast<size_t>(i)] >= hi_lo &&
            freqs_mhz[static_cast<size_t>(i)] <= hi_hi) {
            hi_window[static_cast<size_t>(i)] = 1;
            valid_mask[static_cast<size_t>(i)] = 0; // Protected H I window
        }
    }

    // Transform power to homoscedastic log/dB space so passband gain variation (2-10x across
    // crests/troughs) yields uniform noise dispersion everywhere.
    thread_local std::vector<double> tl_p_db;
    thread_local std::vector<double> tl_residuals;
    thread_local std::vector<double> tl_residuals_valid;
    thread_local std::vector<uint8_t> core_rfi;
    thread_local std::vector<uint8_t> is_rfi;
    thread_local std::vector<int> tl_rfi_indices;

    tl_p_db.resize(static_cast<size_t>(nfft));
    tl_residuals.resize(static_cast<size_t>(nfft));
    SimdDsp::power_to_db_f64(power.data(), tl_p_db.data(), static_cast<size_t>(nfft), 1e-30);

    const double* pdb_ptr = tl_p_db.data();
    double* res_ptr = tl_residuals.data();

    // Rolling median-15 filter in dB space (AVX2 Batcher sorting network for interior)
    for (int i = 0; i < 7; ++i) {
        res_ptr[i] = pdb_ptr[i] - calc_median_boundary(pdb_ptr, nfft, i);
    }
    int i = 7;
#ifdef __AVX2__
    for (; i + 11 <= nfft; i += 4) {
        __m256d med = calc_median_15_avx2_4x(pdb_ptr + i - 7);
        __m256d val = _mm256_loadu_pd(pdb_ptr + i);
        _mm256_storeu_pd(res_ptr + i, _mm256_sub_pd(val, med));
    }
#endif
    for (; i <= nfft - 8; ++i) {
        res_ptr[i] = pdb_ptr[i] - calc_median_15(pdb_ptr + i - 7);
    }
    for (; i < nfft; ++i) {
        res_ptr[i] = pdb_ptr[i] - calc_median_boundary(pdb_ptr, nfft, i);
    }

    // Estimate homoscedastic thermal noise dispersion via MAD on valid continuum bins
    tl_residuals_valid.clear();
    if (tl_residuals_valid.capacity() < static_cast<size_t>(nfft)) {
        tl_residuals_valid.reserve(static_cast<size_t>(nfft));
    }
    for (int idx = 0; idx < nfft; ++idx) {
        if (valid_mask[static_cast<size_t>(idx)]) {
            tl_residuals_valid.push_back(res_ptr[idx]);
        }
    }

    if (tl_residuals_valid.empty()) {
        out_num_rfi = 0;
        return clean_power;
    }

    const double med_res = calc_median_inplace(tl_residuals_valid);
    for (double& r : tl_residuals_valid) {
        r = std::abs(r - med_res);
    }
    const double mad = calc_median_inplace(tl_residuals_valid);
    // Floor at 1e-4 dB so smooth noiseless synthetic passband curvature cannot false-trigger
    const double sigma_est = std::max(1e-4, 1.4826 * mad);

    const double threshold = std::max(0.0, config.rfi_threshold_mad) * sigma_est;
    core_rfi.assign(static_cast<size_t>(nfft), 0);
    is_rfi.assign(static_cast<size_t>(nfft), 0);

    // 1. Flag continuum RFI outliers strictly inside valid_mask
    for (int idx = 0; idx < nfft; ++idx) {
        if (valid_mask[static_cast<size_t>(idx)] && (res_ptr[idx] - med_res) > threshold) {
            core_rfi[static_cast<size_t>(idx)] = 1;
            is_rfi[static_cast<size_t>(idx)] = 1;
        }
    }

    // 2. Dilate flagged continuum RFI runs by +/- 1 bin (strictly within valid_mask)
    //    to capture FFT window mainlobe/sidelobe skirts without bleeding into hi_window.
    for (int idx = 0; idx < nfft; ++idx) {
        if (!core_rfi[static_cast<size_t>(idx)]) continue;
        if (idx - 1 >= 0 && valid_mask[static_cast<size_t>(idx - 1)]) {
            is_rfi[static_cast<size_t>(idx - 1)] = 1;
        }
        if (idx + 1 < nfft && valid_mask[static_cast<size_t>(idx + 1)]) {
            is_rfi[static_cast<size_t>(idx + 1)] = 1;
        }
    }

    // 3. Inside hi_window (within passband), protect all broad thermal H I emission,
    //    but excise extreme narrow 1-2 bin CW birdies so single-bin spurs cannot corrupt line fits.
    const double cw_thresh_db = std::max(12.0 * sigma_est, 6.0);
    for (int idx = 0; idx < nfft; ) {
        if (passband_mask[static_cast<size_t>(idx)] &&
            hi_window[static_cast<size_t>(idx)] &&
            (res_ptr[idx] - med_res) > cw_thresh_db) {
            const int run_start = idx;
            int run_end = idx;
            while (run_end + 1 < nfft &&
                   passband_mask[static_cast<size_t>(run_end + 1)] &&
                   hi_window[static_cast<size_t>(run_end + 1)] &&
                   (res_ptr[run_end + 1] - med_res) > cw_thresh_db) {
                ++run_end;
            }
            const int run_width = run_end - run_start + 1;
            if (run_width <= 2) {
                for (int k = run_start; k <= run_end; ++k) {
                    is_rfi[static_cast<size_t>(k)] = 1;
                }
            }
            idx = run_end + 1;
        } else {
            ++idx;
        }
    }

    // 4. Collect all flagged indices and linearly interpolate across each contiguous run in dB space
    tl_rfi_indices.clear();
    if (tl_rfi_indices.capacity() < static_cast<size_t>(nfft)) {
        tl_rfi_indices.reserve(static_cast<size_t>(nfft));
    }
    for (int idx = 0; idx < nfft; ++idx) {
        if (is_rfi[static_cast<size_t>(idx)]) {
            tl_rfi_indices.push_back(idx);
        }
    }

    out_num_rfi = static_cast<int>(tl_rfi_indices.size());
    for (size_t k = 0; k < tl_rfi_indices.size(); ) {
        const int run_start = tl_rfi_indices[k];
        int run_end = run_start;
        while (k + 1 < tl_rfi_indices.size() && tl_rfi_indices[k + 1] == run_end + 1) {
            run_end = tl_rfi_indices[++k];
        }
        ++k;

        int l_idx = run_start - 1;
        while (l_idx >= 0 && is_rfi[static_cast<size_t>(l_idx)]) --l_idx;
        int r_idx = run_end + 1;
        while (r_idx < nfft && is_rfi[static_cast<size_t>(r_idx)]) ++r_idx;

        const bool has_left = (l_idx >= 0);
        const bool has_right = (r_idx < nfft);
        if (!has_left && !has_right) continue;

        if (!has_left) {
            const double fill_p = power[static_cast<size_t>(r_idx)];
            for (int idx = run_start; idx <= run_end; ++idx) {
                clean_power[static_cast<size_t>(idx)] = fill_p;
            }
        } else if (!has_right) {
            const double fill_p = power[static_cast<size_t>(l_idx)];
            for (int idx = run_start; idx <= run_end; ++idx) {
                clean_power[static_cast<size_t>(idx)] = fill_p;
            }
        } else {
            const double db_l = pdb_ptr[l_idx];
            const double db_r = pdb_ptr[r_idx];
            const double inv_span = 1.0 / static_cast<double>(r_idx - l_idx);
            for (int idx = run_start; idx <= run_end; ++idx) {
                const double t = static_cast<double>(idx - l_idx) * inv_span;
                const double interp_db = db_l + t * (db_r - db_l);
                clean_power[static_cast<size_t>(idx)] = std::pow(10.0, interp_db * 0.1);
            }
        }
    }

    return clean_power;
}

} // namespace astro
