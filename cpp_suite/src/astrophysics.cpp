#include "astrophysics.h"
#include "simd_dsp.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace astro {

std::vector<double> Astrophysics::compute_doppler_velocity(const std::vector<double>& freqs_hz) {
    size_t n = freqs_hz.size();
    std::vector<double> v(n);
    constexpr double inv_rest_c = SPEED_OF_LIGHT_KMS / HYDROGEN_REST_FREQ_HZ;
    for (size_t i = 0; i < n; ++i) {
        v[i] = (HYDROGEN_REST_FREQ_HZ - freqs_hz[i]) * inv_rest_c;
    }
    return v;
}

// Invert 3x3 symmetric matrix A * inv = I
static bool invert_3x3(const double A[3][3], double inv[3][3]) {
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    double scale = std::max(1.0, std::max({std::abs(A[0][0]), std::abs(A[1][1]), std::abs(A[2][2])}));
    if (std::abs(det) < 1e-18 * scale) return false;
    double inv_det = 1.0 / det;

    inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
    inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;

    inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
    inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;

    inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;
    inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;
    inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

    return true;
}

GaussianFit Astrophysics::fit_gaussian(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<bool>& hi_window
) {
    std::vector<uint8_t> mask(hi_window.size(), 0);
    for (size_t i = 0; i < hi_window.size(); ++i) {
        mask[i] = hi_window[i] ? 1 : 0;
    }
    return fit_gaussian(vel_kms, temp_k, mask);
}

GaussianFit Astrophysics::fit_gaussian(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window
) {
    GaussianFit res;
    size_t n_all = vel_kms.size();
    res.fitted_curve.assign(n_all, 0.0);

    // Extract subset inside HI window
    std::vector<double> v_sub, t_sub;
    v_sub.reserve(n_all);
    t_sub.reserve(n_all);
    for (size_t i = 0; i < n_all; ++i) {
        if (hi_window[i]) {
            v_sub.push_back(vel_kms[i]);
            t_sub.push_back(temp_k[i]);
        }
    }

    if (v_sub.size() < 5) return res;

    // Initial non-parametric estimate
    size_t max_idx = 0;
    double max_t = t_sub[0];
    for (size_t i = 1; i < t_sub.size(); ++i) {
        if (t_sub[i] > max_t) {
            max_t = t_sub[i];
            max_idx = i;
        }
    }

    // Handle weak/negative peak or no emission: fallback gracefully
    if (max_t <= 0.0) {
        res.success = false;
        res.amplitude_k = 0.0;
        res.v_center_kms = v_sub[max_idx];
        res.fwhm_kms = 25.0;
        res.sigma_kms = 25.0 / GAUSSIAN_SIGMA_TO_FWHM;
        return res;
    }

    double a_init = max_t;
    double v0_init = v_sub[max_idx];
    double half_max = a_init * 0.5;

    // Measure contiguous FWHM around the primary peak (avoids disconnected outer noise spikes)
    size_t left_idx = max_idx;
    while (left_idx > 0 && t_sub[left_idx - 1] >= half_max) {
        --left_idx;
    }
    size_t right_idx = max_idx;
    while (right_idx + 1 < t_sub.size() && t_sub[right_idx + 1] >= half_max) {
        ++right_idx;
    }
    double dv_sub = (t_sub.size() > 1) ? std::abs(v_sub[1] - v_sub[0]) : 1.0;
    double span_half = std::abs(v_sub[right_idx] - v_sub[left_idx]);
    double fwhm_init = (right_idx > left_idx) ? std::max(span_half, dv_sub) : 25.0;
    double sigma_init = std::max(0.5, fwhm_init / GAUSSIAN_SIGMA_TO_FWHM);

    double a = a_init;
    double v0 = v0_init;
    double sigma = sigma_init;

    // Levenberg-Marquardt optimizer for 3 parameters: p = [a, v0, sigma]
    double lambda = 1.0;
    size_t m = v_sub.size();

    auto eval_chi2 = [&](double cur_a, double cur_v0, double cur_sig) {
        double sum_sq = 0.0;
        const double neg_half_inv_sig2 = -0.5 / (cur_sig * cur_sig);
        size_t i = 0;
#if defined(__AVX2__)
        const __m256d v_a = _mm256_set1_pd(cur_a);
        const __m256d v_v0 = _mm256_set1_pd(cur_v0);
        const __m256d v_c = _mm256_set1_pd(neg_half_inv_sig2);
        __m256d v_sum0 = _mm256_setzero_pd();
        __m256d v_sum1 = _mm256_setzero_pd();
        for (; i + 8 <= m; i += 8) {
            __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(v_sub.data() + i), v_v0);
            __m256d d1 = _mm256_sub_pd(_mm256_loadu_pd(v_sub.data() + i + 4), v_v0);
            __m256d e0 = SimdDsp::fast_exp_pd_avx2(_mm256_mul_pd(v_c, _mm256_mul_pd(d0, d0)));
            __m256d e1 = SimdDsp::fast_exp_pd_avx2(_mm256_mul_pd(v_c, _mm256_mul_pd(d1, d1)));
            __m256d r0 = _mm256_sub_pd(_mm256_loadu_pd(t_sub.data() + i), _mm256_mul_pd(v_a, e0));
            __m256d r1 = _mm256_sub_pd(_mm256_loadu_pd(t_sub.data() + i + 4), _mm256_mul_pd(v_a, e1));
#if SIMD_HAS_FMA
            v_sum0 = _mm256_fmadd_pd(r0, r0, v_sum0);
            v_sum1 = _mm256_fmadd_pd(r1, r1, v_sum1);
#else
            v_sum0 = _mm256_add_pd(v_sum0, _mm256_mul_pd(r0, r0));
            v_sum1 = _mm256_add_pd(v_sum1, _mm256_mul_pd(r1, r1));
#endif
        }
        for (; i + 4 <= m; i += 4) {
            __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(v_sub.data() + i), v_v0);
            __m256d e0 = SimdDsp::fast_exp_pd_avx2(_mm256_mul_pd(v_c, _mm256_mul_pd(d0, d0)));
            __m256d r0 = _mm256_sub_pd(_mm256_loadu_pd(t_sub.data() + i), _mm256_mul_pd(v_a, e0));
#if SIMD_HAS_FMA
            v_sum0 = _mm256_fmadd_pd(r0, r0, v_sum0);
#else
            v_sum0 = _mm256_add_pd(v_sum0, _mm256_mul_pd(r0, r0));
#endif
        }
        __m256d v_tot = _mm256_add_pd(v_sum0, v_sum1);
        alignas(32) double acc_arr[4];
        _mm256_store_pd(acc_arr, v_tot);
        sum_sq = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];
#endif
        for (; i < m; ++i) {
            double diff = v_sub[i] - cur_v0;
            double model = cur_a * std::exp(neg_half_inv_sig2 * diff * diff);
            double r = t_sub[i] - model;
            sum_sq += r * r;
        }
        return sum_sq;
    };

    double initial_chi2 = eval_chi2(a, v0, sigma);
    double current_chi2 = initial_chi2;
    const double max_a_bound = std::max(10.0, a_init * 2.5 + 5.0);

    bool converged = false;
    int successful_steps = 0;

    // Cache Jacobian normal equations when a trial step is rejected (parameters unchanged)
    bool jacobian_valid = false;
    double j00 = 0.0, j01 = 0.0, j02 = 0.0, j11 = 0.0, j12 = 0.0, j22 = 0.0;
    double g0 = 0.0, g1 = 0.0, g2 = 0.0;

    for (int iter = 0; iter < 100; ++iter) {
        if (!jacobian_valid) {
            j00 = j01 = j02 = j11 = j12 = j22 = 0.0;
            g0 = g1 = g2 = 0.0;

            const double inv_sig = 1.0 / sigma;
            const double inv_sig2 = inv_sig * inv_sig;
            const double inv_sig3 = inv_sig2 * inv_sig;
            const double neg_half_inv_sig2 = -0.5 * inv_sig2;
            const double a_inv_sig2 = a * inv_sig2;
            const double a_inv_sig3 = a * inv_sig3;

            size_t i = 0;
#if defined(__AVX2__)
            const __m256d v_a = _mm256_set1_pd(a);
            const __m256d v_v0 = _mm256_set1_pd(v0);
            const __m256d v_c = _mm256_set1_pd(neg_half_inv_sig2);
            const __m256d v_ainv2 = _mm256_set1_pd(a_inv_sig2);
            const __m256d v_ainv3 = _mm256_set1_pd(a_inv_sig3);

            __m256d vg0 = _mm256_setzero_pd(), vg1 = _mm256_setzero_pd(), vg2 = _mm256_setzero_pd();
            __m256d vj00 = _mm256_setzero_pd(), vj01 = _mm256_setzero_pd(), vj02 = _mm256_setzero_pd();
            __m256d vj11 = _mm256_setzero_pd(), vj12 = _mm256_setzero_pd(), vj22 = _mm256_setzero_pd();

            for (; i + 4 <= m; i += 4) {
                __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(v_sub.data() + i), v_v0);
                __m256d diff2 = _mm256_mul_pd(diff, diff);
                __m256d exp_val = SimdDsp::fast_exp_pd_avx2(_mm256_mul_pd(v_c, diff2));
                __m256d r = _mm256_sub_pd(_mm256_loadu_pd(t_sub.data() + i), _mm256_mul_pd(v_a, exp_val));

                __m256d d0 = exp_val;
                __m256d d1 = _mm256_mul_pd(_mm256_mul_pd(v_ainv2, diff), exp_val);
                __m256d d2 = _mm256_mul_pd(_mm256_mul_pd(v_ainv3, diff2), exp_val);

#if SIMD_HAS_FMA
                vg0 = _mm256_fmadd_pd(d0, r, vg0);
                vg1 = _mm256_fmadd_pd(d1, r, vg1);
                vg2 = _mm256_fmadd_pd(d2, r, vg2);
                vj00 = _mm256_fmadd_pd(d0, d0, vj00);
                vj01 = _mm256_fmadd_pd(d0, d1, vj01);
                vj02 = _mm256_fmadd_pd(d0, d2, vj02);
                vj11 = _mm256_fmadd_pd(d1, d1, vj11);
                vj12 = _mm256_fmadd_pd(d1, d2, vj12);
                vj22 = _mm256_fmadd_pd(d2, d2, vj22);
#else
                vg0 = _mm256_add_pd(vg0, _mm256_mul_pd(d0, r));
                vg1 = _mm256_add_pd(vg1, _mm256_mul_pd(d1, r));
                vg2 = _mm256_add_pd(vg2, _mm256_mul_pd(d2, r));
                vj00 = _mm256_add_pd(vj00, _mm256_mul_pd(d0, d0));
                vj01 = _mm256_add_pd(vj01, _mm256_mul_pd(d0, d1));
                vj02 = _mm256_add_pd(vj02, _mm256_mul_pd(d0, d2));
                vj11 = _mm256_add_pd(vj11, _mm256_mul_pd(d1, d1));
                vj12 = _mm256_add_pd(vj12, _mm256_mul_pd(d1, d2));
                vj22 = _mm256_add_pd(vj22, _mm256_mul_pd(d2, d2));
#endif
            }
            auto hsum = [](__m256d v) {
                alignas(32) double tmp[4];
                _mm256_store_pd(tmp, v);
                return tmp[0] + tmp[1] + tmp[2] + tmp[3];
            };
            g0 = hsum(vg0); g1 = hsum(vg1); g2 = hsum(vg2);
            j00 = hsum(vj00); j01 = hsum(vj01); j02 = hsum(vj02);
            j11 = hsum(vj11); j12 = hsum(vj12); j22 = hsum(vj22);
#endif
            for (; i < m; ++i) {
                double diff = v_sub[i] - v0;
                double diff2 = diff * diff;
                double exp_val = std::exp(neg_half_inv_sig2 * diff2);
                double r = t_sub[i] - a * exp_val;

                double d0 = exp_val;
                double d1 = a_inv_sig2 * diff * exp_val;
                double d2 = a_inv_sig3 * diff2 * exp_val;

                g0 += d0 * r;
                g1 += d1 * r;
                g2 += d2 * r;
                j00 += d0 * d0;
                j01 += d0 * d1;
                j02 += d0 * d2;
                j11 += d1 * d1;
                j12 += d1 * d2;
                j22 += d2 * d2;
            }
            jacobian_valid = true;
        }

        double A_damped[3][3] = {
            { j00 + lambda * (j00 + 1e-3), j01, j02 },
            { j01, j11 + lambda * (j11 + 1e-3), j12 },
            { j02, j12, j22 + lambda * (j22 + 1e-3) }
        };

        double inv_A[3][3];
        if (!invert_3x3(A_damped, inv_A)) {
            if (lambda >= 1e5) break;
            lambda = std::min(1e5, lambda * 4.0);
            continue;
        }

        double delta[3] = {
            inv_A[0][0] * g0 + inv_A[0][1] * g1 + inv_A[0][2] * g2,
            inv_A[1][0] * g0 + inv_A[1][1] * g1 + inv_A[1][2] * g2,
            inv_A[2][0] * g0 + inv_A[2][1] * g1 + inv_A[2][2] * g2
        };

        // Trust-region step limiter to maintain numerical stability across disparate parameter scales
        double step_scale = 1.0;
        if (std::abs(delta[0]) > 0.5 * a) step_scale = std::min(step_scale, (0.5 * a) / std::abs(delta[0]));
        if (std::abs(delta[1]) > 5.0) step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
        if (std::abs(delta[2]) > 0.5 * sigma) step_scale = std::min(step_scale, (0.5 * sigma) / std::abs(delta[2]));

        delta[0] *= step_scale;
        delta[1] *= step_scale;
        delta[2] *= step_scale;

        double new_a = std::clamp(a + delta[0], 0.04, max_a_bound);
        double new_v0 = std::clamp(v0 + delta[1], -250.0, 250.0);
        double new_sig = std::clamp(sigma + delta[2], 0.3, 100.0);

        double new_chi2 = eval_chi2(new_a, new_v0, new_sig);
        if (new_chi2 < current_chi2) {
            current_chi2 = new_chi2;
            a = new_a;
            v0 = new_v0;
            sigma = new_sig;
            jacobian_valid = false;
            successful_steps++;
            lambda = std::max(1e-7, lambda * 0.5);
            if (std::abs(delta[0]) < 1e-4 && std::abs(delta[1]) < 1e-4 && std::abs(delta[2]) < 1e-4) {
                converged = true;
                break; // Converged
            }
        } else {
            if (lambda >= 1e5) break; // Saturated at maximum damping
            lambda = std::min(1e5, lambda * 3.0);
        }
    }

    // Determine fit validity: must improve chi2 and stay off degenerate boundaries
    bool fit_valid = (successful_steps > 0 || converged) &&
                     (current_chi2 <= initial_chi2) &&
                     (a > 0.05 && a < max_a_bound * 0.99) &&
                     (sigma > 0.35 && sigma < 98.0);

    if (fit_valid) {
        res.success = true;
        res.method = LineAnalysisMethod::GaussianLS;
        res.amplitude_k = a;
        res.v_center_kms = v0;
        res.sigma_kms = std::abs(sigma);
        res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * res.sigma_kms;
        res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
        res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
        res.fwhm_half_temp_k = 0.5 * res.amplitude_k;
        res.rmse_k = std::sqrt(current_chi2 / static_cast<double>(std::max<size_t>(1, m)));

        const double neg_half_inv_sig2 = -0.5 / (sigma * sigma);
        size_t i = 0;
#if defined(__AVX2__)
        const __m256d v_a = _mm256_set1_pd(a);
        const __m256d v_v0 = _mm256_set1_pd(v0);
        const __m256d v_c = _mm256_set1_pd(neg_half_inv_sig2);
        for (; i + 4 <= n_all; i += 4) {
            __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(vel_kms.data() + i), v_v0);
            __m256d val = _mm256_mul_pd(v_a, SimdDsp::fast_exp_pd_avx2(_mm256_mul_pd(v_c, _mm256_mul_pd(diff, diff))));
            _mm256_storeu_pd(res.fitted_curve.data() + i, val);
        }
#endif
        for (; i < n_all; ++i) {
            double diff = vel_kms[i] - v0;
            res.fitted_curve[i] = a * std::exp(neg_half_inv_sig2 * diff * diff);
        }
    } else {
        // Graceful non-parametric fallback
        res.success = false;
        res.method = LineAnalysisMethod::GaussianLS;
        res.amplitude_k = a_init;
        res.v_center_kms = v0_init;
        res.sigma_kms = sigma_init;
        res.fwhm_kms = fwhm_init;
        res.fwhm_left_kms = v0_init - 0.5 * fwhm_init;
        res.fwhm_right_kms = v0_init + 0.5 * fwhm_init;
        res.fwhm_half_temp_k = 0.5 * a_init;
        std::fill(res.fitted_curve.begin(), res.fitted_curve.end(), 0.0);
    }

    return res;
}

// Generic NxN linear system solver (N <= 9) with partial pivoting for multi-parameter Levenberg-Marquardt
static bool solve_linear_nxn(int n, double A[9][9], const double b[9], double x[9]) {
    double M[9][10] = {};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n] = b[i];
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double max_v = std::abs(M[col][col]);
        for (int r = col + 1; r < n; ++r) {
            if (std::abs(M[r][col]) > max_v) {
                max_v = std::abs(M[r][col]);
                piv = r;
            }
        }
        if (max_v < 1e-15) return false;
        if (piv != col) {
            for (int c = col; c <= n; ++c) std::swap(M[col][c], M[piv][c]);
        }
        const double inv_diag = 1.0 / M[col][col];
        for (int r = col + 1; r < n; ++r) {
            const double factor = M[r][col] * inv_diag;
            for (int c = col; c <= n; ++c) {
                M[r][c] -= factor * M[col][c];
            }
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = M[i][n];
        for (int j = i + 1; j < n; ++j) sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
    return true;
}

static void populate_nonparametric_moments(
    const std::vector<double>& v_sub,
    const std::vector<double>& t_sub,
    GaussianFit& fit
) {
    const size_t n = v_sub.size();
    if (n < 3) return;
    const double dv = std::abs(v_sub[1] - v_sub[0]);

    // Unbiased integrated flux M0 (signed integration preserves E[M0] = 0 on zero-mean noise)
    double m0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m0 += t_sub[i] * dv;
    }
    fit.moment0_k_kms = m0;

    // Estimate channel noise via lag-1 difference MAD: sigma = 1.4826 * median(|dt|) / sqrt(2)
    std::vector<double> diffs;
    diffs.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
        diffs.push_back(std::abs(t_sub[i] - t_sub[i - 1]));
    }
    const size_t mid = diffs.size() / 2;
    std::nth_element(diffs.begin(), diffs.begin() + mid, diffs.end());
    const double sigma_est = std::max(1e-6, 1.4826 * diffs[mid] * 0.7071067811865475);

    // Compute M1 and M2 over signal-bearing channels (t_sub > 1.5 * sigma_est)
    // or fallback to contiguous positive core around the peak.
    double m_sig = 0.0;
    double m1_num = 0.0;
    size_t sig_bins = 0;
    for (size_t i = 0; i < n; ++i) {
        if (t_sub[i] > 1.5 * sigma_est) {
            m_sig += t_sub[i] * dv;
            m1_num += t_sub[i] * v_sub[i] * dv;
            ++sig_bins;
        }
    }

    if (sig_bins >= 3 && m_sig > 1e-9) {
        fit.moment1_kms = m1_num / m_sig;
        double m2_num = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (t_sub[i] > 1.5 * sigma_est) {
                const double dv_c = v_sub[i] - fit.moment1_kms;
                m2_num += t_sub[i] * dv_c * dv_c * dv;
            }
        }
        fit.moment2_sigma_kms = std::sqrt(std::max(0.0, m2_num / m_sig));
    } else {
        size_t pk = 0;
        for (size_t i = 1; i < n; ++i) {
            if (t_sub[i] > t_sub[pk]) pk = i;
        }
        if (t_sub[pk] > 0.0) {
            const double floor_k = 0.1 * t_sub[pk];
            size_t l = pk, r = pk;
            while (l > 0 && t_sub[l - 1] > floor_k) --l;
            while (r + 1 < n && t_sub[r + 1] > floor_k) ++r;
            m_sig = 0.0;
            m1_num = 0.0;
            for (size_t i = l; i <= r; ++i) {
                m_sig += t_sub[i] * dv;
                m1_num += t_sub[i] * v_sub[i] * dv;
            }
            if (m_sig > 1e-9) {
                fit.moment1_kms = m1_num / m_sig;
                double m2_num = 0.0;
                for (size_t i = l; i <= r; ++i) {
                    const double dv_c = v_sub[i] - fit.moment1_kms;
                    m2_num += t_sub[i] * dv_c * dv_c * dv;
                }
                fit.moment2_sigma_kms = std::sqrt(std::max(0.0, m2_num / m_sig));
            }
        }
    }
}

GaussianFit Astrophysics::fit_line_profile(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window,
    LineAnalysisMethod method
) {
    // 1. Always run single Gaussian fit as baseline seed & fallback
    GaussianFit base_fit = fit_gaussian(vel_kms, temp_k, hi_window);
    base_fit.method = method;

    const size_t n_all = vel_kms.size();
    std::vector<double> v_sub, t_sub;
    v_sub.reserve(n_all);
    t_sub.reserve(n_all);
    for (size_t i = 0; i < n_all; ++i) {
        if (hi_window[i]) {
            v_sub.push_back(vel_kms[i]);
            t_sub.push_back(temp_k[i]);
        }
    }
    populate_nonparametric_moments(v_sub, t_sub, base_fit);

    if (v_sub.size() < 7 || method == LineAnalysisMethod::GaussianLS) {
        return base_fit;
    }

    const size_t m = v_sub.size();

    // -------------------------------------------------------------------------
    // Method 5: Non-Parametric Spectral Moments (M0 / M1 / M2 + Gaussian smoothing)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::SpectralMoments) {
        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::SpectralMoments;
        res.fitted_curve.assign(n_all, 0.0);

        // 7-tap Gaussian-weighted smoothing kernel (sigma ~ 1.4 bins) applied continuously across spectrum
        static constexpr double kKernel[7] = {0.045, 0.110, 0.195, 0.300, 0.195, 0.110, 0.045};
        for (size_t i = 0; i < n_all; ++i) {
            double acc = 0.0, wsum = 0.0;
            for (int k = -3; k <= 3; ++k) {
                long long idx = static_cast<long long>(i) + k;
                if (idx >= 0 && idx < static_cast<long long>(n_all)) {
                    const double kw = kKernel[k + 3];
                    acc += kw * temp_k[static_cast<size_t>(idx)];
                    wsum += kw;
                }
            }
            res.fitted_curve[i] = (wsum > 0.0) ? std::max(0.0, acc / wsum) : 0.0;
        }

        // Peak & sub-bin half-maximum width on smoothed curve inside H I window
        size_t pk_i = 0;
        double pk_v = -1e9;
        for (size_t i = 0; i < n_all; ++i) {
            if (hi_window[i] && res.fitted_curve[i] > pk_v) {
                pk_v = res.fitted_curve[i];
                pk_i = i;
            }
        }
        if (pk_v > 0.05) {
            res.success = true;
            res.amplitude_k = pk_v;
            res.v_center_kms = (res.moment0_k_kms > 1e-6) ? res.moment1_kms : vel_kms[pk_i];
            res.sigma_kms = (res.moment2_sigma_kms > 0.5) ? res.moment2_sigma_kms : base_fit.sigma_kms;

            // Direct linear-interpolated half-maximum crossings around pk_i
            const double half_h = 0.5 * pk_v;
            size_t li = pk_i, ri = pk_i;
            while (li > 0 && hi_window[li - 1] && res.fitted_curve[li - 1] >= half_h) --li;
            while (ri + 1 < n_all && hi_window[ri + 1] && res.fitted_curve[ri + 1] >= half_h) ++ri;

            auto interp_v = [&](size_t i0, size_t i1) {
                const double y0 = res.fitted_curve[i0], y1 = res.fitted_curve[i1];
                if (std::abs(y1 - y0) < 1e-9) return vel_kms[i0];
                const double t = std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0);
                return vel_kms[i0] + t * (vel_kms[i1] - vel_kms[i0]);
            };
            const double v_a = (li > 0 && hi_window[li - 1]) ? interp_v(li - 1, li) : vel_kms[li];
            const double v_b = (ri + 1 < n_all && hi_window[ri + 1]) ? interp_v(ri, ri + 1) : vel_kms[ri];
            const double direct_fwhm = std::abs(v_b - v_a);
            res.fwhm_kms = (direct_fwhm > 0.5) ? direct_fwhm : (GAUSSIAN_SIGMA_TO_FWHM * res.sigma_kms);
            res.fwhm_left_kms = std::min(v_a, v_b);
            res.fwhm_right_kms = std::max(v_a, v_b);
            res.fwhm_half_temp_k = half_h;

            double sq_err = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < n_all; ++i) {
                if (hi_window[i]) {
                    const double d = temp_k[i] - res.fitted_curve[i];
                    sq_err += d * d;
                    ++cnt;
                }
            }
            res.rmse_k = std::sqrt(sq_err / static_cast<double>(std::max<size_t>(1, cnt)));
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 2: Double Gaussian (6-parameter Levenberg-Marquardt)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::DoubleGaussian && base_fit.success) {
        // Seed Component 1 from primary peak, Component 2 from highest positive residual outside 0.65*sigma1
        double p[6] = {
            base_fit.amplitude_k * 0.85,
            base_fit.v_center_kms,
            std::clamp(base_fit.sigma_kms * 0.80, 1.0, 60.0),
            base_fit.amplitude_k * 0.30,
            base_fit.v_center_kms + ((base_fit.v_center_kms >= 0.0) ? -14.0 : 14.0),
            std::clamp(base_fit.sigma_kms * 0.90, 1.0, 60.0)
        };

        double best_res = -1e9;
        size_t best_j = 0;
        for (size_t j = 0; j < m; ++j) {
            const double dv1 = v_sub[j] - base_fit.v_center_kms;
            const double g1 = base_fit.amplitude_k * std::exp(-0.5 * (dv1 * dv1) / (base_fit.sigma_kms * base_fit.sigma_kms));
            const double r = t_sub[j] - g1;
            if (std::abs(dv1) > 0.65 * base_fit.sigma_kms && r > best_res) {
                best_res = r;
                best_j = j;
            }
        }
        if (best_res > 0.08 * base_fit.amplitude_k) {
            p[3] = std::max(0.15, best_res);
            p[4] = v_sub[best_j];
        }

        auto eval_2g = [&](const double q[6], double v, double* out_g1 = nullptr, double* out_g2 = nullptr) {
            const double d1 = (v - q[1]) / q[2];
            const double d2 = (v - q[4]) / q[5];
            const double g1 = q[0] * std::exp(-0.5 * d1 * d1);
            const double g2 = q[3] * std::exp(-0.5 * d2 * d2);
            if (out_g1) *out_g1 = g1;
            if (out_g2) *out_g2 = g2;
            return g1 + g2;
        };
        auto chi2_2g = [&](const double q[6]) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const double r = t_sub[i] - eval_2g(q, v_sub[i]);
                s += r * r;
            }
            return s;
        };

        const double init_chi2 = chi2_2g(p);
        double cur_chi2 = init_chi2;
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        for (int iter = 0; iter < 90; ++iter) {
            double JTJ[9][9] = {};
            double JTr[9] = {};
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double diff1 = v - p[1];
                const double isig1 = 1.0 / p[2];
                const double u1 = diff1 * isig1;
                const double e1 = std::exp(-0.5 * u1 * u1);

                const double diff2 = v - p[4];
                const double isig2 = 1.0 / p[5];
                const double u2 = diff2 * isig2;
                const double e2 = std::exp(-0.5 * u2 * u2);

                const double r = t_sub[i] - (p[0] * e1 + p[3] * e2);
                const double J[6] = {
                    e1,
                    p[0] * e1 * u1 * isig1,
                    p[0] * e1 * u1 * u1 * isig1,
                    e2,
                    p[3] * e2 * u2 * isig2,
                    p[3] * e2 * u2 * u2 * isig2
                };
                for (int a = 0; a < 6; ++a) {
                    JTr[a] += J[a] * r;
                    for (int b = a; b < 6; ++b) JTJ[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < a; ++b) JTJ[a][b] = JTJ[b][a];
                JTJ[a][a] += lambda * (JTJ[a][a] + 1e-3);
            }
            double delta[9] = {};
            if (!solve_linear_nxn(6, JTJ, JTr, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }
            double step_scale = 1.0;
            if (std::abs(delta[1]) > 5.0) step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
            if (std::abs(delta[4]) > 5.0) step_scale = std::min(step_scale, 5.0 / std::abs(delta[4]));

            double cand[6] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], -220.0, 220.0),
                std::clamp(p[2] + step_scale * delta[2], 0.8, 90.0),
                std::clamp(p[3] + step_scale * delta[3], 0.02, max_a),
                std::clamp(p[4] + step_scale * delta[4], -220.0, 220.0),
                std::clamp(p[5] + step_scale * delta[5], 0.8, 90.0)
            };
            const double cand_chi2 = chi2_2g(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 6; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 6; ++k) p[k] = cand[k];
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Order so Component 1 is the higher-amplitude primary peak
        if (p[3] > p[0]) {
            std::swap(p[0], p[3]);
            std::swap(p[1], p[4]);
            std::swap(p[2], p[5]);
        }

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::DoubleGaussian;
        res.success = (cur_chi2 <= init_chi2);
        res.amplitude_k = p[0];
        res.v_center_kms = p[1];
        res.sigma_kms = p[2];
        res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * p[2];
        res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
        res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
        res.fwhm_half_temp_k = 0.5 * res.amplitude_k;
        res.has_secondary = (p[3] > 0.04 * p[0] && std::abs(p[4] - p[1]) > 1.5);
        res.amp2_k = p[3];
        res.v2_center_kms = p[4];
        res.sigma2_kms = p[5];
        res.fwhm2_kms = GAUSSIAN_SIGMA_TO_FWHM * p[5];
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));

        res.fitted_curve.resize(n_all);
        res.component1_curve.resize(n_all);
        res.component2_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            double c1 = 0.0, c2 = 0.0;
            res.fitted_curve[i] = eval_2g(p, vel_kms[i], &c1, &c2);
            res.component1_curve[i] = c1;
            res.component2_curve[i] = c2;
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 6: Triple Gaussian (9-parameter Levenberg-Marquardt for multi-arm / CNM+WNM)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::TripleGaussian && base_fit.success) {
        // Seed first two components via DoubleGaussian fit, then seed Component 3 from 2G residuals
        GaussianFit dg_seed = fit_line_profile(vel_kms, temp_k, hi_window, LineAnalysisMethod::DoubleGaussian);
        double p[9] = {
            dg_seed.amplitude_k * 0.90,
            dg_seed.v_center_kms,
            std::clamp(dg_seed.sigma_kms, 1.0, 60.0),
            std::max(0.10, dg_seed.amp2_k * 0.90),
            dg_seed.v2_center_kms,
            std::clamp(dg_seed.sigma2_kms, 1.0, 60.0),
            base_fit.amplitude_k * 0.22,
            base_fit.moment1_kms,
            std::clamp(base_fit.sigma_kms * 1.45, 2.0, 75.0)
        };

        // Search 2-Gaussian residuals for a distinct 3rd kinematic arm
        double best_res3 = -1e9;
        size_t best_j3 = 0;
        for (size_t j = 0; j < m; ++j) {
            const double v = v_sub[j];
            const double d1 = (v - dg_seed.v_center_kms) / std::max(1.0, dg_seed.sigma_kms);
            const double d2 = (v - dg_seed.v2_center_kms) / std::max(1.0, dg_seed.sigma2_kms);
            const double g12 = dg_seed.amplitude_k * std::exp(-0.5 * d1 * d1)
                             + dg_seed.amp2_k * std::exp(-0.5 * d2 * d2);
            const double r = t_sub[j] - g12;
            if (std::abs(d1) > 0.65 && std::abs(d2) > 0.65 && r > best_res3) {
                best_res3 = r;
                best_j3 = j;
            }
        }
        if (best_res3 > 0.05 * base_fit.amplitude_k) {
            p[6] = std::max(0.12, best_res3);
            p[7] = v_sub[best_j3];
            p[8] = std::clamp(base_fit.sigma_kms * 0.75, 1.5, 50.0);
        }

        auto eval_3g = [&](const double q[9], double v, double* c1 = nullptr, double* c2 = nullptr, double* c3 = nullptr) {
            const double d1 = (v - q[1]) / q[2];
            const double d2 = (v - q[4]) / q[5];
            const double d3 = (v - q[7]) / q[8];
            const double g1 = q[0] * std::exp(-0.5 * d1 * d1);
            const double g2 = q[3] * std::exp(-0.5 * d2 * d2);
            const double g3 = q[6] * std::exp(-0.5 * d3 * d3);
            if (c1) *c1 = g1;
            if (c2) *c2 = g2;
            if (c3) *c3 = g3;
            return g1 + g2 + g3;
        };
        auto chi2_3g = [&](const double q[9]) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const double r = t_sub[i] - eval_3g(q, v_sub[i]);
                s += r * r;
            }
            return s;
        };

        const double init_chi2 = chi2_3g(p);
        double cur_chi2 = init_chi2;
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        for (int iter = 0; iter < 100; ++iter) {
            double JTJ[9][9] = {};
            double JTr[9] = {};
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double isig1 = 1.0 / p[2], u1 = (v - p[1]) * isig1, e1 = std::exp(-0.5 * u1 * u1);
                const double isig2 = 1.0 / p[5], u2 = (v - p[4]) * isig2, e2 = std::exp(-0.5 * u2 * u2);
                const double isig3 = 1.0 / p[8], u3 = (v - p[7]) * isig3, e3 = std::exp(-0.5 * u3 * u3);

                const double r = t_sub[i] - (p[0] * e1 + p[3] * e2 + p[6] * e3);
                const double J[9] = {
                    e1, p[0] * e1 * u1 * isig1, p[0] * e1 * u1 * u1 * isig1,
                    e2, p[3] * e2 * u2 * isig2, p[3] * e2 * u2 * u2 * isig2,
                    e3, p[6] * e3 * u3 * isig3, p[6] * e3 * u3 * u3 * isig3
                };
                for (int a = 0; a < 9; ++a) {
                    JTr[a] += J[a] * r;
                    for (int b = a; b < 9; ++b) JTJ[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 9; ++a) {
                for (int b = 0; b < a; ++b) JTJ[a][b] = JTJ[b][a];
                JTJ[a][a] += lambda * (JTJ[a][a] + 1e-3);
            }
            double delta[9] = {};
            if (!solve_linear_nxn(9, JTJ, JTr, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }
            double step_scale = 1.0;
            for (int c_idx : {1, 4, 7}) {
                if (std::abs(delta[c_idx]) > 5.0) {
                    step_scale = std::min(step_scale, 5.0 / std::abs(delta[c_idx]));
                }
            }

            double cand[9] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], -220.0, 220.0),
                std::clamp(p[2] + step_scale * delta[2], 0.8, 90.0),
                std::clamp(p[3] + step_scale * delta[3], 0.02, max_a),
                std::clamp(p[4] + step_scale * delta[4], -220.0, 220.0),
                std::clamp(p[5] + step_scale * delta[5], 0.8, 90.0),
                std::clamp(p[6] + step_scale * delta[6], 0.02, max_a),
                std::clamp(p[7] + step_scale * delta[7], -220.0, 220.0),
                std::clamp(p[8] + step_scale * delta[8], 0.8, 90.0)
            };
            const double cand_chi2 = chi2_3g(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 9; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 9; ++k) p[k] = cand[k];
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Sort components in descending order of amplitude: A1 >= A2 >= A3
        auto swap_comp = [&](int i, int j) {
            std::swap(p[3 * i + 0], p[3 * j + 0]);
            std::swap(p[3 * i + 1], p[3 * j + 1]);
            std::swap(p[3 * i + 2], p[3 * j + 2]);
        };
        if (p[3] > p[0]) swap_comp(0, 1);
        if (p[6] > p[0]) swap_comp(0, 2);
        if (p[6] > p[3]) swap_comp(1, 2);

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::TripleGaussian;
        res.success = (cur_chi2 <= init_chi2);
        res.amplitude_k = p[0];
        res.v_center_kms = p[1];
        res.sigma_kms = p[2];
        res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * p[2];
        res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
        res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
        res.fwhm_half_temp_k = 0.5 * res.amplitude_k;

        res.has_secondary = (p[3] > 0.04 * p[0] && (std::abs(p[4] - p[1]) > 1.5 || std::abs(p[5] - p[2]) > 1.5));
        res.amp2_k = p[3];
        res.v2_center_kms = p[4];
        res.sigma2_kms = p[5];
        res.fwhm2_kms = GAUSSIAN_SIGMA_TO_FWHM * p[5];

        res.has_tertiary = (p[6] > 0.03 * p[0] &&
            ((std::abs(p[7] - p[1]) > 1.5 && std::abs(p[7] - p[4]) > 1.5) || std::abs(p[8] - p[2]) > 2.0));
        res.amp3_k = p[6];
        res.v3_center_kms = p[7];
        res.sigma3_kms = p[8];
        res.fwhm3_kms = GAUSSIAN_SIGMA_TO_FWHM * p[8];
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));

        res.fitted_curve.resize(n_all);
        res.component1_curve.resize(n_all);
        res.component2_curve.resize(n_all);
        res.component3_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            double c1 = 0.0, c2 = 0.0, c3 = 0.0;
            res.fitted_curve[i] = eval_3g(p, vel_kms[i], &c1, &c2, &c3);
            res.component1_curve[i] = c1;
            res.component2_curve[i] = c2;
            res.component3_curve[i] = c3;
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 3: Pseudo-Voigt Profile (4-parameter Levenberg-Marquardt)
    // T(v) = A * [ (1 - eta) * exp(-0.5 * ((v-v0)/sigma)^2) + eta / (1 + ((v-v0)/gamma)^2) ]
    // where gamma = sigma * sqrt(2 * ln(2)) so Gaussian & Lorentzian share equal FWHM.
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::VoigtProfile && base_fit.success) {
        double p[4] = { base_fit.amplitude_k, base_fit.v_center_kms, base_fit.sigma_kms, 0.20 };
        constexpr double kSqrt2Ln2 = 1.1774100225154747; // FWHM / 2 = gamma when matched to sigma

        auto eval_voigt = [&](const double q[4], double v) {
            const double u = (v - q[1]) / q[2];
            const double g = std::exp(-0.5 * u * u);
            const double gamma = q[2] * kSqrt2Ln2;
            const double w = (v - q[1]) / gamma;
            const double lor = 1.0 / (1.0 + w * w);
            return q[0] * ((1.0 - q[3]) * g + q[3] * lor);
        };
        auto chi2_voigt = [&](const double q[4]) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const double r = t_sub[i] - eval_voigt(q, v_sub[i]);
                s += r * r;
            }
            return s;
        };

        const double init_chi2 = chi2_voigt(p);
        double cur_chi2 = init_chi2;
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        for (int iter = 0; iter < 80; ++iter) {
            double JTJ[9][9] = {};
            double JTr[9] = {};
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double r = t_sub[i] - eval_voigt(p, v);
                double J[4] = {};
                const double eps[4] = { 1e-4, 1e-4, 1e-4, 1e-4 };
                for (int k = 0; k < 4; ++k) {
                    double qp[4] = { p[0], p[1], p[2], p[3] };
                    qp[k] += eps[k];
                    J[k] = (eval_voigt(qp, v) - (t_sub[i] - r)) / eps[k];
                }
                for (int a = 0; a < 4; ++a) {
                    JTr[a] += J[a] * r;
                    for (int b = a; b < 4; ++b) JTJ[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < a; ++b) JTJ[a][b] = JTJ[b][a];
                JTJ[a][a] += lambda * (JTJ[a][a] + 1e-3);
            }
            double delta[9] = {};
            if (!solve_linear_nxn(4, JTJ, JTr, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }
            double cand[4] = {
                std::clamp(p[0] + delta[0], 0.1, max_a),
                std::clamp(p[1] + std::clamp(delta[1], -5.0, 5.0), -250.0, 250.0),
                std::clamp(p[2] + std::clamp(delta[2], -0.5 * p[2], 0.5 * p[2]), 0.5, 100.0),
                std::clamp(p[3] + std::clamp(delta[3], -0.25, 0.25), 0.0, 1.0)
            };
            const double cand_chi2 = chi2_voigt(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 4; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 4; ++k) p[k] = cand[k];
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::VoigtProfile;
        res.success = (cur_chi2 <= init_chi2);
        res.amplitude_k = p[0];
        res.v_center_kms = p[1];
        res.sigma_kms = p[2];
        res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * p[2];
        res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
        res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
        res.fwhm_half_temp_k = 0.5 * res.amplitude_k;
        res.voigt_eta = p[3];
        res.lorentz_gamma_kms = p[2] * kSqrt2Ln2;
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.fitted_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            res.fitted_curve[i] = eval_voigt(p, vel_kms[i]);
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 4: Gauss-Hermite Asymmetric Line Profile (5-parameter L-M)
    // T(v) = A * exp(-0.5 * y^2) * [ 1 + h3 * H3(y) + h4 * H4(y) ], y = (v - v0) / sigma
    // H3(y) = (2*y^3 - 3*y) / sqrt(3),  H4(y) = (4*y^4 - 12*y^2 + 3) / sqrt(24)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::GaussHermite && base_fit.success) {
        double p[5] = { base_fit.amplitude_k, base_fit.v_center_kms, base_fit.sigma_kms, 0.0, 0.0 };
        constexpr double kInvSqrt3  = 0.5773502691896258;
        constexpr double kInvSqrt24 = 0.2041241452319315;

        auto eval_gh = [&](const double q[5], double v) {
            const double y = (v - q[1]) / q[2];
            const double y2 = y * y;
            const double g = std::exp(-0.5 * y2);
            const double h3_poly = (2.0 * y2 * y - 3.0 * y) * kInvSqrt3;
            const double h4_poly = (4.0 * y2 * y2 - 12.0 * y2 + 3.0) * kInvSqrt24;
            return q[0] * g * (1.0 + q[3] * h3_poly + q[4] * h4_poly);
        };
        auto chi2_gh = [&](const double q[5]) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const double r = t_sub[i] - eval_gh(q, v_sub[i]);
                s += r * r;
            }
            return s;
        };

        const double init_chi2 = chi2_gh(p);
        double cur_chi2 = init_chi2;
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        for (int iter = 0; iter < 80; ++iter) {
            double JTJ[9][9] = {};
            double JTr[9] = {};
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double f0 = eval_gh(p, v);
                const double r = t_sub[i] - f0;
                double J[5] = {};
                const double eps[5] = { 1e-4, 1e-4, 1e-4, 1e-4, 1e-4 };
                for (int k = 0; k < 5; ++k) {
                    double qp[5] = { p[0], p[1], p[2], p[3], p[4] };
                    qp[k] += eps[k];
                    J[k] = (eval_gh(qp, v) - f0) / eps[k];
                }
                for (int a = 0; a < 5; ++a) {
                    JTr[a] += J[a] * r;
                    for (int b = a; b < 5; ++b) JTJ[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 5; ++a) {
                for (int b = 0; b < a; ++b) JTJ[a][b] = JTJ[b][a];
                JTJ[a][a] += lambda * (JTJ[a][a] + 1e-3);
            }
            double delta[9] = {};
            if (!solve_linear_nxn(5, JTJ, JTr, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }
            double cand[5] = {
                std::clamp(p[0] + delta[0], 0.1, max_a),
                std::clamp(p[1] + std::clamp(delta[1], -5.0, 5.0), -250.0, 250.0),
                std::clamp(p[2] + std::clamp(delta[2], -0.4 * p[2], 0.4 * p[2]), 0.5, 100.0),
                std::clamp(p[3] + std::clamp(delta[3], -0.15, 0.15), -0.45, 0.45),
                std::clamp(p[4] + std::clamp(delta[4], -0.15, 0.15), -0.45, 0.45)
            };
            const double cand_chi2 = chi2_gh(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 5; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 5; ++k) p[k] = cand[k];
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Compute exact numerical peak and FWHM from the fitted Gauss-Hermite profile
        constexpr int kGridN = 401;
        const double v_lo = p[1] - 4.5 * p[2];
        const double v_hi = p[1] + 4.5 * p[2];
        const double v_step = (v_hi - v_lo) / (kGridN - 1);
        double gh_vals[kGridN];
        int pk_idx = kGridN / 2;
        double pk_val = -1e9;
        for (int g = 0; g < kGridN; ++g) {
            gh_vals[g] = eval_gh(p, v_lo + g * v_step);
            if (gh_vals[g] > pk_val) {
                pk_val = gh_vals[g];
                pk_idx = g;
            }
        }

        double fwhm_exact = GAUSSIAN_SIGMA_TO_FWHM * p[2];
        double v_left = p[1] - 0.5 * fwhm_exact;
        double v_right = p[1] + 0.5 * fwhm_exact;
        const double half_h = 0.5 * pk_val;
        if (pk_val > 0.05) {
            int l_g = pk_idx;
            while (l_g > 0 && gh_vals[l_g - 1] >= half_h) --l_g;
            int r_g = pk_idx;
            while (r_g + 1 < kGridN && gh_vals[r_g + 1] >= half_h) ++r_g;

            auto grid_v = [&](int idx) { return v_lo + idx * v_step; };
            if (l_g > 0) {
                double y0 = gh_vals[l_g - 1], y1 = gh_vals[l_g];
                double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_left = grid_v(l_g - 1) + t * v_step;
            } else {
                v_left = grid_v(l_g);
            }
            if (r_g + 1 < kGridN) {
                double y0 = gh_vals[r_g], y1 = gh_vals[r_g + 1];
                double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_right = grid_v(r_g) + t * v_step;
            } else {
                v_right = grid_v(r_g);
            }
            if (v_right > v_left) fwhm_exact = v_right - v_left;
        }

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::GaussHermite;
        res.success = (cur_chi2 <= init_chi2);
        res.amplitude_k = (pk_val > 0.05) ? pk_val : p[0];
        res.v_center_kms = p[1];
        res.sigma_kms = p[2];
        res.fwhm_kms = fwhm_exact;
        res.fwhm_left_kms = v_left;
        res.fwhm_right_kms = v_right;
        res.fwhm_half_temp_k = (pk_val > 0.05) ? half_h : (0.5 * p[0]);
        res.gh_h3 = p[3];
        res.gh_h4 = p[4];
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.fitted_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            res.fitted_curve[i] = eval_gh(p, vel_kms[i]);
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 7: Generalized Busy Function (Westmeier et al. 2014, MNRAS 438, 1176)
    // B(v) = (a / 4) * [erf(b1 * (w + v - v0)) + 1] * [erf(b2 * (w - v + v0)) + 1] * (1 + c * (v - v0)^2)
    // 6 parameters: p = [a, v0, w, b1, b2, c]
    // Models flat-topped, double-horned, and asymmetric disk/galactic 21-cm profiles.
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::BusyFunction && base_fit.success) {
        const double v0_seed = (base_fit.moment0_k_kms > 1e-3) ? base_fit.moment1_kms : base_fit.v_center_kms;
        const double sig_seed = (base_fit.moment2_sigma_kms > 1.0) ? base_fit.moment2_sigma_kms : base_fit.sigma_kms;
        const double w_seed = std::clamp(0.5 * GAUSSIAN_SIGMA_TO_FWHM * sig_seed, 2.0, 120.0);
        const double b_seed = std::clamp(2.0 / std::max(1.0, base_fit.sigma_kms), 0.05, 1.5);

        // Estimate whether a central trough (double-horn) is present
        double center_temp = 0.0;
        int center_cnt = 0;
        double max_sub_t = 0.0;
        for (size_t i = 0; i < m; ++i) {
            max_sub_t = std::max(max_sub_t, t_sub[i]);
            if (std::abs(v_sub[i] - v0_seed) < 0.25 * w_seed) {
                center_temp += t_sub[i];
                ++center_cnt;
            }
        }
        if (center_cnt > 0) center_temp /= center_cnt;
        const bool has_trough = (center_cnt > 0 && center_temp < 0.85 * max_sub_t);
        const double a_seed = has_trough ? std::max(0.2, center_temp) : base_fit.amplitude_k;
        const double c_seed = (has_trough && w_seed > 2.0)
            ? std::clamp((max_sub_t / std::max(0.2, a_seed) - 1.0) / (0.6 * w_seed * w_seed), 0.0, 0.02)
            : 0.0002;

        double p[6] = { a_seed, v0_seed, w_seed, b_seed, b_seed, c_seed };

        auto eval_busy = [&](const double q[6], double v) {
            const double dv = v - q[1];
            const double e1 = std::erf(q[3] * (q[2] + dv)) + 1.0;
            const double e2 = std::erf(q[4] * (q[2] - dv)) + 1.0;
            const double poly = 1.0 + q[5] * dv * dv;
            return 0.25 * q[0] * e1 * e2 * poly;
        };
        auto chi2_busy = [&](const double q[6]) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const double r = t_sub[i] - eval_busy(q, v_sub[i]);
                s += r * r;
            }
            return s;
        };

        const double init_chi2 = chi2_busy(p);
        double cur_chi2 = init_chi2;
        double lambda = 1.0;
        const double max_a = std::max(15.0, max_sub_t * 3.0 + 5.0);

        for (int iter = 0; iter < 90; ++iter) {
            double JTJ[9][9] = {};
            double JTr[9] = {};
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double f0 = eval_busy(p, v);
                const double r = t_sub[i] - f0;
                double J[6] = {};
                const double eps[6] = { 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6 };
                for (int k = 0; k < 6; ++k) {
                    double qp[6] = { p[0], p[1], p[2], p[3], p[4], p[5] };
                    qp[k] += eps[k];
                    J[k] = (eval_busy(qp, v) - f0) / eps[k];
                }
                for (int a = 0; a < 6; ++a) {
                    JTr[a] += J[a] * r;
                    for (int b = a; b < 6; ++b) JTJ[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < a; ++b) JTJ[a][b] = JTJ[b][a];
                JTJ[a][a] += lambda * (JTJ[a][a] + 1e-3);
            }
            double delta[9] = {};
            if (!solve_linear_nxn(6, JTJ, JTr, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }
            double cand[6] = {
                std::clamp(p[0] + std::clamp(delta[0], -0.5 * p[0], 0.5 * p[0]), 0.05, max_a),
                std::clamp(p[1] + std::clamp(delta[1], -5.0, 5.0), -220.0, 220.0),
                std::clamp(p[2] + std::clamp(delta[2], -0.4 * p[2], 0.4 * p[2]), 1.0, 160.0),
                std::clamp(p[3] + std::clamp(delta[3], -0.35 * p[3], 0.35 * p[3]), 0.02, 5.0),
                std::clamp(p[4] + std::clamp(delta[4], -0.35 * p[4], 0.35 * p[4]), 0.02, 5.0),
                std::clamp(p[5] + std::clamp(delta[5], -0.005, 0.005), 0.0, 0.05)
            };
            const double cand_chi2 = chi2_busy(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 6; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 6; ++k) p[k] = cand[k];
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-5) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Dense 401-point evaluation to extract true peak amplitude and outer 50% width
        constexpr int kGridN = 401;
        const double span_v = std::max(25.0, 2.2 * p[2] + 4.0 / std::min(p[3], p[4]));
        const double v_lo = p[1] - span_v;
        const double v_hi = p[1] + span_v;
        const double v_step = (v_hi - v_lo) / (kGridN - 1);
        double bf_vals[kGridN];
        double pk_val = -1e9;
        for (int g = 0; g < kGridN; ++g) {
            bf_vals[g] = eval_busy(p, v_lo + g * v_step);
            if (bf_vals[g] > pk_val) pk_val = bf_vals[g];
        }

        double fwhm_exact = 2.0 * p[2];
        double v_left = p[1] - p[2];
        double v_right = p[1] + p[2];
        const double half_h = 0.5 * pk_val;
        if (pk_val > 0.05) {
            // Search inward from outer flanks so double-horned profiles measure full outer width
            int l_g = 0;
            while (l_g + 1 < kGridN && bf_vals[l_g] < half_h) ++l_g;
            int r_g = kGridN - 1;
            while (r_g > 0 && bf_vals[r_g] < half_h) --r_g;

            auto grid_v = [&](int idx) { return v_lo + idx * v_step; };
            if (l_g > 0 && l_g < kGridN) {
                double y0 = bf_vals[l_g - 1], y1 = bf_vals[l_g];
                double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_left = grid_v(l_g - 1) + t * v_step;
            }
            if (r_g >= 0 && r_g + 1 < kGridN) {
                double y0 = bf_vals[r_g], y1 = bf_vals[r_g + 1];
                double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_right = grid_v(r_g) + t * v_step;
            }
            if (v_right > v_left) fwhm_exact = v_right - v_left;
        }

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::BusyFunction;
        res.success = (cur_chi2 <= init_chi2);
        res.amplitude_k = (pk_val > 0.05) ? pk_val : p[0];
        res.v_center_kms = p[1];
        res.fwhm_kms = fwhm_exact;
        res.sigma_kms = fwhm_exact / GAUSSIAN_SIGMA_TO_FWHM;
        res.fwhm_left_kms = v_left;
        res.fwhm_right_kms = v_right;
        res.fwhm_half_temp_k = (pk_val > 0.05) ? half_h : (0.5 * p[0]);
        res.busy_half_width_kms = p[2];
        res.busy_flank_b1 = p[3];
        res.busy_flank_b2 = p[4];
        res.busy_trough_c = p[5];
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.fitted_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            res.fitted_curve[i] = eval_busy(p, vel_kms[i]);
        }
        return res;
    }

    return base_fit;
}

AstrophysicalMetrics Astrophysics::calculate_metrics(
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& vel_kms,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const AstroConfig& config
) {
    AstrophysicalMetrics m;
    int nfft = static_cast<int>(freqs_mhz.size());
    if (nfft == 0) return m;

    double hi_half_mhz = config.hi_protection_half_width_khz / 1e3;
    std::vector<uint8_t> hi_window(nfft, 0);
    for (int i = 0; i < nfft; ++i) {
        if (freqs_mhz[i] >= (HYDROGEN_REST_FREQ_MHZ - hi_half_mhz) &&
            freqs_mhz[i] <= (HYDROGEN_REST_FREQ_MHZ + hi_half_mhz)) {
            hi_window[i] = 1;
        }
    }

    // Line-free noise floor estimation: |v| in [line_free_velocity_kms, 200] within valid passband
    const int edge_bins = static_cast<int>(
        static_cast<double>(nfft) * std::clamp(config.passband_edge_fraction, 0.0, 0.45));
    std::vector<double> free_k, free_db;
    free_k.reserve(nfft);
    free_db.reserve(nfft);
    for (int i = edge_bins; i < nfft - edge_bins; ++i) {
        double v = vel_kms[i];
        if (v >= -200.0 && v <= 200.0 && std::abs(v) >= config.line_free_velocity_kms && !hi_window[i]) {
            free_k.push_back(calibrated_k[i]);
            free_db.push_back(calibrated_db[i]);
        }
    }

    if (free_k.size() < 10) {
        // Fallback: use passband excluding HI
        for (int i = edge_bins; i < nfft - edge_bins; ++i) {
            if (!hi_window[i]) {
                free_k.push_back(calibrated_k[i]);
                free_db.push_back(calibrated_db[i]);
            }
        }
    }

    auto calc_std = [](const std::vector<double>& v) {
        if (v.size() < 2) return 0.05;
        double sum = 0.0;
        for (double val : v) sum += val;
        double mean = sum / v.size();
        double sum_sq = 0.0;
        for (double val : v) sum_sq += (val - mean) * (val - mean);
        return std::sqrt(sum_sq / (v.size() - 1));
    };

    m.noise_std_k = calc_std(free_k);
    m.noise_std_db = calc_std(free_db);

    // Peak detection in HI window
    int peak_idx = nfft / 2;
    double max_k = -1e9;
    bool found_peak = false;
    for (int i = 0; i < nfft; ++i) {
        if (hi_window[i] && calibrated_k[i] > max_k) {
            max_k = calibrated_k[i];
            peak_idx = i;
            found_peak = true;
        }
    }

    if (found_peak) {
        m.peak_freq_mhz = freqs_mhz[peak_idx];
        m.peak_vel_kms = vel_kms[peak_idx];
        m.peak_temp_k = calibrated_k[peak_idx];
        m.peak_power_db = calibrated_db[peak_idx];
    } else {
        m.peak_freq_mhz = HYDROGEN_REST_FREQ_MHZ;
        m.peak_vel_kms = 0.0;
        m.peak_temp_k = 0.0;
        m.peak_power_db = 0.0;
    }

    m.snr = m.peak_temp_k / std::max(1e-6, m.noise_std_k);

    // Unbiased integrated line flux: W = sum(T_A) * dv [K * km/s]
    double dv = (nfft > 1) ? std::abs(vel_kms[1] - vel_kms[0]) : 1.0;
    double integral_w = 0.0;
    for (int i = 0; i < nfft; ++i) {
        if (hi_window[i]) {
            integral_w += calibrated_k[i] * dv;
        }
    }
    m.integrated_flux_k_kms = integral_w;
    m.column_density_cm2 = HI_COLUMN_DENSITY_FACTOR * std::max(0.0, integral_w);

    // Line Profile Analysis (Single/Double/Triple Gaussian, Pseudo-Voigt, Gauss-Hermite, Moments, Busy Function)
    m.gauss_fit = fit_line_profile(vel_kms, calibrated_k, hi_window, config.line_method);
    m.fwhm_kms = m.gauss_fit.fwhm_kms;
    m.sigma_kms = m.gauss_fit.sigma_kms;

    // Derived 21-cm physical & kinematic metrics
    // 1. Upper-bound thermal kinetic temperature: T_k,max = (m_H * c^2 / (8 * k_B * ln 2)) * FWHM^2
    m.doppler_temp_max_k = HI_THERMAL_TEMP_FACTOR_K * m.fwhm_kms * m.fwhm_kms;

    // 2. Equivalent width: W_eq = int(T_B dv) / T_peak [km/s]
    m.equivalent_width_kms = (m.peak_temp_k > 0.05) ? (std::max(0.0, integral_w) / m.peak_temp_k) : 0.0;

    // 3. Optical depth & optically-thick column density under harmonic-mean spin temperature T_s
    const double t_spin = std::max(10.0, config.spin_temp_k);
    if (m.peak_temp_k > 0.0) {
        const double ratio_pk = std::min(0.999, m.peak_temp_k / t_spin);
        m.peak_optical_depth = -std::log(1.0 - ratio_pk);
    }
    double tau_integral_kms = 0.0;
    for (int i = 0; i < nfft; ++i) {
        if (hi_window[i]) {
            const double tb = calibrated_k[i];
            if (tb > 0.0) {
                const double ratio = std::min(0.999, tb / t_spin);
                tau_integral_kms += (-t_spin * std::log(1.0 - ratio)) * dv;
            } else {
                // Preserve unbiased noise cancellation on negative fluctuations
                tau_integral_kms += tb * dv;
            }
        }
    }
    m.column_density_thick_cm2 = std::max(
        m.column_density_cm2,
        HI_COLUMN_DENSITY_FACTOR * std::max(0.0, tau_integral_kms));

    // 4. Full velocity widths W50 (at 50% peak) and W20 (at 20% peak)
    auto measure_profile_width = [&](double frac, double fallback_w) {
        if (!m.gauss_fit.success || m.gauss_fit.fitted_curve.size() != static_cast<size_t>(nfft)) {
            return fallback_w;
        }
        double model_pk = -1e9;
        for (int i = 0; i < nfft; ++i) {
            if (hi_window[i] && m.gauss_fit.fitted_curve[i] > model_pk) {
                model_pk = m.gauss_fit.fitted_curve[i];
            }
        }
        if (model_pk <= 0.05) return fallback_w;
        const double thresh = frac * model_pk;

        int first_hi = -1, last_hi = -1;
        for (int i = 0; i < nfft; ++i) {
            if (hi_window[i]) {
                if (first_hi < 0) first_hi = i;
                last_hi = i;
            }
        }
        if (first_hi < 0 || last_hi <= first_hi) return fallback_w;

        // Search inward from HI window edges to capture outermost flanks of multi-peaked / double-horn profiles
        int li = first_hi;
        while (li < last_hi && m.gauss_fit.fitted_curve[li] < thresh) ++li;
        int ri = last_hi;
        while (ri > first_hi && m.gauss_fit.fitted_curve[ri] < thresh) --ri;
        if (ri <= li) return fallback_w;

        auto interp_cross = [&](int i0, int i1) {
            const double y0 = m.gauss_fit.fitted_curve[i0];
            const double y1 = m.gauss_fit.fitted_curve[i1];
            if (std::abs(y1 - y0) < 1e-12) return vel_kms[i0];
            const double t = std::clamp((thresh - y0) / (y1 - y0), 0.0, 1.0);
            return vel_kms[i0] + t * (vel_kms[i1] - vel_kms[i0]);
        };
        const double v_a = (li > first_hi) ? interp_cross(li - 1, li) : vel_kms[li];
        const double v_b = (ri < last_hi) ? interp_cross(ri, ri + 1) : vel_kms[ri];
        const double w = std::abs(v_b - v_a);
        return (w > 0.5) ? w : fallback_w;
    };

    // For a pure Gaussian, W20 / W50 = sqrt(ln(5) / ln(2)) = 1.5238824
    m.w50_kms = measure_profile_width(0.50, m.fwhm_kms);
    m.w20_kms = measure_profile_width(0.20, m.fwhm_kms * 1.5238824);

    // SDR LO Offset and DC notch analysis
    m.lo_offset_khz = (config.center_freq_hz - HYDROGEN_REST_FREQ_HZ) / 1e3;
    m.dc_bin = nfft / 2;

    int closest_hi_bin = 0;
    double min_f_diff = 1e12;
    for (int i = 0; i < nfft; ++i) {
        double diff = std::abs(freqs_mhz[i] - HYDROGEN_REST_FREQ_MHZ);
        if (diff < min_f_diff) {
            min_f_diff = diff;
            closest_hi_bin = i;
        }
    }
    m.hi_bin = closest_hi_bin;
    m.bin_separation = std::abs(m.dc_bin - m.hi_bin);

    return m;
}

} // namespace astro
