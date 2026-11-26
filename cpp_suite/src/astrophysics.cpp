#include "astrophysics.h"
#include "simd_dsp.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace astro {

namespace {

constexpr double kInvSqrt2 = 0.7071067811865475244;
constexpr double kTwoOverSqrtPi = 1.1283791670955125739;

// Exact power-spectrum bin-correlation factor F_corr = 1 + 2 * sum_{L>=1} rho_power(L)
double window_corr_factor(WindowType w) {
    switch (w) {
        case WindowType::Hann:           return 1.944444;
        case WindowType::BlackmanHarris: return 2.7632;
        case WindowType::Blackman:
        default:                         return 2.3480;
    }
}

double compute_median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    if (v.size() % 2 == 1) {
        return v[mid];
    }
    const double hi = v[mid];
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lo + hi);
}

double estimate_mad_std(const std::vector<double>& vals) {
    if (vals.size() < 2) return 0.0;
    std::vector<double> tmp;
    tmp.reserve(vals.size());
    for (double x : vals) {
        if (std::isfinite(x)) tmp.push_back(x);
    }
    if (tmp.size() < 2) return 0.0;
    const double med = compute_median(tmp);
    for (double& x : tmp) x = std::abs(x - med);
    return 1.4826 * compute_median(tmp);
}

// Unbiased lag-6 difference MAD thermal noise estimator:
// sigma = 1.4826 * median(|y[i] - y[i - lag]|) / sqrt(2)
// Unbiased to <0.01% for Hann, Blackman, and Blackman-Harris windows (rho(6) = 0)
// and immune to smooth standing-wave ripples and isolated RFI outliers.
double estimate_lag6_thermal_rms(
    const std::vector<double>& y,
    const std::vector<uint8_t>& clean_mask
) {
    const size_t n = std::min(y.size(), clean_mask.size());
    if (n < 2) return 0.0;

    static constexpr size_t kLags[] = {6, 4, 3, 2, 1};
    std::vector<double> diffs;
    diffs.reserve(n);
    std::vector<double> best_diffs;

    for (size_t lag : kLags) {
        if (n <= lag) continue;
        diffs.clear();
        for (size_t i = lag; i < n; ++i) {
            if (clean_mask[i] && clean_mask[i - lag]) {
                const double a = y[i];
                const double b = y[i - lag];
                if (std::isfinite(a) && std::isfinite(b)) {
                    diffs.push_back(std::abs(a - b));
                }
            }
        }
        if (diffs.size() >= 16) {
            return 1.4826 * compute_median(diffs) * kInvSqrt2;
        }
        if (diffs.size() > best_diffs.size()) {
            best_diffs = diffs;
        }
    }

    if (best_diffs.size() >= 4) {
        return 1.4826 * compute_median(best_diffs) * kInvSqrt2;
    }

    std::vector<double> clean_vals;
    clean_vals.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (clean_mask[i] && std::isfinite(y[i])) {
            clean_vals.push_back(y[i]);
        }
    }
    return estimate_mad_std(clean_vals);
}

double estimate_lag6_thermal_rms(const std::vector<double>& y) {
    std::vector<uint8_t> all_clean(y.size(), 1);
    return estimate_lag6_thermal_rms(y, all_clean);
}

// Robust sigma-clipped RMS (matches true Gaussian & sinusoidal ripple RMS while rejecting spikes)
double calc_robust_rms(const std::vector<double>& vals) {
    std::vector<double> clean;
    clean.reserve(vals.size());
    for (double x : vals) {
        if (std::isfinite(x)) clean.push_back(x);
    }
    if (clean.size() < 2) return 0.0;

    std::vector<double> tmp = clean;
    const double med = compute_median(tmp);
    for (double& x : tmp) x = std::abs(x - med);
    const double mad_sig = 1.4826 * compute_median(tmp);

    if (mad_sig <= 1e-12) {
        double sum = 0.0;
        for (double x : clean) sum += x;
        const double mean = sum / static_cast<double>(clean.size());
        double sq = 0.0;
        for (double x : clean) {
            const double d = x - mean;
            sq += d * d;
        }
        return std::sqrt(sq / static_cast<double>(clean.size() - 1));
    }

    const double clip = 4.5 * mad_sig;
    double sum = 0.0;
    size_t cnt = 0;
    for (double x : clean) {
        if (std::abs(x - med) <= clip) {
            sum += x;
            ++cnt;
        }
    }
    if (cnt < 2) return mad_sig;
    const double mean = sum / static_cast<double>(cnt);
    double sq = 0.0;
    for (double x : clean) {
        if (std::abs(x - med) <= clip) {
            const double d = x - mean;
            sq += d * d;
        }
    }
    return std::sqrt(sq / static_cast<double>(cnt - 1));
}

// Empirical bin-correlation factor F_corr = 1 + 2 * sum_{L=1..5} rho(L) for standalone profile fits
double estimate_empirical_f_corr(const std::vector<double>& r) {
    const size_t n = r.size();
    if (n < 32) return 1.0;
    double mean = 0.0;
    for (double x : r) mean += x;
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (double x : r) {
        const double d = x - mean;
        var += d * d;
    }
    if (var <= 1e-18) return 1.0;

    double f_corr = 1.0;
    for (size_t lag = 1; lag <= 5 && lag < n; ++lag) {
        double cov = 0.0;
        for (size_t i = lag; i < n; ++i) {
            cov += (r[i] - mean) * (r[i - lag] - mean);
        }
        const double rho = cov / var;
        if (rho > 0.02) {
            f_corr += 2.0 * rho;
        } else {
            break;
        }
    }
    return std::clamp(f_corr, 1.0, 3.0);
}

// Velocity-space Gaussian smoothing (preserves signed values; never clamps negative noise to 0)
std::vector<double> smooth_gaussian_velocity(
    const std::vector<double>& v,
    const std::vector<double>& y,
    double sigma_v_kms,
    double* out_kernel_sum_w2 = nullptr,
    double* out_kernel_w0 = nullptr
) {
    const size_t n = std::min(v.size(), y.size());
    std::vector<double> out(n, 0.0);
    if (out_kernel_sum_w2) *out_kernel_sum_w2 = 1.0;
    if (out_kernel_w0) *out_kernel_w0 = 1.0;
    if (n == 0) return out;
    if (n == 1 || !std::isfinite(sigma_v_kms) || sigma_v_kms <= 0.0) {
        out.assign(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n));
        return out;
    }

    const double span_v = std::abs(v[n - 1] - v[0]);
    if (!std::isfinite(span_v) || span_v <= 0.0) {
        out.assign(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n));
        return out;
    }
    const double dv = std::max(1e-9, span_v / static_cast<double>(n - 1));
    const double sigma_bins = sigma_v_kms / dv;
    if (!std::isfinite(sigma_bins) || sigma_bins < 0.25) {
        out.assign(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n));
        return out;
    }

    const double raw_rad = std::ceil(3.5 * sigma_bins);
    const double max_rad_d = static_cast<double>(n - 1);
    const int radius = static_cast<int>(std::isfinite(raw_rad)
        ? std::clamp(raw_rad, 2.0, std::max(2.0, max_rad_d))
        : max_rad_d);
    const int clamped_radius = std::min<int>(static_cast<int>(n) - 1, radius);
    const size_t klen = static_cast<size_t>(2 * clamped_radius + 1);
    std::vector<double> kernel(klen);
    const double inv_2s2 = 0.5 / (sigma_bins * sigma_bins);
    double k_sum = 0.0;
    for (int k = -clamped_radius; k <= clamped_radius; ++k) {
        const double w = std::exp(-static_cast<double>(k * k) * inv_2s2);
        kernel[static_cast<size_t>(k + clamped_radius)] = w;
        k_sum += w;
    }
    if (k_sum > 0.0) {
        double sum_w2 = 0.0;
        for (double& w : kernel) {
            w /= k_sum;
            sum_w2 += w * w;
        }
        if (out_kernel_sum_w2) *out_kernel_sum_w2 = sum_w2;
        if (out_kernel_w0) *out_kernel_w0 = kernel[static_cast<size_t>(clamped_radius)];
    }

    for (size_t i = 0; i < n; ++i) {
        const int ii = static_cast<int>(i);
        const int j_lo = std::max(0, ii - clamped_radius);
        const int j_hi = std::min(static_cast<int>(n) - 1, ii + clamped_radius);
        double acc = 0.0;
        double wsum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) {
            const double yj = y[static_cast<size_t>(j)];
            if (!std::isfinite(yj)) continue;
            const double w = kernel[static_cast<size_t>(j - ii + clamped_radius)];
            acc += w * yj;
            wsum += w;
        }
        out[i] = (wsum > 0.0) ? (acc / wsum) : (std::isfinite(y[i]) ? y[i] : 0.0);
    }
    return out;
}

struct FitGoodness {
    double rss_null = 0.0;
    double rss_fit = 0.0;
    double sigma_th = 0.0;
    double delta_bic = 0.0;
    double reduced_chi2 = 0.0;
    bool passes_bic = false;
};

FitGoodness evaluate_fit_goodness(
    const std::vector<double>& t_sub,
    const std::vector<double>& model_sub,
    int k_params,
    double peak_amp,
    double f_corr_hint = 0.0
) {
    FitGoodness g;
    const size_t m = std::min(t_sub.size(), model_sub.size());
    if (m == 0) return g;

    std::vector<double> resid(m);
    for (size_t i = 0; i < m; ++i) {
        g.rss_null += t_sub[i] * t_sub[i];
        resid[i] = t_sub[i] - model_sub[i];
        g.rss_fit += resid[i] * resid[i];
    }

    g.sigma_th = estimate_lag6_thermal_rms(resid);
    if (g.sigma_th <= 1e-12) {
        const double mad_r = estimate_mad_std(resid);
        if (mad_r > g.sigma_th) g.sigma_th = mad_r;
    }

    const double f_corr = (f_corr_hint >= 1.0) ? f_corr_hint : estimate_empirical_f_corr(resid);
    const double n_eff = std::max(2.0, static_cast<double>(m) / std::max(1.0, f_corr));

    if (g.rss_null > 1e-18) {
        const double ratio = g.rss_null / std::max(g.rss_fit, 1e-18);
        g.delta_bic = n_eff * std::log(std::max(1e-18, ratio))
                    - static_cast<double>(k_params) * std::log(std::max(2.0, n_eff));
    } else {
        g.delta_bic = -static_cast<double>(k_params) * std::log(std::max(2.0, n_eff));
    }

    const double dof = std::max(1.0, static_cast<double>(m) - static_cast<double>(k_params));
    if (g.sigma_th > 1e-9) {
        g.reduced_chi2 = g.rss_fit / (dof * g.sigma_th * g.sigma_th);
    } else {
        g.reduced_chi2 = (g.rss_fit <= 1e-12) ? 0.0 : (g.rss_fit / dof);
    }

    const bool noiseless_high_r2 = (g.rss_null > 1e-12) &&
                                   (g.rss_fit < 0.25 * g.rss_null) &&
                                   (peak_amp > 0.05) &&
                                   (g.sigma_th < 0.02 * peak_amp);
    g.passes_bic = (g.delta_bic > 6.0) || noiseless_high_r2;
    return g;
}

// Invert 3x3 symmetric matrix A * inv = I
bool invert_3x3(const double A[3][3], double inv[3][3]) {
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    double scale = std::max(1.0, std::max({std::abs(A[0][0]), std::abs(A[1][1]), std::abs(A[2][2])}));
    const double scale3 = scale * scale * scale;
    if (!std::isfinite(det) || !std::isfinite(scale3) || std::abs(det) < 1e-18 * scale3) return false;
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

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(inv[r][c])) return false;
        }
    }
    return true;
}

// Generic NxN linear system solver (N <= 9) with partial pivoting for multi-parameter Levenberg-Marquardt
bool solve_linear_nxn(int n, const double A[9][9], const double b[9], double x[9]) {
    double M[9][10] = {};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n] = b[i];
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double max_v = std::abs(M[col][col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::abs(M[r][col]);
            if (v > max_v) {
                max_v = v;
                piv = r;
            }
        }
        if (!std::isfinite(max_v) || max_v < 1e-15) return false;
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
        if (!std::isfinite(M[i][i]) || std::abs(M[i][i]) < 1e-15) return false;
        x[i] = sum / M[i][i];
        if (!std::isfinite(x[i])) return false;
    }
    return true;
}

void populate_nonparametric_moments(
    const std::vector<double>& v_sub,
    const std::vector<double>& t_sub,
    GaussianFit& fit
) {
    const size_t n = v_sub.size();
    if (n < 3) return;
    const double dv = std::max(1e-9, std::abs(v_sub.back() - v_sub.front()) / static_cast<double>(n - 1));

    // Unbiased signed integrated flux M0 (never clamps negative noise to 0)
    double m0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m0 += t_sub[i] * dv;
    }
    fit.moment0_k_kms = m0;

    // Estimate channel noise via unbiased lag-6 difference MAD
    const double sigma_est = estimate_lag6_thermal_rms(t_sub);

    // Identify signal support via physical velocity-space smoothing to reject outer noise spikes
    double k_sum_w2 = 1.0;
    const double sm_sig = std::max(1.5, 1.5 * dv);
    const std::vector<double> t_sm = smooth_gaussian_velocity(v_sub, t_sub, sm_sig, &k_sum_w2, nullptr);
    const double f_corr_est = estimate_empirical_f_corr(t_sub);
    const double sm_noise = sigma_est * std::sqrt(std::min(1.0, f_corr_est * k_sum_w2));

    size_t sm_pk_idx = 0;
    double sm_pk = t_sm[0];
    for (size_t i = 1; i < n; ++i) {
        if (t_sm[i] > sm_pk) {
            sm_pk = t_sm[i];
            sm_pk_idx = i;
        }
    }

    // For noiseless / very high SNR curves, include all bins with t_sub > 1e-6;
    // for noisy curves, restrict M1/M2 support to contiguous smoothed signal > 2*sm_noise
    // while keeping signed t_sub inside integration (no negative clamping bias).
    double m_sig = 0.0;
    double m1_num = 0.0;
    size_t sig_bins = 0;
    std::vector<uint8_t> sig_mask(n, 0);

    if (sigma_est <= 1e-3) {
        const double thr = std::max(1e-6, 1.5 * sigma_est);
        for (size_t i = 0; i < n; ++i) {
            if (t_sub[i] > thr) {
                sig_mask[i] = 1;
                m_sig += t_sub[i] * dv;
                m1_num += t_sub[i] * v_sub[i] * dv;
                ++sig_bins;
            }
        }
    } else if (sm_pk > 2.0 * sm_noise) {
        const double sm_thr = std::max(2.0 * sm_noise, 1e-4 * sm_pk);
        size_t l = sm_pk_idx, r = sm_pk_idx;
        while (l > 0 && t_sm[l - 1] >= sm_thr) --l;
        while (r + 1 < n && t_sm[r + 1] >= sm_thr) ++r;
        for (size_t i = l; i <= r; ++i) {
            sig_mask[i] = 1;
            m_sig += t_sub[i] * dv;
            m1_num += t_sub[i] * v_sub[i] * dv;
            ++sig_bins;
        }
    }

    if (sig_bins >= 3 && m_sig > 1e-9) {
        fit.moment1_kms = m1_num / m_sig;
        double m2_num = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (sig_mask[i]) {
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

GaussianFit fit_gaussian_impl(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window,
    double f_corr_hint
) {
    GaussianFit res;
    res.method = LineAnalysisMethod::GaussianLS;
    const size_t n_all = std::min({vel_kms.size(), temp_k.size(), hi_window.size()});
    res.fitted_curve.assign(n_all, 0.0);

    std::vector<double> v_sub, t_sub;
    v_sub.reserve(n_all);
    t_sub.reserve(n_all);
    for (size_t i = 0; i < n_all; ++i) {
        if (hi_window[i] && std::isfinite(vel_kms[i]) && std::isfinite(temp_k[i])) {
            v_sub.push_back(vel_kms[i]);
            t_sub.push_back(temp_k[i]);
        }
    }

    const size_t m = v_sub.size();
    if (m < 5) return res;

    auto mm_v = std::minmax_element(v_sub.begin(), v_sub.end());
    const double v_hi_min = *mm_v.first;
    const double v_hi_max = *mm_v.second;
    const double hi_span = std::max(1.0, v_hi_max - v_hi_min);
    const double dv_sub = std::max(1e-6, hi_span / static_cast<double>(m - 1));
    const double sig_min = std::max(0.3, 0.5 * dv_sub);
    const double sig_max = std::min(100.0, std::max(5.0, 0.8 * hi_span));

    // Matched-filter / velocity-smoothed seeding so LM never locks onto a 1-bin noise needle
    const double smooth_sig = std::max(1.5, 2.0 * dv_sub);
    const std::vector<double> t_sm = smooth_gaussian_velocity(v_sub, t_sub, smooth_sig);

    size_t max_idx = 0;
    double sm_peak = t_sm[0];
    double max_raw_t = t_sub[0];
    for (size_t i = 1; i < m; ++i) {
        if (t_sm[i] > sm_peak) {
            sm_peak = t_sm[i];
            max_idx = i;
        }
        if (t_sub[i] > max_raw_t) {
            max_raw_t = t_sub[i];
        }
    }

    if (sm_peak <= 0.0) {
        res.success = false;
        res.amplitude_k = 0.0;
        res.v_center_kms = v_sub[max_idx];
        res.v_peak_kms = v_sub[max_idx];
        res.sigma_kms = 0.0;
        res.fwhm_kms = 0.0;
        return res;
    }

    const double v0_init = std::clamp(v_sub[max_idx], v_hi_min, v_hi_max);
    const double half_max = 0.5 * sm_peak;

    size_t left_idx = max_idx;
    while (left_idx > 0 && t_sm[left_idx - 1] >= half_max) {
        --left_idx;
    }
    size_t right_idx = max_idx;
    while (right_idx + 1 < m && t_sm[right_idx + 1] >= half_max) {
        ++right_idx;
    }

    const double span_half = std::abs(v_sub[right_idx] - v_sub[left_idx]);
    const double fwhm_sm = (right_idx > left_idx)
        ? std::max(span_half, std::max(dv_sub, smooth_sig))
        : std::max(4.0, GAUSSIAN_SIGMA_TO_FWHM * smooth_sig);
    const double sigma_sm = fwhm_sm / GAUSSIAN_SIGMA_TO_FWHM;
    // Deconvolve smoothing width gently for initial sigma seed
    const double sigma_deconv = std::sqrt(std::max(0.25, sigma_sm * sigma_sm - 0.5 * smooth_sig * smooth_sig));
    const double sigma_init = std::clamp(sigma_deconv, std::max(0.5, dv_sub), std::min(sig_max * 0.9, 0.45 * hi_span));

    // Linear least-squares projection for optimal initial amplitude at (v0_init, sigma_init)
    double sum_tg = 0.0, sum_g2 = 0.0;
    const double inv_2s2_init = -0.5 / (sigma_init * sigma_init);
    for (size_t i = 0; i < m; ++i) {
        const double d = v_sub[i] - v0_init;
        const double g = std::exp(inv_2s2_init * d * d);
        sum_tg += t_sub[i] * g;
        sum_g2 += g * g;
    }
    const double a_proj = (sum_g2 > 1e-12) ? (sum_tg / sum_g2) : sm_peak;
    const double a_init = (a_proj > 0.05) ? a_proj : sm_peak;

    double a = a_init;
    double v0 = v0_init;
    double sigma = sigma_init;

    double lambda = 1.0;

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

    const double initial_chi2 = eval_chi2(a, v0, sigma);
    double current_chi2 = initial_chi2;
    const double max_a_bound = std::max(10.0, std::max(a_init, max_raw_t) * 2.5 + 5.0);

    bool converged = false;
    int successful_steps = 0;

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

        // Proportional trust-region step limiter preserving exact Gauss-Newton direction
        double step_scale = 1.0;
        if (std::abs(delta[0]) > 0.5 * std::max(0.2, a)) step_scale = std::min(step_scale, (0.5 * std::max(0.2, a)) / std::abs(delta[0]));
        if (std::abs(delta[1]) > 5.0) step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
        if (std::abs(delta[2]) > 0.5 * sigma) step_scale = std::min(step_scale, (0.5 * sigma) / std::abs(delta[2]));

        delta[0] *= step_scale;
        delta[1] *= step_scale;
        delta[2] *= step_scale;

        double new_a = std::clamp(a + delta[0], 0.04, max_a_bound);
        double new_v0 = std::clamp(v0 + delta[1], v_hi_min, v_hi_max);
        double new_sig = std::clamp(sigma + delta[2], sig_min, sig_max);

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
                break;
            }
        } else {
            if (lambda >= 1e5) break;
            lambda = std::min(1e5, lambda * 3.0);
        }
    }

    // Evaluate model inside hi_window for BIC & reduced chi-squared
    std::vector<double> model_sub(m);
    const double neg_half_inv_sig2 = -0.5 / (sigma * sigma);
    for (size_t i = 0; i < m; ++i) {
        const double diff = v_sub[i] - v0;
        model_sub[i] = a * std::exp(neg_half_inv_sig2 * diff * diff);
    }

    const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 3, a, f_corr_hint);
    res.delta_bic = good.delta_bic;
    res.reduced_chi2 = good.reduced_chi2;
    res.rmse_k = std::sqrt(current_chi2 / static_cast<double>(std::max<size_t>(1, m)));

    const bool fit_valid = (successful_steps > 0 || converged || good.passes_bic) &&
                           good.passes_bic &&
                           (current_chi2 <= initial_chi2 + 1e-12) &&
                           (a > 0.05 && a < max_a_bound * 0.99) &&
                           (sigma > sig_min * 1.05 && sigma < sig_max * 0.98);

    if (fit_valid) {
        res.success = true;
        res.amplitude_k = a;
        res.v_center_kms = v0;
        res.v_peak_kms = v0;
        res.sigma_kms = std::abs(sigma);
        res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * res.sigma_kms;
        res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
        res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
        res.fwhm_half_temp_k = 0.5 * res.amplitude_k;

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
        res.success = false;
        res.amplitude_k = 0.0;
        res.v_center_kms = v0_init;
        res.v_peak_kms = v0_init;
        res.sigma_kms = 0.0;
        res.fwhm_kms = 0.0;
        res.fwhm_left_kms = v0_init;
        res.fwhm_right_kms = v0_init;
        res.fwhm_half_temp_k = 0.0;
        std::fill(res.fitted_curve.begin(), res.fitted_curve.end(), 0.0);
    }

    return res;
}

// Reusable 2-Component Gaussian solver (avoids redundant fit_gaussian in TripleGaussian)
GaussianFit fit_double_gaussian_from_base(
    const std::vector<double>& vel_kms,
    const std::vector<double>& v_sub,
    const std::vector<double>& t_sub,
    double v_hi_min,
    double v_hi_max,
    double dv_sub,
    const GaussianFit& base_fit,
    double f_corr_hint
) {
    const size_t n_all = vel_kms.size();
    const size_t m = v_sub.size();
    const double hi_span = std::max(1.0, v_hi_max - v_hi_min);
    const double sig_min = std::max(0.5, 0.5 * dv_sub);
    const double sig_max = std::min(90.0, std::max(5.0, 0.8 * hi_span));

    // Seed Component 2 from a velocity-smoothed residual (never a single raw noisy bin)
    std::vector<double> res1(m);
    const double inv_s1 = 1.0 / std::max(0.5, base_fit.sigma_kms);
    for (size_t j = 0; j < m; ++j) {
        const double u1 = (v_sub[j] - base_fit.v_center_kms) * inv_s1;
        res1[j] = t_sub[j] - base_fit.amplitude_k * std::exp(-0.5 * u1 * u1);
    }
    const std::vector<double> res1_sm = smooth_gaussian_velocity(v_sub, res1, std::max(1.5, 2.0 * dv_sub));

    const double shift_fallback = std::min(14.0, std::max(3.0, 2.0 * base_fit.sigma_kms));
    const double v2_fallback = std::clamp(
        base_fit.v_center_kms + ((base_fit.v_center_kms >= 0.5 * (v_hi_min + v_hi_max)) ? -shift_fallback : shift_fallback),
        v_hi_min, v_hi_max
    );

    double p[6] = {
        base_fit.amplitude_k * 0.85,
        std::clamp(base_fit.v_center_kms, v_hi_min, v_hi_max),
        std::clamp(base_fit.sigma_kms * 0.80, sig_min, sig_max),
        std::max(0.10, base_fit.amplitude_k * 0.25),
        v2_fallback,
        std::clamp(base_fit.sigma_kms * 0.90, sig_min, sig_max)
    };

    double best_res = -1e9;
    size_t best_j = 0;
    for (size_t j = 0; j < m; ++j) {
        const double dv1 = std::abs(v_sub[j] - base_fit.v_center_kms);
        if (dv1 > 0.65 * base_fit.sigma_kms && res1_sm[j] > best_res) {
            best_res = res1_sm[j];
            best_j = j;
        }
    }
    if (best_res > 0.08 * base_fit.amplitude_k) {
        p[3] = std::max(0.15, best_res);
        p[4] = std::clamp(v_sub[best_j], v_hi_min, v_hi_max);
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

    double cur_chi2 = chi2_2g(p);
    double lambda = 1.0;
    const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

    bool jac_valid = false;
    double JTJ_cached[9][9] = {};
    double JTr_cached[9] = {};

    for (int iter = 0; iter < 90; ++iter) {
        if (!jac_valid) {
            for (int a = 0; a < 6; ++a) {
                JTr_cached[a] = 0.0;
                for (int b = 0; b < 6; ++b) JTJ_cached[a][b] = 0.0;
            }
            for (size_t i = 0; i < m; ++i) {
                const double v = v_sub[i];
                const double isig1 = 1.0 / p[2], u1 = (v - p[1]) * isig1, e1 = std::exp(-0.5 * u1 * u1);
                const double isig2 = 1.0 / p[5], u2 = (v - p[4]) * isig2, e2 = std::exp(-0.5 * u2 * u2);
                const double r = t_sub[i] - (p[0] * e1 + p[3] * e2);
                const double J[6] = {
                    e1, p[0] * e1 * u1 * isig1, p[0] * e1 * u1 * u1 * isig1,
                    e2, p[3] * e2 * u2 * isig2, p[3] * e2 * u2 * u2 * isig2
                };
                for (int a = 0; a < 6; ++a) {
                    JTr_cached[a] += J[a] * r;
                    for (int b = a; b < 6; ++b) JTJ_cached[a][b] += J[a] * J[b];
                }
            }
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < a; ++b) JTJ_cached[a][b] = JTJ_cached[b][a];
            }
            jac_valid = true;
        }

        double A_damped[9][9] = {};
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) A_damped[a][b] = JTJ_cached[a][b];
            A_damped[a][a] += lambda * (JTJ_cached[a][a] + 1e-3);
        }

        double delta[9] = {};
        if (!solve_linear_nxn(6, A_damped, JTr_cached, delta)) {
            if (lambda >= 1e5) break;
            lambda = std::min(1e5, lambda * 4.0);
            continue;
        }

        // Proportional trust-region step scaling
        double step_scale = 1.0;
        const double lim_a0 = 0.5 * std::max(0.2, p[0]);
        const double lim_a3 = 0.5 * std::max(0.2, p[3]);
        if (std::abs(delta[0]) > lim_a0) step_scale = std::min(step_scale, lim_a0 / std::abs(delta[0]));
        if (std::abs(delta[1]) > 5.0)    step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
        if (std::abs(delta[2]) > 0.5 * p[2]) step_scale = std::min(step_scale, (0.5 * p[2]) / std::abs(delta[2]));
        if (std::abs(delta[3]) > lim_a3) step_scale = std::min(step_scale, lim_a3 / std::abs(delta[3]));
        if (std::abs(delta[4]) > 5.0)    step_scale = std::min(step_scale, 5.0 / std::abs(delta[4]));
        if (std::abs(delta[5]) > 0.5 * p[5]) step_scale = std::min(step_scale, (0.5 * p[5]) / std::abs(delta[5]));

        double cand[6] = {
            std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
            std::clamp(p[1] + step_scale * delta[1], v_hi_min, v_hi_max),
            std::clamp(p[2] + step_scale * delta[2], sig_min, sig_max),
            std::clamp(p[3] + step_scale * delta[3], 0.02, max_a),
            std::clamp(p[4] + step_scale * delta[4], v_hi_min, v_hi_max),
            std::clamp(p[5] + step_scale * delta[5], sig_min, sig_max)
        };
        const double cand_chi2 = chi2_2g(cand);
        if (cand_chi2 < cur_chi2) {
            double max_step = 0.0;
            for (int k = 0; k < 6; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
            cur_chi2 = cand_chi2;
            for (int k = 0; k < 6; ++k) p[k] = cand[k];
            jac_valid = false;
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

    std::vector<double> model_sub(m);
    double pk_val = -1e9;
    double pk_vel = p[1];
    for (size_t i = 0; i < m; ++i) {
        model_sub[i] = eval_2g(p, v_sub[i]);
        if (model_sub[i] > pk_val) {
            pk_val = model_sub[i];
            pk_vel = v_sub[i];
        }
    }
    const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 6, std::max(p[0], pk_val), f_corr_hint);

    GaussianFit res = base_fit;
    res.method = LineAnalysisMethod::DoubleGaussian;
    res.delta_bic = good.delta_bic;
    res.reduced_chi2 = good.reduced_chi2;
    res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
    res.success = good.passes_bic && (p[0] > 0.05);

    if (res.success) {
        res.amplitude_k = p[0];
        res.v_center_kms = p[1];
        res.v_peak_kms = pk_vel;
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

        res.fitted_curve.resize(n_all);
        res.component1_curve.resize(n_all);
        res.component2_curve.resize(n_all);
        for (size_t i = 0; i < n_all; ++i) {
            double c1 = 0.0, c2 = 0.0;
            res.fitted_curve[i] = eval_2g(p, vel_kms[i], &c1, &c2);
            res.component1_curve[i] = c1;
            res.component2_curve[i] = c2;
        }
    } else {
        res.amplitude_k = 0.0;
        res.sigma_kms = 0.0;
        res.fwhm_kms = 0.0;
        res.fwhm_left_kms = 0.0;
        res.fwhm_right_kms = 0.0;
        res.fwhm_half_temp_k = 0.0;
        res.has_secondary = false;
        res.fitted_curve.assign(n_all, 0.0);
        res.component1_curve.clear();
        res.component2_curve.clear();
    }
    return res;
}

GaussianFit fit_line_profile_impl(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window,
    LineAnalysisMethod method,
    double f_corr_hint
) {
    // 1. Always run single Gaussian fit as baseline seed & fallback
    GaussianFit base_fit = fit_gaussian_impl(vel_kms, temp_k, hi_window, f_corr_hint);
    base_fit.method = method;

    const size_t n_all = std::min({vel_kms.size(), temp_k.size(), hi_window.size()});
    std::vector<double> v_sub, t_sub;
    v_sub.reserve(n_all);
    t_sub.reserve(n_all);
    for (size_t i = 0; i < n_all; ++i) {
        if (hi_window[i] && std::isfinite(vel_kms[i]) && std::isfinite(temp_k[i])) {
            v_sub.push_back(vel_kms[i]);
            t_sub.push_back(temp_k[i]);
        }
    }
    populate_nonparametric_moments(v_sub, t_sub, base_fit);

    if (v_sub.size() < 7 || method == LineAnalysisMethod::GaussianLS) {
        return base_fit;
    }

    const size_t m = v_sub.size();
    auto mm_v = std::minmax_element(v_sub.begin(), v_sub.end());
    const double v_hi_min = *mm_v.first;
    const double v_hi_max = *mm_v.second;
    const double hi_span = std::max(1.0, v_hi_max - v_hi_min);
    const double dv_sub = std::max(1e-6, hi_span / static_cast<double>(m - 1));
    const double sig_min = std::max(0.5, 0.5 * dv_sub);
    const double sig_max = std::min(90.0, std::max(5.0, 0.8 * hi_span));

    // -------------------------------------------------------------------------
    // Method 5: Non-Parametric Spectral Moments (M0 / M1 / M2 + velocity Gaussian smoothing)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::SpectralMoments) {
        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::SpectralMoments;

        const double dv_all = (n_all > 1) ? std::max(1e-6, std::abs(vel_kms[1] - vel_kms[0])) : dv_sub;
        const double sm_sigma_v = std::max(1.5, 1.5 * dv_all);
        double kernel_sum_w2 = 1.0;
        double kernel_w0 = 1.0;
        res.fitted_curve = smooth_gaussian_velocity(vel_kms, temp_k, sm_sigma_v, &kernel_sum_w2, &kernel_w0);

        size_t pk_i = 0;
        double pk_v = -1e9;
        bool has_hi = false;
        std::vector<double> model_sub;
        model_sub.reserve(m);
        for (size_t i = 0; i < n_all; ++i) {
            if (hi_window[i] && std::isfinite(vel_kms[i]) && std::isfinite(temp_k[i])) {
                model_sub.push_back(res.fitted_curve[i]);
                if (!has_hi || res.fitted_curve[i] > pk_v) {
                    pk_v = res.fitted_curve[i];
                    pk_i = i;
                    has_hi = true;
                }
            }
        }

        // Lag-6 diff MAD thermal noise + exact kernel noise propagation
        const double sigma_ch = estimate_lag6_thermal_rms(t_sub);
        const double f_corr = (f_corr_hint >= 1.0) ? f_corr_hint : estimate_empirical_f_corr(t_sub);
        const double smoothed_noise = sigma_ch * std::sqrt(std::min(1.0, f_corr * kernel_sum_w2));

        const double n_eff = std::max(2.0, static_cast<double>(m) / std::max(1.0, f_corr));
        const int k_eff = std::clamp(static_cast<int>(std::lround(n_eff * kernel_w0)), 3, std::max(3, static_cast<int>(m) - 1));
        const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, k_eff, pk_v, f_corr);
        res.delta_bic = good.delta_bic;
        res.reduced_chi2 = good.reduced_chi2;
        res.rmse_k = std::sqrt(good.rss_fit / static_cast<double>(std::max<size_t>(1, m)));
        res.v_peak_kms = has_hi ? vel_kms[pk_i] : 0.0;

        const bool sig_detected = has_hi && (pk_v > 0.05) &&
                                  (smoothed_noise <= 1e-6 || pk_v > 3.0 * smoothed_noise);
        if (sig_detected) {
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
        } else {
            res.success = false;
            res.amplitude_k = 0.0;
            res.sigma_kms = 0.0;
            res.fwhm_kms = 0.0;
            res.fwhm_half_temp_k = 0.0;
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 2: Double Gaussian (6-parameter Levenberg-Marquardt)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::DoubleGaussian && base_fit.success) {
        return fit_double_gaussian_from_base(vel_kms, v_sub, t_sub, v_hi_min, v_hi_max, dv_sub, base_fit, f_corr_hint);
    }

    // -------------------------------------------------------------------------
    // Method 6: Triple Gaussian (9-parameter Levenberg-Marquardt for multi-arm / CNM+WNM)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::TripleGaussian && base_fit.success) {
        // Reuse base_fit directly (no redundant second fit_gaussian call)
        GaussianFit dg_seed = fit_double_gaussian_from_base(vel_kms, v_sub, t_sub, v_hi_min, v_hi_max, dv_sub, base_fit, f_corr_hint);
        const double dg_a1 = dg_seed.success ? dg_seed.amplitude_k : base_fit.amplitude_k;
        const double dg_v1 = dg_seed.success ? dg_seed.v_center_kms : base_fit.v_center_kms;
        const double dg_s1 = dg_seed.success ? dg_seed.sigma_kms : base_fit.sigma_kms;
        const double dg_a2 = dg_seed.success ? dg_seed.amp2_k : (base_fit.amplitude_k * 0.25);
        const double dg_v2 = dg_seed.success ? dg_seed.v2_center_kms : std::clamp(base_fit.v_center_kms + 12.0, v_hi_min, v_hi_max);
        const double dg_s2 = dg_seed.success ? dg_seed.sigma2_kms : base_fit.sigma_kms;

        // Compute smoothed 2-Gaussian residual to seed Component 3 robustly
        std::vector<double> res2(m);
        const double is1 = 1.0 / std::max(0.5, dg_s1);
        const double is2 = 1.0 / std::max(0.5, dg_s2);
        for (size_t j = 0; j < m; ++j) {
            const double d1 = (v_sub[j] - dg_v1) * is1;
            const double d2 = (v_sub[j] - dg_v2) * is2;
            res2[j] = t_sub[j] - (dg_a1 * std::exp(-0.5 * d1 * d1) + dg_a2 * std::exp(-0.5 * d2 * d2));
        }
        const std::vector<double> res2_sm = smooth_gaussian_velocity(v_sub, res2, std::max(1.5, 2.0 * dv_sub));

        double p[9] = {
            std::max(0.10, dg_a1 * 0.90),
            std::clamp(dg_v1, v_hi_min, v_hi_max),
            std::clamp(dg_s1, sig_min, sig_max),
            std::max(0.10, dg_a2 * 0.90),
            std::clamp(dg_v2, v_hi_min, v_hi_max),
            std::clamp(dg_s2, sig_min, sig_max),
            std::max(0.10, base_fit.amplitude_k * 0.20),
            std::clamp(base_fit.moment1_kms, v_hi_min, v_hi_max),
            std::clamp(base_fit.sigma_kms * 1.45, std::max(1.5, sig_min), sig_max)
        };

        double best_res3 = -1e9;
        size_t best_j3 = 0;
        for (size_t j = 0; j < m; ++j) {
            const double d1 = std::abs(v_sub[j] - dg_v1) * is1;
            const double d2 = std::abs(v_sub[j] - dg_v2) * is2;
            if (d1 > 0.65 && d2 > 0.65 && res2_sm[j] > best_res3) {
                best_res3 = res2_sm[j];
                best_j3 = j;
            }
        }
        if (best_res3 > 0.05 * base_fit.amplitude_k) {
            p[6] = std::max(0.12, best_res3);
            p[7] = std::clamp(v_sub[best_j3], v_hi_min, v_hi_max);
            p[8] = std::clamp(base_fit.sigma_kms * 0.75, std::max(1.0, sig_min), std::min(50.0, sig_max));
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

        double cur_chi2 = chi2_3g(p);
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        bool jac_valid = false;
        double JTJ_cached[9][9] = {};
        double JTr_cached[9] = {};

        for (int iter = 0; iter < 100; ++iter) {
            if (!jac_valid) {
                for (int a = 0; a < 9; ++a) {
                    JTr_cached[a] = 0.0;
                    for (int b = 0; b < 9; ++b) JTJ_cached[a][b] = 0.0;
                }
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
                        JTr_cached[a] += J[a] * r;
                        for (int b = a; b < 9; ++b) JTJ_cached[a][b] += J[a] * J[b];
                    }
                }
                for (int a = 0; a < 9; ++a) {
                    for (int b = 0; b < a; ++b) JTJ_cached[a][b] = JTJ_cached[b][a];
                }
                jac_valid = true;
            }

            double A_damped[9][9] = {};
            for (int a = 0; a < 9; ++a) {
                for (int b = 0; b < 9; ++b) A_damped[a][b] = JTJ_cached[a][b];
                A_damped[a][a] += lambda * (JTJ_cached[a][a] + 1e-3);
            }

            double delta[9] = {};
            if (!solve_linear_nxn(9, A_damped, JTr_cached, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }

            double step_scale = 1.0;
            for (int a_idx : {0, 3, 6}) {
                const double lim = 0.5 * std::max(0.2, p[a_idx]);
                if (std::abs(delta[a_idx]) > lim) step_scale = std::min(step_scale, lim / std::abs(delta[a_idx]));
            }
            for (int c_idx : {1, 4, 7}) {
                if (std::abs(delta[c_idx]) > 5.0) step_scale = std::min(step_scale, 5.0 / std::abs(delta[c_idx]));
            }
            for (int s_idx : {2, 5, 8}) {
                if (std::abs(delta[s_idx]) > 0.5 * p[s_idx]) step_scale = std::min(step_scale, (0.5 * p[s_idx]) / std::abs(delta[s_idx]));
            }

            double cand[9] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], v_hi_min, v_hi_max),
                std::clamp(p[2] + step_scale * delta[2], sig_min, sig_max),
                std::clamp(p[3] + step_scale * delta[3], 0.02, max_a),
                std::clamp(p[4] + step_scale * delta[4], v_hi_min, v_hi_max),
                std::clamp(p[5] + step_scale * delta[5], sig_min, sig_max),
                std::clamp(p[6] + step_scale * delta[6], 0.02, max_a),
                std::clamp(p[7] + step_scale * delta[7], v_hi_min, v_hi_max),
                std::clamp(p[8] + step_scale * delta[8], sig_min, sig_max)
            };
            const double cand_chi2 = chi2_3g(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 9; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 9; ++k) p[k] = cand[k];
                jac_valid = false;
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

        std::vector<double> model_sub(m);
        double pk_val = -1e9;
        double pk_vel = p[1];
        for (size_t i = 0; i < m; ++i) {
            model_sub[i] = eval_3g(p, v_sub[i]);
            if (model_sub[i] > pk_val) {
                pk_val = model_sub[i];
                pk_vel = v_sub[i];
            }
        }
        const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 9, std::max(p[0], pk_val), f_corr_hint);

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::TripleGaussian;
        res.delta_bic = good.delta_bic;
        res.reduced_chi2 = good.reduced_chi2;
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.success = good.passes_bic && (p[0] > 0.05);

        if (res.success) {
            res.amplitude_k = p[0];
            res.v_center_kms = p[1];
            res.v_peak_kms = pk_vel;
            res.sigma_kms = p[2];
            res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * p[2];
            res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
            res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
            res.fwhm_half_temp_k = 0.5 * res.amplitude_k;

            res.has_secondary = (p[3] > 0.04 * p[0] &&
                (std::abs(p[4] - p[1]) > 1.5 || std::abs(p[5] - p[2]) > 1.5));
            res.amp2_k = p[3];
            res.v2_center_kms = p[4];
            res.sigma2_kms = p[5];
            res.fwhm2_kms = GAUSSIAN_SIGMA_TO_FWHM * p[5];

            const bool distinct_from_1 = (std::abs(p[7] - p[1]) > 1.5 || std::abs(p[8] - p[2]) > 2.0);
            const bool distinct_from_2 = (std::abs(p[7] - p[4]) > 1.5 || std::abs(p[8] - p[5]) > 2.0);
            res.has_tertiary = (p[6] > 0.03 * p[0] && distinct_from_1 && distinct_from_2);
            res.amp3_k = p[6];
            res.v3_center_kms = p[7];
            res.sigma3_kms = p[8];
            res.fwhm3_kms = GAUSSIAN_SIGMA_TO_FWHM * p[8];

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
        } else {
            res.amplitude_k = 0.0;
            res.sigma_kms = 0.0;
            res.fwhm_kms = 0.0;
            res.fwhm_left_kms = 0.0;
            res.fwhm_right_kms = 0.0;
            res.fwhm_half_temp_k = 0.0;
            res.has_secondary = false;
            res.has_tertiary = false;
            res.fitted_curve.assign(n_all, 0.0);
            res.component1_curve.clear();
            res.component2_curve.clear();
            res.component3_curve.clear();
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 3: Pseudo-Voigt Profile (4-parameter L-M with exact analytical Jacobian)
    // T(v) = A * [ (1 - eta) * exp(-0.5 * ((v-v0)/sigma)^2) + eta / (1 + ((v-v0)/gamma)^2) ]
    // where gamma = sigma * sqrt(2 * ln(2)) so Gaussian & Lorentzian share equal FWHM.
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::VoigtProfile && base_fit.success) {
        double p[4] = {
            base_fit.amplitude_k,
            std::clamp(base_fit.v_center_kms, v_hi_min, v_hi_max),
            std::clamp(base_fit.sigma_kms, sig_min, sig_max),
            0.20
        };
        constexpr double kSqrt2Ln2 = 1.1774100225154747;

        auto eval_voigt = [&](const double q[4], double v) {
            const double d = v - q[1];
            const double u = d / q[2];
            const double g = std::exp(-0.5 * u * u);
            const double w = d / (q[2] * kSqrt2Ln2);
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

        double cur_chi2 = chi2_voigt(p);
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        bool jac_valid = false;
        double JTJ_cached[9][9] = {};
        double JTr_cached[9] = {};

        for (int iter = 0; iter < 80; ++iter) {
            if (!jac_valid) {
                for (int a = 0; a < 4; ++a) {
                    JTr_cached[a] = 0.0;
                    for (int b = 0; b < 4; ++b) JTJ_cached[a][b] = 0.0;
                }
                const double inv_sig = 1.0 / p[2];
                const double inv_gam = inv_sig / kSqrt2Ln2;
                const double one_m_eta = 1.0 - p[3];

                for (size_t i = 0; i < m; ++i) {
                    const double d = v_sub[i] - p[1];
                    const double u = d * inv_sig;
                    const double g = std::exp(-0.5 * u * u);
                    const double w = d * inv_gam;
                    const double lor = 1.0 / (1.0 + w * w);
                    const double lor2 = lor * lor;
                    const double mix = one_m_eta * g + p[3] * lor;
                    const double r = t_sub[i] - p[0] * mix;

                    // Exact analytical partial derivatives of T(v) wrt [A, v0, sigma, eta]
                    const double J[4] = {
                        mix,
                        p[0] * (one_m_eta * (u * inv_sig) * g + p[3] * (2.0 * w * inv_gam) * lor2),
                        p[0] * inv_sig * (one_m_eta * (u * u) * g + p[3] * (2.0 * w * w) * lor2),
                        p[0] * (lor - g)
                    };
                    for (int a = 0; a < 4; ++a) {
                        JTr_cached[a] += J[a] * r;
                        for (int b = a; b < 4; ++b) JTJ_cached[a][b] += J[a] * J[b];
                    }
                }
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < a; ++b) JTJ_cached[a][b] = JTJ_cached[b][a];
                }
                jac_valid = true;
            }

            double A_damped[9][9] = {};
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) A_damped[a][b] = JTJ_cached[a][b];
                A_damped[a][a] += lambda * (JTJ_cached[a][a] + 1e-3);
            }

            double delta[9] = {};
            if (!solve_linear_nxn(4, A_damped, JTr_cached, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }

            double step_scale = 1.0;
            const double lim_a = 0.5 * std::max(0.2, p[0]);
            if (std::abs(delta[0]) > lim_a)      step_scale = std::min(step_scale, lim_a / std::abs(delta[0]));
            if (std::abs(delta[1]) > 5.0)        step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
            if (std::abs(delta[2]) > 0.5 * p[2]) step_scale = std::min(step_scale, (0.5 * p[2]) / std::abs(delta[2]));
            if (std::abs(delta[3]) > 0.25)       step_scale = std::min(step_scale, 0.25 / std::abs(delta[3]));

            double cand[4] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], v_hi_min, v_hi_max),
                std::clamp(p[2] + step_scale * delta[2], sig_min, sig_max),
                std::clamp(p[3] + step_scale * delta[3], 0.0, 1.0)
            };
            const double cand_chi2 = chi2_voigt(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 4; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 4; ++k) p[k] = cand[k];
                jac_valid = false;
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        std::vector<double> model_sub(m);
        for (size_t i = 0; i < m; ++i) model_sub[i] = eval_voigt(p, v_sub[i]);
        const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 4, p[0], f_corr_hint);

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::VoigtProfile;
        res.delta_bic = good.delta_bic;
        res.reduced_chi2 = good.reduced_chi2;
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.success = good.passes_bic && (p[0] > 0.05);

        if (res.success) {
            res.amplitude_k = p[0];
            res.v_center_kms = p[1];
            res.v_peak_kms = p[1];
            res.sigma_kms = p[2];
            res.fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * p[2];
            res.fwhm_left_kms = res.v_center_kms - 0.5 * res.fwhm_kms;
            res.fwhm_right_kms = res.v_center_kms + 0.5 * res.fwhm_kms;
            res.fwhm_half_temp_k = 0.5 * res.amplitude_k;
            res.voigt_eta = p[3];
            res.lorentz_gamma_kms = p[2] * kSqrt2Ln2;
            res.fitted_curve.resize(n_all);
            for (size_t i = 0; i < n_all; ++i) {
                res.fitted_curve[i] = eval_voigt(p, vel_kms[i]);
            }
        } else {
            res.amplitude_k = 0.0;
            res.sigma_kms = 0.0;
            res.fwhm_kms = 0.0;
            res.fwhm_left_kms = 0.0;
            res.fwhm_right_kms = 0.0;
            res.fwhm_half_temp_k = 0.0;
            res.fitted_curve.assign(n_all, 0.0);
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 4: Gauss-Hermite Asymmetric Line Profile (5-parameter L-M, analytical Jacobian)
    // T(v) = A * exp(-0.5 * y^2) * [ 1 + h3 * H3(y) + h4 * H4(y) ], y = (v - v0) / sigma
    // H3(y) = (2*y^3 - 3*y) / sqrt(3),  H4(y) = (4*y^4 - 12*y^2 + 3) / sqrt(24)
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::GaussHermite && base_fit.success) {
        double p[5] = {
            base_fit.amplitude_k,
            std::clamp(base_fit.v_center_kms, v_hi_min, v_hi_max),
            std::clamp(base_fit.sigma_kms, sig_min, sig_max),
            0.0,
            0.0
        };
        constexpr double kInvSqrt3  = 0.5773502691896258;
        constexpr double kSqrt3     = 1.7320508075688772;
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

        double cur_chi2 = chi2_gh(p);
        double lambda = 1.0;
        const double max_a = std::max(10.0, base_fit.amplitude_k * 2.5 + 5.0);

        bool jac_valid = false;
        double JTJ_cached[9][9] = {};
        double JTr_cached[9] = {};

        for (int iter = 0; iter < 80; ++iter) {
            if (!jac_valid) {
                for (int a = 0; a < 5; ++a) {
                    JTr_cached[a] = 0.0;
                    for (int b = 0; b < 5; ++b) JTJ_cached[a][b] = 0.0;
                }
                const double inv_sig = 1.0 / p[2];
                for (size_t i = 0; i < m; ++i) {
                    const double y = (v_sub[i] - p[1]) * inv_sig;
                    const double y2 = y * y;
                    const double g = std::exp(-0.5 * y2);
                    const double H3 = (2.0 * y2 * y - 3.0 * y) * kInvSqrt3;
                    const double H4 = (4.0 * y2 * y2 - 12.0 * y2 + 3.0) * kInvSqrt24;
                    const double dH3 = kSqrt3 * (2.0 * y2 - 1.0);
                    const double dH4 = (16.0 * y2 * y - 24.0 * y) * kInvSqrt24;

                    const double P = 1.0 + p[3] * H3 + p[4] * H4;
                    const double dP = p[3] * dH3 + p[4] * dH4;
                    const double f0 = p[0] * g * P;
                    const double r = t_sub[i] - f0;

                    // Exact analytical partial derivatives of T(v) wrt [A, v0, sigma, h3, h4]
                    const double dT_dv0 = p[0] * g * inv_sig * (y * P - dP);
                    const double J[5] = {
                        g * P,
                        dT_dv0,
                        y * dT_dv0,
                        p[0] * g * H3,
                        p[0] * g * H4
                    };
                    for (int a = 0; a < 5; ++a) {
                        JTr_cached[a] += J[a] * r;
                        for (int b = a; b < 5; ++b) JTJ_cached[a][b] += J[a] * J[b];
                    }
                }
                for (int a = 0; a < 5; ++a) {
                    for (int b = 0; b < a; ++b) JTJ_cached[a][b] = JTJ_cached[b][a];
                }
                jac_valid = true;
            }

            double A_damped[9][9] = {};
            for (int a = 0; a < 5; ++a) {
                for (int b = 0; b < 5; ++b) A_damped[a][b] = JTJ_cached[a][b];
                A_damped[a][a] += lambda * (JTJ_cached[a][a] + 1e-3);
            }

            double delta[9] = {};
            if (!solve_linear_nxn(5, A_damped, JTr_cached, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }

            double step_scale = 1.0;
            const double lim_a = 0.5 * std::max(0.2, p[0]);
            if (std::abs(delta[0]) > lim_a)      step_scale = std::min(step_scale, lim_a / std::abs(delta[0]));
            if (std::abs(delta[1]) > 5.0)        step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
            if (std::abs(delta[2]) > 0.4 * p[2]) step_scale = std::min(step_scale, (0.4 * p[2]) / std::abs(delta[2]));
            if (std::abs(delta[3]) > 0.12)       step_scale = std::min(step_scale, 0.12 / std::abs(delta[3]));
            if (std::abs(delta[4]) > 0.12)       step_scale = std::min(step_scale, 0.12 / std::abs(delta[4]));

            // Clamp |h3|, |h4| <= 0.25 to prevent unphysical negative wing lobes
            double cand[5] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], v_hi_min, v_hi_max),
                std::clamp(p[2] + step_scale * delta[2], sig_min, sig_max),
                std::clamp(p[3] + step_scale * delta[3], -0.25, 0.25),
                std::clamp(p[4] + step_scale * delta[4], -0.25, 0.25)
            };
            const double cand_chi2 = chi2_gh(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 5; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 5; ++k) p[k] = cand[k];
                jac_valid = false;
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-4) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Dense 401-point evaluation for exact curve peak velocity and inward-flank FWHM
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
        double v_peak_exact = v_lo + pk_idx * v_step;
        if (pk_idx > 0 && pk_idx + 1 < kGridN) {
            const double ym = gh_vals[pk_idx - 1], y0 = gh_vals[pk_idx], yp = gh_vals[pk_idx + 1];
            const double denom = ym - 2.0 * y0 + yp;
            if (std::abs(denom) > 1e-12) {
                const double frac = std::clamp(0.5 * (ym - yp) / denom, -0.5, 0.5);
                v_peak_exact = v_lo + (static_cast<double>(pk_idx) + frac) * v_step;
            }
        }

        double fwhm_exact = GAUSSIAN_SIGMA_TO_FWHM * p[2];
        double v_left = p[1] - 0.5 * fwhm_exact;
        double v_right = p[1] + 0.5 * fwhm_exact;
        const double half_h = 0.5 * pk_val;
        if (pk_val > 0.05) {
            // Inward flank search captures full outer width cleanly on boxy (h4 < 0) profiles
            int l_g = 0;
            while (l_g + 1 < kGridN && gh_vals[l_g] < half_h) ++l_g;
            int r_g = kGridN - 1;
            while (r_g > 0 && gh_vals[r_g] < half_h) --r_g;

            auto grid_v = [&](int idx) { return v_lo + idx * v_step; };
            if (l_g > 0 && l_g < kGridN) {
                const double y0 = gh_vals[l_g - 1], y1 = gh_vals[l_g];
                const double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_left = grid_v(l_g - 1) + t * v_step;
            } else {
                v_left = grid_v(l_g);
            }
            if (r_g >= 0 && r_g + 1 < kGridN) {
                const double y0 = gh_vals[r_g], y1 = gh_vals[r_g + 1];
                const double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_right = grid_v(r_g) + t * v_step;
            } else {
                v_right = grid_v(r_g);
            }
            if (v_right > v_left) fwhm_exact = v_right - v_left;
        }

        std::vector<double> model_sub(m);
        for (size_t i = 0; i < m; ++i) model_sub[i] = eval_gh(p, v_sub[i]);
        const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 5, pk_val, f_corr_hint);

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::GaussHermite;
        res.delta_bic = good.delta_bic;
        res.reduced_chi2 = good.reduced_chi2;
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.success = good.passes_bic && (pk_val > 0.05);

        if (res.success) {
            res.amplitude_k = pk_val;
            res.v_center_kms = p[1];
            res.v_peak_kms = v_peak_exact;
            res.sigma_kms = p[2];
            res.fwhm_kms = fwhm_exact;
            res.fwhm_left_kms = v_left;
            res.fwhm_right_kms = v_right;
            res.fwhm_half_temp_k = half_h;
            res.gh_h3 = p[3];
            res.gh_h4 = p[4];
            res.fitted_curve.resize(n_all);
            for (size_t i = 0; i < n_all; ++i) {
                res.fitted_curve[i] = eval_gh(p, vel_kms[i]);
            }
        } else {
            res.amplitude_k = 0.0;
            res.sigma_kms = 0.0;
            res.fwhm_kms = 0.0;
            res.fwhm_left_kms = 0.0;
            res.fwhm_right_kms = 0.0;
            res.fwhm_half_temp_k = 0.0;
            res.fitted_curve.assign(n_all, 0.0);
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Method 7: Generalized Busy Function (Westmeier et al. 2014, MNRAS 438, 1176)
    // Scale-invariant parameterization with dimensionless trough curvature c_tilde = c * w^2 in [0, 4]:
    // B(v) = (a / 4) * [erf(b1 * (w + dv)) + 1] * [erf(b2 * (w - dv)) + 1] * (1 + c_tilde * (dv / w)^2)
    // 6 parameters: p = [a, v0, w, b1, b2, c_tilde]
    // -------------------------------------------------------------------------
    if (method == LineAnalysisMethod::BusyFunction && base_fit.success) {
        const double v0_seed = std::clamp(
            (base_fit.moment0_k_kms > 1e-3) ? base_fit.moment1_kms : base_fit.v_center_kms,
            v_hi_min, v_hi_max
        );
        const double sig_seed = (base_fit.moment2_sigma_kms > 1.0) ? base_fit.moment2_sigma_kms : base_fit.sigma_kms;
        const double w_min = std::max(1.0, dv_sub);
        const double w_max = std::min(160.0, std::max(10.0, 0.8 * hi_span));
        const double w_seed = std::clamp(0.5 * GAUSSIAN_SIGMA_TO_FWHM * sig_seed, w_min, std::min(120.0, 0.45 * hi_span));
        const double b_seed = std::clamp(2.0 / std::max(1.0, base_fit.sigma_kms), 0.05, 1.5);

        // Seed double-horn / central trough from velocity-smoothed profile (immune to raw single-bin spikes)
        const std::vector<double> t_sm = smooth_gaussian_velocity(v_sub, t_sub, std::max(1.5, 2.0 * dv_sub));
        double center_temp = 0.0;
        int center_cnt = 0;
        double max_sm_t = 0.0;
        for (size_t i = 0; i < m; ++i) {
            if (t_sm[i] > max_sm_t) max_sm_t = t_sm[i];
            if (std::abs(v_sub[i] - v0_seed) < 0.25 * w_seed) {
                center_temp += t_sm[i];
                ++center_cnt;
            }
        }
        if (center_cnt > 0) center_temp /= static_cast<double>(center_cnt);
        const bool has_trough = (center_cnt > 0 && max_sm_t > 0.05 && center_temp < 0.85 * max_sm_t);
        const double a_seed = has_trough ? std::max(0.2, center_temp) : std::max(0.2, std::max(base_fit.amplitude_k, max_sm_t));
        const double c_tilde_seed = has_trough
            ? std::clamp((max_sm_t / std::max(0.2, a_seed) - 1.0) / 0.6, 0.05, 2.5)
            : 0.05;

        double p[6] = { a_seed, v0_seed, w_seed, b_seed, b_seed, c_tilde_seed };

        auto eval_busy = [&](const double q[6], double v) {
            const double dv = v - q[1];
            const double e1 = std::erf(q[3] * (q[2] + dv)) + 1.0;
            const double e2 = std::erf(q[4] * (q[2] - dv)) + 1.0;
            const double u = dv / q[2];
            const double poly = 1.0 + q[5] * u * u;
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

        double cur_chi2 = chi2_busy(p);
        double lambda = 1.0;
        const double max_a = std::max(15.0, std::max(max_sm_t, base_fit.amplitude_k) * 3.0 + 5.0);

        bool jac_valid = false;
        double JTJ_cached[9][9] = {};
        double JTr_cached[9] = {};

        for (int iter = 0; iter < 90; ++iter) {
            if (!jac_valid) {
                for (int a = 0; a < 6; ++a) {
                    JTr_cached[a] = 0.0;
                    for (int b = 0; b < 6; ++b) JTJ_cached[a][b] = 0.0;
                }
                const double inv_w = 1.0 / p[2];
                for (size_t i = 0; i < m; ++i) {
                    const double dv = v_sub[i] - p[1];
                    const double wp = p[2] + dv;
                    const double wm = p[2] - dv;
                    const double x1 = p[3] * wp;
                    const double x2 = p[4] * wm;
                    const double e1 = std::erf(x1) + 1.0;
                    const double e2 = std::erf(x2) + 1.0;
                    const double d1 = kTwoOverSqrtPi * std::exp(-x1 * x1);
                    const double d2 = kTwoOverSqrtPi * std::exp(-x2 * x2);
                    const double u = dv * inv_w;
                    const double u2 = u * u;
                    const double poly = 1.0 + p[5] * u2;

                    const double f0 = 0.25 * p[0] * e1 * e2 * poly;
                    const double r = t_sub[i] - f0;

                    const double q_a = 0.25 * p[0];
                    const double e1e2 = e1 * e2;
                    const double J[6] = {
                        0.25 * e1e2 * poly,
                        q_a * ((-p[3] * d1 * e2 + e1 * p[4] * d2) * poly - e1e2 * (2.0 * p[5] * u * inv_w)),
                        q_a * (( p[3] * d1 * e2 + e1 * p[4] * d2) * poly - e1e2 * (2.0 * p[5] * u2 * inv_w)),
                        q_a * (d1 * wp * e2 * poly),
                        q_a * (e1 * d2 * wm * poly),
                        q_a * (e1e2 * u2)
                    };
                    for (int a = 0; a < 6; ++a) {
                        JTr_cached[a] += J[a] * r;
                        for (int b = a; b < 6; ++b) JTJ_cached[a][b] += J[a] * J[b];
                    }
                }
                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < a; ++b) JTJ_cached[a][b] = JTJ_cached[b][a];
                }
                jac_valid = true;
            }

            double A_damped[9][9] = {};
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) A_damped[a][b] = JTJ_cached[a][b];
                A_damped[a][a] += lambda * (JTJ_cached[a][a] + 1e-3);
            }

            double delta[9] = {};
            if (!solve_linear_nxn(6, A_damped, JTr_cached, delta)) {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 4.0);
                continue;
            }

            double step_scale = 1.0;
            const double lim_a = 0.5 * std::max(0.2, p[0]);
            if (std::abs(delta[0]) > lim_a)       step_scale = std::min(step_scale, lim_a / std::abs(delta[0]));
            if (std::abs(delta[1]) > 5.0)         step_scale = std::min(step_scale, 5.0 / std::abs(delta[1]));
            if (std::abs(delta[2]) > 0.4 * p[2])  step_scale = std::min(step_scale, (0.4 * p[2]) / std::abs(delta[2]));
            if (std::abs(delta[3]) > 0.35 * p[3]) step_scale = std::min(step_scale, (0.35 * p[3]) / std::abs(delta[3]));
            if (std::abs(delta[4]) > 0.35 * p[4]) step_scale = std::min(step_scale, (0.35 * p[4]) / std::abs(delta[4]));
            if (std::abs(delta[5]) > 0.5)         step_scale = std::min(step_scale, 0.5 / std::abs(delta[5]));

            double cand[6] = {
                std::clamp(p[0] + step_scale * delta[0], 0.05, max_a),
                std::clamp(p[1] + step_scale * delta[1], v_hi_min, v_hi_max),
                std::clamp(p[2] + step_scale * delta[2], w_min, w_max),
                std::clamp(p[3] + step_scale * delta[3], 0.02, 5.0),
                std::clamp(p[4] + step_scale * delta[4], 0.02, 5.0),
                std::clamp(p[5] + step_scale * delta[5], 0.0, 4.0)
            };
            const double cand_chi2 = chi2_busy(cand);
            if (cand_chi2 < cur_chi2) {
                double max_step = 0.0;
                for (int k = 0; k < 6; ++k) max_step = std::max(max_step, std::abs(cand[k] - p[k]));
                cur_chi2 = cand_chi2;
                for (int k = 0; k < 6; ++k) p[k] = cand[k];
                jac_valid = false;
                lambda = std::max(1e-7, lambda * 0.5);
                if (max_step < 1e-5) break;
            } else {
                if (lambda >= 1e5) break;
                lambda = std::min(1e5, lambda * 3.0);
            }
        }

        // Dense 401-point evaluation with guaranteed-decayed flank bounds so l_g/r_g always cross cleanly
        constexpr int kGridN = 401;
        const double left_span  = std::max(25.0, 1.25 * (p[2] + 6.0 / std::max(0.02, p[3])));
        const double right_span = std::max(25.0, 1.25 * (p[2] + 6.0 / std::max(0.02, p[4])));
        const double v_lo = p[1] - std::min(500.0, left_span);
        const double v_hi = p[1] + std::min(500.0, right_span);
        const double v_step = (v_hi - v_lo) / (kGridN - 1);
        double bf_vals[kGridN];
        double pk_val = -1e9;
        int pk_idx = kGridN / 2;
        for (int g = 0; g < kGridN; ++g) {
            bf_vals[g] = eval_busy(p, v_lo + g * v_step);
            if (bf_vals[g] > pk_val) {
                pk_val = bf_vals[g];
                pk_idx = g;
            }
        }
        const double v_peak_exact = v_lo + pk_idx * v_step;

        double fwhm_exact = 2.0 * p[2];
        double v_left = p[1] - p[2];
        double v_right = p[1] + p[2];
        const double half_h = 0.5 * pk_val;
        if (pk_val > 0.05) {
            int l_g = 0;
            while (l_g + 1 < kGridN && bf_vals[l_g] < half_h) ++l_g;
            int r_g = kGridN - 1;
            while (r_g > 0 && bf_vals[r_g] < half_h) --r_g;

            auto grid_v = [&](int idx) { return v_lo + idx * v_step; };
            if (l_g > 0 && l_g < kGridN) {
                const double y0 = bf_vals[l_g - 1], y1 = bf_vals[l_g];
                const double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_left = grid_v(l_g - 1) + t * v_step;
            } else {
                v_left = grid_v(l_g);
            }
            if (r_g >= 0 && r_g + 1 < kGridN) {
                const double y0 = bf_vals[r_g], y1 = bf_vals[r_g + 1];
                const double t = (std::abs(y1 - y0) > 1e-12) ? std::clamp((half_h - y0) / (y1 - y0), 0.0, 1.0) : 0.0;
                v_right = grid_v(r_g) + t * v_step;
            } else {
                v_right = grid_v(r_g);
            }
            if (v_right > v_left) fwhm_exact = v_right - v_left;
        }

        std::vector<double> model_sub(m);
        for (size_t i = 0; i < m; ++i) model_sub[i] = eval_busy(p, v_sub[i]);
        const FitGoodness good = evaluate_fit_goodness(t_sub, model_sub, 6, pk_val, f_corr_hint);

        GaussianFit res = base_fit;
        res.method = LineAnalysisMethod::BusyFunction;
        res.delta_bic = good.delta_bic;
        res.reduced_chi2 = good.reduced_chi2;
        res.rmse_k = std::sqrt(cur_chi2 / static_cast<double>(std::max<size_t>(1, m)));
        res.success = good.passes_bic && (pk_val > 0.05);

        if (res.success) {
            res.amplitude_k = pk_val;
            res.v_center_kms = p[1];
            res.v_peak_kms = v_peak_exact;
            res.fwhm_kms = fwhm_exact;
            res.sigma_kms = fwhm_exact / GAUSSIAN_SIGMA_TO_FWHM;
            res.fwhm_left_kms = v_left;
            res.fwhm_right_kms = v_right;
            res.fwhm_half_temp_k = half_h;
            res.busy_half_width_kms = p[2];
            res.busy_flank_b1 = p[3];
            res.busy_flank_b2 = p[4];
            res.busy_trough_c = p[5] / (p[2] * p[2]); // Convert dimensionless c_tilde back to physical [(km/s)^-2]
            res.fitted_curve.resize(n_all);
            for (size_t i = 0; i < n_all; ++i) {
                res.fitted_curve[i] = eval_busy(p, vel_kms[i]);
            }
        } else {
            res.amplitude_k = 0.0;
            res.sigma_kms = 0.0;
            res.fwhm_kms = 0.0;
            res.fwhm_left_kms = 0.0;
            res.fwhm_right_kms = 0.0;
            res.fwhm_half_temp_k = 0.0;
            res.fitted_curve.assign(n_all, 0.0);
        }
        return res;
    }

    return base_fit;
}

} // namespace

std::vector<double> Astrophysics::compute_doppler_velocity(const std::vector<double>& freqs_hz) {
    size_t n = freqs_hz.size();
    std::vector<double> v(n);
    constexpr double inv_rest_c = SPEED_OF_LIGHT_KMS / HYDROGEN_REST_FREQ_HZ;
    for (size_t i = 0; i < n; ++i) {
        v[i] = (HYDROGEN_REST_FREQ_HZ - freqs_hz[i]) * inv_rest_c;
    }
    return v;
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
    return fit_gaussian_impl(vel_kms, temp_k, mask, 0.0);
}

GaussianFit Astrophysics::fit_gaussian(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window
) {
    return fit_gaussian_impl(vel_kms, temp_k, hi_window, 0.0);
}

GaussianFit Astrophysics::fit_line_profile(
    const std::vector<double>& vel_kms,
    const std::vector<double>& temp_k,
    const std::vector<uint8_t>& hi_window,
    LineAnalysisMethod method
) {
    return fit_line_profile_impl(vel_kms, temp_k, hi_window, method, 0.0);
}

AstrophysicalMetrics Astrophysics::calculate_metrics(
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& vel_kms,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const AstroConfig& config,
    const std::vector<uint8_t>* effective_passband_mask
) {
    AstrophysicalMetrics m;
    const int nfft = static_cast<int>(freqs_mhz.size());
    if (nfft == 0 ||
        vel_kms.size() != static_cast<size_t>(nfft) ||
        calibrated_k.size() != static_cast<size_t>(nfft) ||
        calibrated_db.size() != static_cast<size_t>(nfft)) {
        return m;
    }

    const double hi_half_mhz = config.hi_protection_half_width_khz / 1e3;
    std::vector<uint8_t> hi_window(nfft, 0);
    for (int i = 0; i < nfft; ++i) {
        if (freqs_mhz[i] >= (HYDROGEN_REST_FREQ_MHZ - hi_half_mhz) &&
            freqs_mhz[i] <= (HYDROGEN_REST_FREQ_MHZ + hi_half_mhz)) {
            hi_window[i] = 1;
        }
    }

    // Effective passband mask (excludes steep IF roll-off skirts)
    std::vector<uint8_t> passband_mask(nfft, 0);
    if (effective_passband_mask != nullptr && effective_passband_mask->size() == static_cast<size_t>(nfft)) {
        passband_mask = *effective_passband_mask;
    } else {
        const double safe_edge_frac = std::isfinite(config.passband_edge_fraction)
            ? std::clamp(config.passband_edge_fraction, 0.0, 0.45)
            : 0.08;
        const int edge_bins = static_cast<int>(static_cast<double>(nfft) * safe_edge_frac);
        for (int i = edge_bins; i < nfft - edge_bins; ++i) {
            passband_mask[i] = 1;
        }
    }

    // Clean off-line noise region inside effective passband & [-local_baseline_outer_kms, +local_baseline_outer_kms]
    const double outer_kms = std::max(config.line_free_velocity_kms + 10.0, config.local_baseline_outer_kms);
    std::vector<uint8_t> off_line_mask(nfft, 0);
    int off_line_count = 0;
    for (int i = 0; i < nfft; ++i) {
        if (passband_mask[i] && !hi_window[i] &&
            std::isfinite(calibrated_k[i]) && std::isfinite(calibrated_db[i])) {
            const double av = std::abs(vel_kms[i]);
            if (av >= config.line_free_velocity_kms && av <= outer_kms) {
                off_line_mask[i] = 1;
                ++off_line_count;
            }
        }
    }

    if (off_line_count < 16) {
        // Fallback: entire effective passband excluding H I window (still excludes IF skirts!)
        std::fill(off_line_mask.begin(), off_line_mask.end(), 0);
        off_line_count = 0;
        for (int i = 0; i < nfft; ++i) {
            if (passband_mask[i] && !hi_window[i] &&
                std::isfinite(calibrated_k[i]) && std::isfinite(calibrated_db[i])) {
                off_line_mask[i] = 1;
                ++off_line_count;
            }
        }
    }
    if (off_line_count < 4) {
        for (int i = 0; i < nfft; ++i) {
            if (!hi_window[i] && std::isfinite(calibrated_k[i]) && std::isfinite(calibrated_db[i])) {
                off_line_mask[i] = 1;
                ++off_line_count;
            }
        }
    }

    std::vector<double> free_k, free_db;
    free_k.reserve(static_cast<size_t>(std::max(0, off_line_count)));
    free_db.reserve(static_cast<size_t>(std::max(0, off_line_count)));
    for (int i = 0; i < nfft; ++i) {
        if (off_line_mask[i]) {
            free_k.push_back(calibrated_k[i]);
            free_db.push_back(calibrated_db[i]);
        }
    }

    // 1. Exact window bin-correlation factor & thermal vs off-line noise decomposition
    m.window_corr_factor = window_corr_factor(config.window);
    m.thermal_noise_std_k = estimate_lag6_thermal_rms(calibrated_k, off_line_mask);
    m.off_line_rms_k = calc_robust_rms(free_k);
    m.noise_std_k = m.off_line_rms_k;
    m.noise_std_db = calc_robust_rms(free_db);
    m.baseline_sys_error_k = std::sqrt(
        std::max(0.0, m.off_line_rms_k * m.off_line_rms_k - m.thermal_noise_std_k * m.thermal_noise_std_k));
    m.baseline_quality_ratio = (m.thermal_noise_std_k > 1e-9)
        ? (m.off_line_rms_k / m.thermal_noise_std_k)
        : 1.0;

    // 2. Raw single-bin peak inside H I window
    int peak_idx = nfft / 2;
    double max_k = -std::numeric_limits<double>::infinity();
    bool found_peak = false;
    std::vector<double> v_hi, t_hi;
    v_hi.reserve(nfft);
    t_hi.reserve(nfft);
    for (int i = 0; i < nfft; ++i) {
        if (hi_window[i] && std::isfinite(calibrated_k[i]) && std::isfinite(vel_kms[i])) {
            v_hi.push_back(vel_kms[i]);
            t_hi.push_back(calibrated_k[i]);
            if (calibrated_k[i] > max_k) {
                max_k = calibrated_k[i];
                peak_idx = i;
                found_peak = true;
            }
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

    m.snr_single_bin = m.peak_temp_k / std::max(1e-6, m.noise_std_k);

    const double raw_dv = (nfft > 1) ? std::abs(vel_kms[1] - vel_kms[0]) : 1.0;
    const double dv = (std::isfinite(raw_dv) && raw_dv > 0.0) ? std::max(1e-9, raw_dv) : 1.0;
    double hi_span_kms = 0.0;
    if (v_hi.size() >= 2) {
        auto mm_hi = std::minmax_element(v_hi.begin(), v_hi.end());
        const double raw_span = *mm_hi.second - *mm_hi.first;
        hi_span_kms = std::isfinite(raw_span) ? std::max(dv, raw_span) : dv;
    }

    // 3. Multi-Scale Matched-Filter SNR across physical H I widths {2, 4, 8, 15, 25} km/s
    double best_snr_mf = 0.0;
    double best_amp_k = 0.0;
    double best_vel_kms = m.peak_vel_kms;
    bool has_mf_eval = false;

    // Normalize single-bin thermal noise to reference velocity resolution dv_ref = 0.125 km/s
    // (N_fft = 4096 @ 2.4 MSPS) so gap-interpolation uncertainty and amplitude floor are resolution-invariant.
    const double dv_norm_factor = std::sqrt(std::min(1.0, dv / 0.125));
    const double sigma_th_norm_k = m.thermal_noise_std_k * dv_norm_factor;
    const double eff_noise_norm_k = std::max(sigma_th_norm_k, std::hypot(sigma_th_norm_k, m.baseline_sys_error_k));

    if (v_hi.size() >= 3) {
        const size_t m_hi = v_hi.size();
        const double sigma_th2_fcorr = m.thermal_noise_std_k * m.thermal_noise_std_k * m.window_corr_factor;
        const double sigma_base2 = m.baseline_sys_error_k * m.baseline_sys_error_k;

        static constexpr double kTemplateWidths[] = {2.0, 4.0, 8.0, 15.0, 25.0};
        std::vector<double> valid_scales;
        valid_scales.reserve(5);
        for (double s : kTemplateWidths) {
            if (s >= dv && s <= 0.45 * hi_span_kms) {
                valid_scales.push_back(s);
            }
        }
        if (valid_scales.empty()) {
            valid_scales.push_back(std::max(dv, 0.2 * hi_span_kms));
        }

        for (double s_kms : valid_scales) {
            const double sigma_bins = s_kms / dv;
            const double raw_rad = std::ceil(4.0 * sigma_bins);
            const double max_rad_d = static_cast<double>(m_hi - 1);
            const int radius = static_cast<int>(std::isfinite(raw_rad)
                ? std::clamp(raw_rad, 1.0, std::max(1.0, max_rad_d))
                : max_rad_d);
            std::vector<double> g_tab(static_cast<size_t>(radius + 1));
            std::vector<double> g2_tab(static_cast<size_t>(radius + 1));
            const double inv_2s2 = -0.5 / (sigma_bins * sigma_bins);
            for (int d = 0; d <= radius; ++d) {
                const double g = std::exp(inv_2s2 * static_cast<double>(d * d));
                g_tab[static_cast<size_t>(d)] = g;
                g2_tab[static_cast<size_t>(d)] = g * g;
            }

            const double template_fwhm_kms = GAUSSIAN_SIGMA_TO_FWHM * s_kms;
            const double gap_coupling = std::clamp(template_fwhm_kms / std::max(10.0, 0.5 * hi_span_kms), 0.15, 1.5);
            const double sigma_gap = 0.42 * eff_noise_norm_k * gap_coupling;
            const double sigma_coherent2 = sigma_base2 + sigma_gap * sigma_gap;

            auto eval_center = [&](int c, double& out_amp, double& out_snr) {
                const int j_lo = std::max(0, c - radius);
                const int j_hi = std::min(static_cast<int>(m_hi) - 1, c + radius);
                double sum_tg = 0.0;
                double sum_g2 = 0.0;
                for (int j = j_lo; j <= j_hi; ++j) {
                    const int d = std::abs(j - c);
                    sum_tg += t_hi[static_cast<size_t>(j)] * g_tab[static_cast<size_t>(d)];
                    sum_g2 += g2_tab[static_cast<size_t>(d)];
                }
                if (sum_g2 <= 1e-12) {
                    out_amp = 0.0;
                    out_snr = 0.0;
                    return;
                }
                out_amp = sum_tg / sum_g2;
                const double sigma_a_tot = std::sqrt(sigma_th2_fcorr / sum_g2 + sigma_coherent2);
                out_snr = out_amp / std::max(1e-6, sigma_a_tot);
            };

            const int step = (m_hi <= 400 || sigma_bins < 8.0) ? 1 : std::max(1, static_cast<int>(0.25 * sigma_bins));
            if (step == 1) {
                for (int c = 0; c < static_cast<int>(m_hi); ++c) {
                    double amp = 0.0, snr = 0.0;
                    eval_center(c, amp, snr);
                    if (!has_mf_eval || snr > best_snr_mf) {
                        has_mf_eval = true;
                        best_snr_mf = snr;
                        best_amp_k = amp;
                        best_vel_kms = v_hi[static_cast<size_t>(c)];
                    }
                }
            } else {
                // Coarse scan + exact 1-bin refinement around top 3 coarse peaks
                struct CoarseCand { double snr; int idx; };
                std::vector<CoarseCand> coarse;
                coarse.reserve(m_hi / static_cast<size_t>(step) + 2);
                for (int c = 0; c < static_cast<int>(m_hi); c += step) {
                    double amp = 0.0, snr = 0.0;
                    eval_center(c, amp, snr);
                    coarse.push_back({snr, c});
                }
                if (coarse.back().idx != static_cast<int>(m_hi) - 1) {
                    double amp = 0.0, snr = 0.0;
                    eval_center(static_cast<int>(m_hi) - 1, amp, snr);
                    coarse.push_back({snr, static_cast<int>(m_hi) - 1});
                }
                const size_t k_top = std::min<size_t>(3, coarse.size());
                std::partial_sort(coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(k_top), coarse.end(),
                                  [](const CoarseCand& a, const CoarseCand& b) { return a.snr > b.snr; });
                for (size_t k = 0; k < k_top; ++k) {
                    const int c_lo = std::max(0, coarse[k].idx - step);
                    const int c_hi = std::min(static_cast<int>(m_hi) - 1, coarse[k].idx + step);
                    for (int c = c_lo; c <= c_hi; ++c) {
                        double amp = 0.0, snr = 0.0;
                        eval_center(c, amp, snr);
                        if (!has_mf_eval || snr > best_snr_mf) {
                            has_mf_eval = true;
                            best_snr_mf = snr;
                            best_amp_k = amp;
                            best_vel_kms = v_hi[static_cast<size_t>(c)];
                        }
                    }
                }
            }
        }
    }

    m.snr_matched = has_mf_eval ? std::max(0.0, best_snr_mf) : 0.0;
    m.peak_temp_smoothed_k = has_mf_eval ? best_amp_k : m.peak_temp_k;
    m.peak_vel_smoothed_kms = has_mf_eval ? best_vel_kms : m.peak_vel_kms;
    m.snr = m.snr_matched;

    // 4. Unbiased integrated line flux: W = sum(T_A) * dv [K * km/s]
    double integral_w = 0.0;
    int n_hi_bins = 0;
    for (int i = 0; i < nfft; ++i) {
        if (hi_window[i] && std::isfinite(calibrated_k[i])) {
            integral_w += calibrated_k[i] * dv;
            ++n_hi_bins;
        }
    }
    m.integrated_flux_k_kms = integral_w;
    m.column_density_cm2 = HI_COLUMN_DENSITY_FACTOR * std::max(0.0, integral_w);

    // 5. Line Profile Analysis (passing exact window F_corr)
    m.gauss_fit = fit_line_profile_impl(vel_kms, calibrated_k, hi_window, config.line_method, m.window_corr_factor);
    m.fwhm_kms = m.gauss_fit.fwhm_kms;
    m.sigma_kms = m.gauss_fit.sigma_kms;

    // 6. Multi-gate confirmed H I detection decision
    const double eff_noise_k = std::max(m.thermal_noise_std_k, m.off_line_rms_k);
    const double model_amp_k = std::max(m.gauss_fit.amplitude_k, m.peak_temp_smoothed_k);
    const bool amp_above_floor = (eff_noise_k <= 1e-6) ||
                                 (model_amp_k >= 2.85 * eff_noise_norm_k &&
                                  (m.snr_single_bin >= 3.8 || (m.peak_temp_k >= 3.8 * eff_noise_norm_k))) ||
                                 (m.snr_matched >= std::max(6.0, config.detection_snr_threshold + 1.5) &&
                                  m.gauss_fit.delta_bic > 20.0 &&
                                  model_amp_k >= 1.8 * eff_noise_norm_k);
    m.is_detected = (m.snr_matched >= config.detection_snr_threshold) &&
                    amp_above_floor &&
                    m.gauss_fit.success &&
                    (m.gauss_fit.delta_bic > 8.0) &&
                    (m.fwhm_kms >= std::max(1.0, 1.5 * dv)) &&
                    (hi_span_kms > 0.0 && m.fwhm_kms <= 0.95 * hi_span_kms);

    // 7. Integrated flux uncertainty, column density uncertainty, and integrated support SNR
    const double w_hi_kms = static_cast<double>(std::max(1, n_hi_bins)) * dv;
    const double err_hi_th = m.thermal_noise_std_k * dv * std::sqrt(static_cast<double>(std::max(1, n_hi_bins)) * m.window_corr_factor);
    const double err_hi_base = m.baseline_sys_error_k * w_hi_kms;
    m.integrated_flux_err_k_kms = std::hypot(err_hi_th, err_hi_base);
    m.column_density_err_cm2 = HI_COLUMN_DENSITY_FACTOR * m.integrated_flux_err_k_kms;

    const bool has_model_curve = m.gauss_fit.success && (m.gauss_fit.fitted_curve.size() == static_cast<size_t>(nfft));
    double model_peak_hi = 0.0;
    if (has_model_curve) {
        for (int i = 0; i < nfft; ++i) {
            if (hi_window[i] && m.gauss_fit.fitted_curve[i] > model_peak_hi) {
                model_peak_hi = m.gauss_fit.fitted_curve[i];
            }
        }
    }

    if (has_model_curve && model_peak_hi > 0.05) {
        const double supp_thresh = 0.05 * model_peak_hi;
        double flux_supp = 0.0;
        int n_supp = 0;
        for (int i = 0; i < nfft; ++i) {
            if (hi_window[i] && std::isfinite(calibrated_k[i]) && m.gauss_fit.fitted_curve[i] >= supp_thresh) {
                flux_supp += calibrated_k[i] * dv;
                ++n_supp;
            }
        }
        if (n_supp < 3) {
            flux_supp = integral_w;
            n_supp = std::max(1, n_hi_bins);
        }
        const double err_supp_th = m.thermal_noise_std_k * dv * std::sqrt(static_cast<double>(n_supp) * m.window_corr_factor);
        const double err_supp_base = m.baseline_sys_error_k * (static_cast<double>(n_supp) * dv);
        const double err_supp = std::hypot(err_supp_th, err_supp_base);
        m.snr_integrated = (err_supp > 1e-9)
            ? std::max(0.0, flux_supp / err_supp)
            : ((flux_supp > 0.05) ? m.snr_matched : 0.0);
    } else {
        m.snr_integrated = (m.integrated_flux_err_k_kms > 1e-9)
            ? std::max(0.0, integral_w / m.integrated_flux_err_k_kms)
            : 0.0;
    }

    // 8. Derived 21-cm physical & kinematic metrics computed from fitted/smoothed model T_model(v)
    if (!m.is_detected && !m.gauss_fit.success) {
        // Zero out profile widths and thermal upper bound on non-detection (no bogus 25 km/s fallback)
        m.fwhm_kms = 0.0;
        m.sigma_kms = 0.0;
        m.w50_kms = 0.0;
        m.w20_kms = 0.0;
        m.doppler_temp_max_k = 0.0;
        m.peak_optical_depth = 0.0;
        m.equivalent_width_kms = 0.0;
        m.column_density_thick_cm2 = m.column_density_cm2;
    } else {
        // Upper-bound thermal kinetic temperature: T_k,max = 21.855312 * FWHM^2
        m.doppler_temp_max_k = HI_THERMAL_TEMP_FACTOR_K * m.fwhm_kms * m.fwhm_kms;

        const double t_spin = std::max(10.0, config.spin_temp_k);
        if (has_model_curve && model_peak_hi > 0.05) {
            const double ratio_pk = std::min(0.999, model_peak_hi / t_spin);
            m.peak_optical_depth = -std::log(1.0 - ratio_pk);

            double model_integral_kms = 0.0;
            double tau_integral_kms = 0.0;
            for (int i = 0; i < nfft; ++i) {
                if (hi_window[i]) {
                    const double tb = m.gauss_fit.fitted_curve[i];
                    model_integral_kms += tb * dv;
                    if (tb > 0.0) {
                        const double ratio = std::min(0.999, tb / t_spin);
                        tau_integral_kms += (-t_spin * std::log(1.0 - ratio)) * dv;
                    } else {
                        tau_integral_kms += tb * dv;
                    }
                }
            }
            m.equivalent_width_kms = std::max(0.0, model_integral_kms) / model_peak_hi;
            m.column_density_thick_cm2 = std::max(
                m.column_density_cm2,
                HI_COLUMN_DENSITY_FACTOR * std::max(0.0, tau_integral_kms));
        } else {
            const double smooth_pk = std::max(0.0, m.peak_temp_smoothed_k);
            if (smooth_pk > 0.05) {
                const double ratio_pk = std::min(0.999, smooth_pk / t_spin);
                m.peak_optical_depth = -std::log(1.0 - ratio_pk);
                m.equivalent_width_kms = std::max(0.0, integral_w) / smooth_pk;
            }
            m.column_density_thick_cm2 = m.column_density_cm2;
        }

        // Full velocity widths W50 (50% peak) and W20 (20% peak) with widened wing search up to +/- local_baseline_outer_kms
        auto measure_profile_width = [&](double frac, double fallback_w) {
            if (!has_model_curve || model_peak_hi <= 0.05) {
                return fallback_w;
            }
            const double thresh = frac * model_peak_hi;

            int first_hi = -1, last_hi = -1;
            int search_lo = -1, search_hi = -1;
            for (int i = 0; i < nfft; ++i) {
                if (hi_window[i]) {
                    if (first_hi < 0) first_hi = i;
                    last_hi = i;
                }
                if (hi_window[i] || std::abs(vel_kms[i]) <= outer_kms) {
                    if (search_lo < 0) search_lo = i;
                    search_hi = i;
                }
            }
            if (first_hi < 0 || last_hi <= first_hi) return fallback_w;
            if (search_lo < 0 || search_hi <= search_lo) {
                search_lo = first_hi;
                search_hi = last_hi;
            }

            int li = search_lo;
            int ri = search_hi;
            if (config.line_method == LineAnalysisMethod::SpectralMoments) {
                // Non-parametric smoothed trace: find outermost flank inside hi_window,
                // then extend continuously outward up to +/- outer_kms if wing exceeds hi_window
                li = first_hi;
                while (li < last_hi && m.gauss_fit.fitted_curve[li] < thresh) ++li;
                while (li > search_lo && m.gauss_fit.fitted_curve[li - 1] >= thresh) --li;

                ri = last_hi;
                while (ri > first_hi && m.gauss_fit.fitted_curve[ri] < thresh) --ri;
                while (ri < search_hi && m.gauss_fit.fitted_curve[ri + 1] >= thresh) ++ri;
            } else {
                // Parametric model decays smoothly to 0 outside line: search inward across full +/- outer_kms span
                while (li < search_hi && m.gauss_fit.fitted_curve[li] < thresh) ++li;
                while (ri > search_lo && m.gauss_fit.fitted_curve[ri] < thresh) --ri;
            }
            if (ri < li) return fallback_w;

            auto interp_cross = [&](int i0, int i1) {
                const double y0 = m.gauss_fit.fitted_curve[i0];
                const double y1 = m.gauss_fit.fitted_curve[i1];
                if (std::abs(y1 - y0) < 1e-12) return vel_kms[i0];
                const double t = std::clamp((thresh - y0) / (y1 - y0), 0.0, 1.0);
                return vel_kms[i0] + t * (vel_kms[i1] - vel_kms[i0]);
            };
            const double v_a = (li > search_lo) ? interp_cross(li - 1, li) : vel_kms[li];
            const double v_b = (ri < search_hi) ? interp_cross(ri, ri + 1) : vel_kms[ri];
            const double w = std::abs(v_b - v_a);
            return (w > 0.5) ? w : fallback_w;
        };

        // For a pure Gaussian, W20 / W50 = sqrt(ln(5) / ln(2)) = 1.5238824
        m.w50_kms = measure_profile_width(0.50, m.fwhm_kms);
        m.w20_kms = measure_profile_width(0.20, m.fwhm_kms * 1.5238824);
    }

    // 9. SDR LO Offset and DC notch analysis
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
