#include "baseline_fitter.h"
#include "simd_dsp.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace astro {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kSqrt2Minus1 = 0.41421356237309504880;

double median_of_inplace(std::vector<double>& v) {
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

bool solve_3x3(const double A_in[3][3], const double b_in[3], double x_out[3]) {
    double M[3][4];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) M[r][c] = A_in[r][c];
        M[r][3] = b_in[r];
    }

    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double max_abs = std::abs(M[col][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double a = std::abs(M[r][col]);
            if (a > max_abs) {
                max_abs = a;
                piv = r;
            }
        }
        if (max_abs < 1e-14) return false;
        if (piv != col) {
            for (int c = col; c < 4; ++c) std::swap(M[col][c], M[piv][c]);
        }
        const double inv_piv = 1.0 / M[col][col];
        for (int r = col + 1; r < 3; ++r) {
            const double factor = M[r][col] * inv_piv;
            for (int c = col + 1; c < 4; ++c) {
                M[r][c] -= factor * M[col][c];
            }
            M[r][col] = 0.0;
        }
    }

    for (int r = 2; r >= 0; --r) {
        double sum = M[r][3];
        for (int c = r + 1; c < 3; ++c) sum -= M[r][c] * x_out[c];
        if (std::abs(M[r][r]) < 1e-14) return false;
        x_out[r] = sum / M[r][r];
    }
    return true;
}

// Smooth, C1-continuous, bounded tangent extrapolation away from a passband edge.
// sign_dir = -1.0 when extrapolating to the left (x < x_edge, dist_pos = x_edge - x > 0)
// sign_dir = +1.0 when extrapolating to the right (x > x_edge, dist_pos = x - x_edge > 0)
inline double smooth_tangent_extrap(
    double y_edge,
    double slope_per_unit,
    double dist_pos,
    double sign_dir,
    double scale_len,
    double max_excursion
) {
    if (dist_pos <= 0.0 || std::abs(slope_per_unit) < 1e-15) return y_edge;
    double eff_scale = std::max(1e-6, scale_len);
    const double abs_slope = std::abs(slope_per_unit);
    if (abs_slope * eff_scale > max_excursion) {
        eff_scale = max_excursion / abs_slope;
    }
    return y_edge + sign_dir * slope_per_unit * eff_scale * std::tanh(dist_pos / eff_scale);
}

// Compact L1/L2-resident Chunked Householder QR least-squares solver.
// Mathematically identical to full-matrix Householder QR (preserves condition number kappa(A),
// NOT kappa(A)^2) with O(p * 256) memory footprint regardless of N_fft.
struct ChunkedHouseholderQR {
    static constexpr int kBlockRows = 256;
    int p = 0;
    int cols = 0;
    int max_rows = 0;
    int cur_rows = 0;
    double rss = 0.0;
    std::vector<double> buf;

    void reset(int num_cols) {
        p = num_cols;
        cols = p + 1;
        max_rows = p + kBlockRows;
        cur_rows = p;
        rss = 0.0;
        buf.assign(static_cast<size_t>(max_rows) * static_cast<size_t>(cols), 0.0);
    }

    void flush() {
        if (cur_rows <= p || p <= 0) return;
        double* M = buf.data();
        for (int col = 0; col < p; ++col) {
            double tail_norm2 = 0.0;
            for (int r = col + 1; r < cur_rows; ++r) {
                const double val = M[r * cols + col];
                tail_norm2 += val * val;
            }
            if (tail_norm2 <= 1e-30) continue;

            const double x0 = M[col * cols + col];
            const double norm = std::sqrt(x0 * x0 + tail_norm2);
            const double alpha = (x0 >= 0.0) ? -norm : norm;
            const double v0 = x0 - alpha;
            const double v_norm2 = v0 * v0 + tail_norm2;
            if (v_norm2 <= 1e-30) continue;

            const double beta = 2.0 / v_norm2;
            M[col * cols + col] = alpha;

            for (int c = col + 1; c < cols; ++c) {
                double dot = v0 * M[col * cols + c];
                for (int r = col + 1; r < cur_rows; ++r) {
                    dot += M[r * cols + col] * M[r * cols + c];
                }
                const double scale = beta * dot;
                M[col * cols + c] -= scale * v0;
                for (int r = col + 1; r < cur_rows; ++r) {
                    M[r * cols + c] -= scale * M[r * cols + col];
                }
            }
        }

        // Accumulate exact residual sum of squares from rows [p .. cur_rows - 1]
        for (int r = p; r < cur_rows; ++r) {
            const double rv = M[r * cols + p];
            rss += rv * rv;
        }

        // Restore clean zero strict lower triangle of upper-triangular factor R
        for (int r = 1; r < p; ++r) {
            for (int c = 0; c < r; ++c) {
                M[r * cols + c] = 0.0;
            }
        }
        cur_rows = p;
    }

    inline void add_row(const double* phi, double y, double sqrt_w) {
        if (sqrt_w <= 0.0) return;
        double* dst = buf.data() + static_cast<size_t>(cur_rows) * static_cast<size_t>(cols);
        for (int c = 0; c < p; ++c) {
            dst[c] = sqrt_w * phi[c];
        }
        dst[p] = sqrt_w * y;
        ++cur_rows;
        if (cur_rows == max_rows) {
            flush();
        }
    }

    inline void add_reg_diag(int col_idx, double reg_weight) {
        if (col_idx < 0 || col_idx >= p || reg_weight <= 0.0) return;
        double* dst = buf.data() + static_cast<size_t>(cur_rows) * static_cast<size_t>(cols);
        std::fill_n(dst, cols, 0.0);
        dst[col_idx] = reg_weight;
        ++cur_rows;
        if (cur_rows == max_rows) {
            flush();
        }
    }

    bool solve(double* out_coeffs, double* out_rss = nullptr) {
        flush();
        const double* M = buf.data();
        for (int r = p - 1; r >= 0; --r) {
            double sum = M[r * cols + p];
            for (int c = r + 1; c < p; ++c) {
                sum -= M[r * cols + c] * out_coeffs[c];
            }
            const double diag = M[r * cols + r];
            if (std::abs(diag) < 1e-13) {
                out_coeffs[r] = 0.0;
            } else {
                out_coeffs[r] = sum / diag;
            }
        }
        if (out_rss) *out_rss = rss;
        return true;
    }
};

inline void eval_chebyshev_basis(double u, int deg, double* T) {
    T[0] = 1.0;
    if (deg >= 1) T[1] = u;
    const double two_u = 2.0 * u;
    for (int k = 2; k <= deg; ++k) {
        T[k] = two_u * T[k - 1] - T[k - 2];
    }
}

inline double eval_chebyshev_clenshaw(double u, const double* c, int deg) {
    double b1 = 0.0;
    double b2 = 0.0;
    const double two_u = 2.0 * u;
    for (int k = deg; k >= 1; --k) {
        const double b0 = two_u * b1 - b2 + c[k];
        b2 = b1;
        b1 = b0;
    }
    return u * b1 - b2 + c[0];
}

// Exact derivative of sum_{k=0..deg} c_k T_k(u) at u = sign_pm1 (+1 or -1):
// T_k'(+1) = k^2, T_k'(-1) = (-1)^(k-1) * k^2
inline double eval_chebyshev_endpoint_deriv(const double* c, int deg, double sign_pm1) {
    double deriv = 0.0;
    double sgn = 1.0;
    for (int k = 1; k <= deg; ++k) {
        deriv += sgn * static_cast<double>(k * k) * c[k];
        sgn *= sign_pm1;
    }
    return deriv;
}

// Double-precision AVX2+FMA Stockham Auto-Sort Complex FFT (Radix-4 first pass + Radix-2 upper passes).
// Twiddles stored with +sin for inverse; forward uses -sin. Unscaled on inverse.
struct DoubleFftPlan {
    int n = 0;
    std::vector<double> st_wr;
    std::vector<double> st_wi;
    std::vector<int> st_s;
    std::vector<double> re_a, im_a, re_b, im_b;

    void prepare(int nfft) {
        if (n == nfft) return;
        n = nfft;
        re_a.resize(static_cast<size_t>(nfft));
        im_a.resize(static_cast<size_t>(nfft));
        re_b.resize(static_cast<size_t>(nfft));
        im_b.resize(static_cast<size_t>(nfft));
        st_wr.clear();
        st_wi.clear();
        st_s.clear();

        for (int s = 4; s < nfft; s <<= 1) {
            st_s.push_back(s);
            const double base_ang = kTwoPi / static_cast<double>(2 * s);
            for (int j = 0; j < s; ++j) {
                const double ang = base_ang * static_cast<double>(j);
                st_wr.push_back(std::cos(ang));
                st_wi.push_back(std::sin(ang));
            }
        }
    }

    void run(const double* __restrict in_re, const double* __restrict in_im,
             double* __restrict out_re, double* __restrict out_im, bool inverse) {
        const int M = n / 4;
        double* dst_r = st_s.empty() ? out_re : re_a.data();
        double* dst_i = st_s.empty() ? out_im : im_a.data();
        double* alt_r = re_b.data();
        double* alt_i = im_b.data();

#if defined(__AVX2__)
        if (M >= 4) {
            for (int k = 0; k < M; k += 4) {
                __m256d x0r = _mm256_loadu_pd(in_re + 0 * M + k), x0i = _mm256_loadu_pd(in_im + 0 * M + k);
                __m256d x1r = _mm256_loadu_pd(in_re + 1 * M + k), x1i = _mm256_loadu_pd(in_im + 1 * M + k);
                __m256d x2r = _mm256_loadu_pd(in_re + 2 * M + k), x2i = _mm256_loadu_pd(in_im + 2 * M + k);
                __m256d x3r = _mm256_loadu_pd(in_re + 3 * M + k), x3i = _mm256_loadu_pd(in_im + 3 * M + k);

                __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
                __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
                __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
                __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);

                __m256d y0r = _mm256_add_pd(t0r, t2r), y0i = _mm256_add_pd(t0i, t2i);
                __m256d y2r = _mm256_sub_pd(t0r, t2r), y2i = _mm256_sub_pd(t0i, t2i);
                __m256d y1r, y1i, y3r, y3i;
                if (inverse) {
                    y1r = _mm256_sub_pd(t1r, t3i); y1i = _mm256_add_pd(t1i, t3r);
                    y3r = _mm256_add_pd(t1r, t3i); y3i = _mm256_sub_pd(t1i, t3r);
                } else {
                    y1r = _mm256_add_pd(t1r, t3i); y1i = _mm256_sub_pd(t1i, t3r);
                    y3r = _mm256_sub_pd(t1r, t3i); y3i = _mm256_add_pd(t1i, t3r);
                }

                auto transpose4x4 = [](__m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3) {
                    __m256d u0 = _mm256_unpacklo_pd(r0, r1);
                    __m256d u1 = _mm256_unpackhi_pd(r0, r1);
                    __m256d u2 = _mm256_unpacklo_pd(r2, r3);
                    __m256d u3 = _mm256_unpackhi_pd(r2, r3);
                    r0 = _mm256_permute2f128_pd(u0, u2, 0x20);
                    r1 = _mm256_permute2f128_pd(u1, u3, 0x20);
                    r2 = _mm256_permute2f128_pd(u0, u2, 0x31);
                    r3 = _mm256_permute2f128_pd(u1, u3, 0x31);
                };
                transpose4x4(y0r, y1r, y2r, y3r);
                transpose4x4(y0i, y1i, y2i, y3i);

                _mm256_storeu_pd(dst_r + 4 * (k + 0), y0r); _mm256_storeu_pd(dst_i + 4 * (k + 0), y0i);
                _mm256_storeu_pd(dst_r + 4 * (k + 1), y1r); _mm256_storeu_pd(dst_i + 4 * (k + 1), y1i);
                _mm256_storeu_pd(dst_r + 4 * (k + 2), y2r); _mm256_storeu_pd(dst_i + 4 * (k + 2), y2i);
                _mm256_storeu_pd(dst_r + 4 * (k + 3), y3r); _mm256_storeu_pd(dst_i + 4 * (k + 3), y3i);
            }
        } else
#endif
        {
            for (int k = 0; k < M; ++k) {
                const double x0r = in_re[0 * M + k], x0i = in_im[0 * M + k];
                const double x1r = in_re[1 * M + k], x1i = in_im[1 * M + k];
                const double x2r = in_re[2 * M + k], x2i = in_im[2 * M + k];
                const double x3r = in_re[3 * M + k], x3i = in_im[3 * M + k];
                const double t0r = x0r + x2r, t0i = x0i + x2i;
                const double t1r = x0r - x2r, t1i = x0i - x2i;
                const double t2r = x1r + x3r, t2i = x1i + x3i;
                const double t3r = x1r - x3r, t3i = x1i - x3i;
                dst_r[4 * k + 0] = t0r + t2r; dst_i[4 * k + 0] = t0i + t2i;
                dst_r[4 * k + 2] = t0r - t2r; dst_i[4 * k + 2] = t0i - t2i;
                if (inverse) {
                    dst_r[4 * k + 1] = t1r - t3i; dst_i[4 * k + 1] = t1i + t3r;
                    dst_r[4 * k + 3] = t1r + t3i; dst_i[4 * k + 3] = t1i - t3r;
                } else {
                    dst_r[4 * k + 1] = t1r + t3i; dst_i[4 * k + 1] = t1i - t3r;
                    dst_r[4 * k + 3] = t1r - t3i; dst_i[4 * k + 3] = t1i + t3r;
                }
            }
        }

        double* cur_r = dst_r;
        double* cur_i = dst_i;
        double* nxt_r = alt_r;
        double* nxt_i = alt_i;
        const int half_n = n >> 1;
        size_t tw_off = 0;

        for (size_t st = 0; st < st_s.size(); ++st) {
            const int s = st_s[st];
            const int rem = n / (2 * s);
            const double* wr_ptr = st_wr.data() + tw_off;
            const double* wi_ptr = st_wi.data() + tw_off;
            tw_off += static_cast<size_t>(s);

            const bool is_last = (st + 1 == st_s.size());
            double* write_r = is_last ? out_re : nxt_r;
            double* write_i = is_last ? out_im : nxt_i;

            for (int k = 0; k < rem; ++k) {
                const int in_off = k * s;
                const int out_off = k * (2 * s);
                int j = 0;
#if defined(__AVX2__)
                for (; j + 4 <= s; j += 4) {
                    __m256d ar = _mm256_loadu_pd(cur_r + in_off + j);
                    __m256d ai = _mm256_loadu_pd(cur_i + in_off + j);
                    __m256d br0 = _mm256_loadu_pd(cur_r + in_off + half_n + j);
                    __m256d bi0 = _mm256_loadu_pd(cur_i + in_off + half_n + j);
                    __m256d wr = _mm256_loadu_pd(wr_ptr + j);
                    __m256d wi = _mm256_loadu_pd(wi_ptr + j);
                    __m256d br, bi;
                    if (inverse) {
#if SIMD_HAS_FMA
                        br = _mm256_fmsub_pd(br0, wr, _mm256_mul_pd(bi0, wi));
                        bi = _mm256_fmadd_pd(br0, wi, _mm256_mul_pd(bi0, wr));
#else
                        br = _mm256_sub_pd(_mm256_mul_pd(br0, wr), _mm256_mul_pd(bi0, wi));
                        bi = _mm256_add_pd(_mm256_mul_pd(br0, wi), _mm256_mul_pd(bi0, wr));
#endif
                    } else {
#if SIMD_HAS_FMA
                        br = _mm256_fmadd_pd(br0, wr, _mm256_mul_pd(bi0, wi));
                        bi = _mm256_fmsub_pd(bi0, wr, _mm256_mul_pd(br0, wi));
#else
                        br = _mm256_add_pd(_mm256_mul_pd(br0, wr), _mm256_mul_pd(bi0, wi));
                        bi = _mm256_sub_pd(_mm256_mul_pd(bi0, wr), _mm256_mul_pd(br0, wi));
#endif
                    }
                    _mm256_storeu_pd(write_r + out_off + j,     _mm256_add_pd(ar, br));
                    _mm256_storeu_pd(write_i + out_off + j,     _mm256_add_pd(ai, bi));
                    _mm256_storeu_pd(write_r + out_off + s + j, _mm256_sub_pd(ar, br));
                    _mm256_storeu_pd(write_i + out_off + s + j, _mm256_sub_pd(ai, bi));
                }
#endif
                for (; j < s; ++j) {
                    const double ar = cur_r[in_off + j], ai = cur_i[in_off + j];
                    const double br0 = cur_r[in_off + half_n + j], bi0 = cur_i[in_off + half_n + j];
                    const double wr = wr_ptr[j], wi = wi_ptr[j];
                    const double br = inverse ? (br0 * wr - bi0 * wi) : (br0 * wr + bi0 * wi);
                    const double bi = inverse ? (br0 * wi + bi0 * wr) : (bi0 * wr - br0 * wi);
                    write_r[out_off + j]     = ar + br;
                    write_i[out_off + j]     = ai + bi;
                    write_r[out_off + s + j] = ar - br;
                    write_i[out_off + s + j] = ai - bi;
                }
            }
            if (!is_last) {
                std::swap(cur_r, nxt_r);
                std::swap(cur_i, nxt_i);
            }
        }
    }
};

// Fill padded signal [0 .. n + 2*radius - 1] with smooth edge-slope continuation (or flat clamp)
void fill_padded_signal(
    const std::vector<double>& input,
    int radius,
    bool slope_pad,
    double* dst
) {
    const int n = static_cast<int>(input.size());
    std::memcpy(dst + radius, input.data(), static_cast<size_t>(n) * sizeof(double));

    if (!slope_pad || n < 4) {
        std::fill_n(dst, radius, input.front());
        std::fill_n(dst + radius + n, radius, input.back());
        return;
    }

    const int K = std::clamp(n / 64, 2, 16);
    const double mean_k = 0.5 * static_cast<double>(K - 1);
    const double var_k = static_cast<double>(K) * (static_cast<double>(K) * static_cast<double>(K) - 1.0) / 12.0;
    const double inv_var = (var_k > 1e-12) ? (1.0 / var_k) : 0.0;

    // Left edge linear fit over [0 .. K - 1]
    double sum_yl = 0.0, sum_xyl = 0.0;
    for (int k = 0; k < K; ++k) {
        const double y = input[static_cast<size_t>(k)];
        sum_yl += y;
        sum_xyl += (static_cast<double>(k) - mean_k) * y;
    }
    const double s_l = sum_xyl * inv_var;
    const double a_l = (sum_yl / static_cast<double>(K)) - s_l * mean_k;

    // Right edge linear fit over [n - K .. n - 1]
    double sum_yr = 0.0, sum_xyr = 0.0;
    for (int k = 0; k < K; ++k) {
        const double y = input[static_cast<size_t>(n - K + k)];
        sum_yr += y;
        sum_xyr += (static_cast<double>(k) - mean_k) * y;
    }
    const double s_r = sum_xyr * inv_var;
    const double a_r = (sum_yr / static_cast<double>(K)) + s_r * (static_cast<double>(K - 1) - mean_k);

    const double pad_scale = std::max(32.0, static_cast<double>(radius));
    for (int j = 1; j <= radius; ++j) {
        dst[radius - j] = smooth_tangent_extrap(a_l, s_l, static_cast<double>(j), -1.0, pad_scale * 2.0, 20.0);
        dst[radius + n - 1 + j] = smooth_tangent_extrap(a_r, s_r, static_cast<double>(j), +1.0, pad_scale * 2.0, 20.0);
    }
}

// Symmetric 1D convolution with O(N log N) AVX2+FMA DoubleFftPlan for radius > 24
// and direct vectorized AVX2+FMA loop for radius <= 24.
std::vector<double> convolve_symmetric_kernel_1d(
    const std::vector<double>& input,
    const std::vector<double>& half_kernel,
    bool slope_pad
) {
    const int n = static_cast<int>(input.size());
    const int radius = static_cast<int>(half_kernel.size()) - 1;
    if (n < 2 || radius <= 0) return input;

    std::vector<double> output(static_cast<size_t>(n));
    const double* hk = half_kernel.data();

    if (radius > 24) {
        const int L = n + 2 * radius;
        int nfft = 64;
        while (nfft < L) nfft <<= 1;

        thread_local DoubleFftPlan plan;
        thread_local std::vector<double> cached_hk;
        thread_local std::vector<double> k_spec_re, k_spec_im;
        thread_local std::vector<double> sig_re, sig_im, s_re, s_im, out_re, out_im;
        thread_local int cached_nfft = 0;

        plan.prepare(nfft);
        sig_re.resize(static_cast<size_t>(nfft));
        sig_im.resize(static_cast<size_t>(nfft));
        s_re.resize(static_cast<size_t>(nfft));
        s_im.resize(static_cast<size_t>(nfft));
        out_re.resize(static_cast<size_t>(nfft));
        out_im.resize(static_cast<size_t>(nfft));

        const bool kernel_matches =
            (cached_nfft == nfft) &&
            (cached_hk.size() == half_kernel.size()) &&
            (std::memcmp(cached_hk.data(), half_kernel.data(), half_kernel.size() * sizeof(double)) == 0);

        if (!kernel_matches) {
            cached_nfft = nfft;
            cached_hk = half_kernel;
            k_spec_re.resize(static_cast<size_t>(nfft));
            k_spec_im.resize(static_cast<size_t>(nfft));

            const double inv_nfft = 1.0 / static_cast<double>(nfft);
            std::fill(sig_re.begin(), sig_re.end(), 0.0);
            std::fill(sig_im.begin(), sig_im.end(), 0.0);
            sig_re[static_cast<size_t>(radius)] = hk[0] * inv_nfft;
            for (int m = 1; m <= radius; ++m) {
                const double w = hk[m] * inv_nfft;
                sig_re[static_cast<size_t>(radius - m)] = w;
                sig_re[static_cast<size_t>(radius + m)] = w;
            }
            plan.run(sig_re.data(), sig_im.data(), k_spec_re.data(), k_spec_im.data(), false);
        }

        fill_padded_signal(input, radius, slope_pad, sig_re.data());
        std::fill_n(sig_re.data() + L, nfft - L, 0.0);
        std::fill_n(sig_im.data(), nfft, 0.0);

        plan.run(sig_re.data(), sig_im.data(), s_re.data(), s_im.data(), false);

        int idx = 0;
#if defined(__AVX2__)
        for (; idx + 4 <= nfft; idx += 4) {
            __m256d ar = _mm256_loadu_pd(s_re.data() + idx);
            __m256d ai = _mm256_loadu_pd(s_im.data() + idx);
            __m256d br = _mm256_loadu_pd(k_spec_re.data() + idx);
            __m256d bi = _mm256_loadu_pd(k_spec_im.data() + idx);
#if SIMD_HAS_FMA
            __m256d pr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
            __m256d pi = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
#else
            __m256d pr = _mm256_sub_pd(_mm256_mul_pd(ar, br), _mm256_mul_pd(ai, bi));
            __m256d pi = _mm256_add_pd(_mm256_mul_pd(ar, bi), _mm256_mul_pd(ai, br));
#endif
            _mm256_storeu_pd(sig_re.data() + idx, pr);
            _mm256_storeu_pd(sig_im.data() + idx, pi);
        }
#endif
        for (; idx < nfft; ++idx) {
            const double ar = s_re[static_cast<size_t>(idx)];
            const double ai = s_im[static_cast<size_t>(idx)];
            const double br = k_spec_re[static_cast<size_t>(idx)];
            const double bi = k_spec_im[static_cast<size_t>(idx)];
            sig_re[static_cast<size_t>(idx)] = ar * br - ai * bi;
            sig_im[static_cast<size_t>(idx)] = ar * bi + ai * br;
        }

        plan.run(sig_re.data(), sig_im.data(), out_re.data(), out_im.data(), true);
        std::memcpy(output.data(), out_re.data() + 2 * radius, static_cast<size_t>(n) * sizeof(double));
        return output;
    }

    // Direct symmetric AVX2+FMA convolution for small kernels (radius <= 24)
    thread_local std::vector<double> tl_padded;
    const size_t pad_len = static_cast<size_t>(n) + 2ULL * static_cast<size_t>(radius);
    if (tl_padded.size() < pad_len) tl_padded.resize(pad_len);
    fill_padded_signal(input, radius, slope_pad, tl_padded.data());

    const double* center_base = tl_padded.data() + radius;
    double* out = output.data();
    int i = 0;

#if defined(__AVX2__)
    const __m256d k0 = _mm256_set1_pd(hk[0]);
    for (; i + 8 <= n; i += 8) {
        const double* c = center_base + i;
        __m256d acc0 = _mm256_mul_pd(k0, _mm256_loadu_pd(c));
        __m256d acc1 = _mm256_mul_pd(k0, _mm256_loadu_pd(c + 4));
        for (int k = 1; k <= radius; ++k) {
            const __m256d kv = _mm256_set1_pd(hk[k]);
            const __m256d s0 = _mm256_add_pd(_mm256_loadu_pd(c - k), _mm256_loadu_pd(c + k));
            const __m256d s1 = _mm256_add_pd(_mm256_loadu_pd(c + 4 - k), _mm256_loadu_pd(c + 4 + k));
#if SIMD_HAS_FMA
            acc0 = _mm256_fmadd_pd(kv, s0, acc0);
            acc1 = _mm256_fmadd_pd(kv, s1, acc1);
#else
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(kv, s0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(kv, s1));
#endif
        }
        _mm256_storeu_pd(out + i, acc0);
        _mm256_storeu_pd(out + i + 4, acc1);
    }
    for (; i + 4 <= n; i += 4) {
        const double* c = center_base + i;
        __m256d acc = _mm256_mul_pd(k0, _mm256_loadu_pd(c));
        for (int k = 1; k <= radius; ++k) {
            const __m256d kv = _mm256_set1_pd(hk[k]);
            const __m256d s = _mm256_add_pd(_mm256_loadu_pd(c - k), _mm256_loadu_pd(c + k));
#if SIMD_HAS_FMA
            acc = _mm256_fmadd_pd(kv, s, acc);
#else
            acc = _mm256_add_pd(acc, _mm256_mul_pd(kv, s));
#endif
        }
        _mm256_storeu_pd(out + i, acc);
    }
#endif
    for (; i < n; ++i) {
        const double* c = center_base + i;
        double acc = hk[0] * c[0];
        for (int k = 1; k <= radius; ++k) {
            acc += hk[k] * (c[-k] + c[k]);
        }
        out[i] = acc;
    }
    return output;
}

std::vector<double> gaussian_smooth_1d(const std::vector<double>& input, double sigma) {
    if (sigma <= 0.0 || input.size() < 3) return input;
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> half_kernel(static_cast<size_t>(radius) + 1);
    const double inv_2sig2 = -0.5 / (sigma * sigma);
    double sum = 1.0;
    half_kernel[0] = 1.0;
    for (int k = 1; k <= radius; ++k) {
        const double kd = static_cast<double>(k);
        const double g = std::exp(inv_2sig2 * kd * kd);
        half_kernel[static_cast<size_t>(k)] = g;
        sum += 2.0 * g;
    }
    const double inv_sum = 1.0 / sum;
    for (int k = 0; k <= radius; ++k) {
        half_kernel[static_cast<size_t>(k)] *= inv_sum;
    }
    return convolve_symmetric_kernel_1d(input, half_kernel, true);
}

inline int compute_macro_bin_count(int valid_span) {
    if (valid_span <= 48) return valid_span;
    return std::clamp(valid_span / 16, 48, 192);
}

// Compute Whittaker penalty parameter lambda from physical half-power (-3 dB) cutoff period
double compute_whittaker_lambda(
    int m,
    double passband_span_khz,
    double cutoff_khz,
    int diff_order,
    double scale_mult
) {
    const double dm_khz = passband_span_khz / static_cast<double>(std::max(1, m - 1));
    const double safe_cutoff = std::clamp(
        cutoff_khz, 2.5 * dm_khz, std::max(3.0 * dm_khz, 0.95 * passband_span_khz)
    );
    const double half_omega = kPi * dm_khz / safe_cutoff;
    const double s2 = std::max(1e-4, 2.0 * std::sin(std::min(1.5, half_omega)));
    const double exponent = (diff_order == 2) ? 4.0 : 6.0;
    const double lam_phys = kSqrt2Minus1 / std::pow(s2, exponent);
    const double max_lam = (diff_order == 2) ? 1e8 : 1e9;
    return std::clamp(lam_phys * std::clamp(scale_mult, 1e-3, 1e4), 1.0, max_lam);
}

// Shared curvature-preserving Whittaker gap bridge (default 3rd-order D3^T D3 quintic Hermite bridge).
// Macro-bins strictly over [first_pass, last_pass], tracks exact centers of mass of clean bins,
// assigns zero weight inside hi_window, solves via heptadiagonal (D3) or pentadiagonal (D2) LDL^T,
// applies symmetric Huber IRLS (|p - 0.5| < 0.05) or asymmetric ALS weights, and upsamples smoothly
// with Catmull-Rom cubic interpolation inside [first_pass, last_pass] and bounded C1 tangent
// continuation on outer skirts.
std::vector<double> solve_whittaker_d3_gap_bridge(
    const std::vector<double>& p_db,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    double lam,
    double asym_p,
    bool use_huber,
    int diff_order = 3
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    int first_pass = -1;
    int last_pass = -1;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[static_cast<size_t>(i)]) {
            if (first_pass < 0) first_pass = i;
            last_pass = i;
        }
    }
    if (first_pass < 0 || last_pass - first_pass < 3) return p_db;

    const int valid_span = last_pass - first_pass + 1;
    const int m = compute_macro_bin_count(valid_span);
    if (m < 4) return p_db;

    thread_local std::vector<double> m_y, m_center, grid_center, base_w, w, z, rhs;
    thread_local std::vector<double> s_d, s_e1, s_e2, s_e3;
    thread_local std::vector<double> A_d, A_e1, A_e2, A_e3;
    thread_local std::vector<double> res_scratch;

    m_y.assign(static_cast<size_t>(m), 0.0);
    m_center.assign(static_cast<size_t>(m), 0.0);
    grid_center.assign(static_cast<size_t>(m), 0.0);
    base_w.assign(static_cast<size_t>(m), 0.0);
    w.assign(static_cast<size_t>(m), 0.0);
    z.assign(static_cast<size_t>(m), 0.0);
    rhs.assign(static_cast<size_t>(m), 0.0);

    int active_macro = 0;
    std::vector<int> clean_counts(static_cast<size_t>(m), 0);
    std::vector<int> bin_counts(static_cast<size_t>(m), 1);

    for (int b = 0; b < m; ++b) {
        const int i0 = first_pass + static_cast<int>((static_cast<int64_t>(b) * valid_span) / m);
        int i1 = first_pass + static_cast<int>((static_cast<int64_t>(b + 1) * valid_span) / m);
        if (i1 <= i0) i1 = std::min(last_pass + 1, i0 + 1);
        const int b_cnt = std::max(1, i1 - i0);
        bin_counts[static_cast<size_t>(b)] = b_cnt;
        grid_center[static_cast<size_t>(b)] = 0.5 * static_cast<double>(i0 + i1 - 1);

        double sum_y = 0.0;
        double sum_idx = 0.0;
        int cnt_clean = 0;
        for (int i = i0; i < i1; ++i) {
            if (passband_mask[static_cast<size_t>(i)] && !hi_window[static_cast<size_t>(i)]) {
                sum_y += p_db[static_cast<size_t>(i)];
                sum_idx += static_cast<double>(i);
                ++cnt_clean;
            }
        }
        clean_counts[static_cast<size_t>(b)] = cnt_clean;
        if (cnt_clean == b_cnt) {
            m_y[static_cast<size_t>(b)] = sum_y / static_cast<double>(cnt_clean);
            m_center[static_cast<size_t>(b)] = sum_idx / static_cast<double>(cnt_clean);
            base_w[static_cast<size_t>(b)] = 1.0;
            ++active_macro;
        } else if (cnt_clean > 0) {
            // Partially masked macro-bin straddling hi_window boundary: store true center of mass
            // for fragmented-mask fallback, but default to w = 0 so shifted slivers never bias
            // uniform-grid D3 gap-edge derivatives.
            m_y[static_cast<size_t>(b)] = sum_y / static_cast<double>(cnt_clean);
            m_center[static_cast<size_t>(b)] = grid_center[static_cast<size_t>(b)];
            base_w[static_cast<size_t>(b)] = 0.0;
        } else {
            m_y[static_cast<size_t>(b)] = 0.0;
            m_center[static_cast<size_t>(b)] = grid_center[static_cast<size_t>(b)];
            base_w[static_cast<size_t>(b)] = 0.0;
        }
    }

    if (active_macro < 8) {
        active_macro = 0;
        for (int b = 0; b < m; ++b) {
            if (clean_counts[static_cast<size_t>(b)] > 0) {
                const int i0 = first_pass + static_cast<int>((static_cast<int64_t>(b) * valid_span) / m);
                int i1 = first_pass + static_cast<int>((static_cast<int64_t>(b + 1) * valid_span) / m);
                if (i1 <= i0) i1 = std::min(last_pass + 1, i0 + 1);
                double sum_idx = 0.0;
                for (int i = i0; i < i1; ++i) {
                    if (passband_mask[static_cast<size_t>(i)] && !hi_window[static_cast<size_t>(i)]) {
                        sum_idx += static_cast<double>(i);
                    }
                }
                m_center[static_cast<size_t>(b)] = sum_idx / static_cast<double>(clean_counts[static_cast<size_t>(b)]);
                base_w[static_cast<size_t>(b)] =
                    static_cast<double>(clean_counts[static_cast<size_t>(b)]) /
                    static_cast<double>(bin_counts[static_cast<size_t>(b)]);
                ++active_macro;
            }
        }
    }
    if (active_macro < 4) return p_db;

    const double safe_lam = std::max(1.0, lam);
    const bool use_d2 = (diff_order == 2) || (m < 5);

    // Exact symmetric PSD assembly via outer products of difference stencil
    s_d.assign(static_cast<size_t>(m), 0.0);
    s_e1.assign(static_cast<size_t>(std::max(1, m - 1)), 0.0);
    s_e2.assign(static_cast<size_t>(std::max(1, m - 2)), 0.0);
    A_d.resize(static_cast<size_t>(m));
    A_e1.resize(static_cast<size_t>(std::max(1, m - 1)));
    A_e2.resize(static_cast<size_t>(std::max(1, m - 2)));

    if (use_d2) {
        static constexpr double c2[3] = {1.0, -2.0, 1.0};
        for (int k = 0; k <= m - 3; ++k) {
            for (int a = 0; a < 3; ++a) {
                s_d[static_cast<size_t>(k + a)] += c2[a] * c2[a];
                if (a + 1 < 3) s_e1[static_cast<size_t>(k + a)] += c2[a] * c2[a + 1];
                if (a + 2 < 3) s_e2[static_cast<size_t>(k + a)] += c2[a] * c2[a + 2];
            }
        }
        for (int i = 0; i < m - 1; ++i) A_e1[static_cast<size_t>(i)] = safe_lam * s_e1[static_cast<size_t>(i)];
        for (int i = 0; i < m - 2; ++i) A_e2[static_cast<size_t>(i)] = safe_lam * s_e2[static_cast<size_t>(i)];
    } else {
        s_e3.assign(static_cast<size_t>(std::max(1, m - 3)), 0.0);
        A_e3.resize(static_cast<size_t>(std::max(1, m - 3)));
        static constexpr double c3[4] = {1.0, -3.0, 3.0, -1.0};
        for (int k = 0; k <= m - 4; ++k) {
            for (int a = 0; a < 4; ++a) {
                s_d[static_cast<size_t>(k + a)] += c3[a] * c3[a];
                if (a + 1 < 4) s_e1[static_cast<size_t>(k + a)] += c3[a] * c3[a + 1];
                if (a + 2 < 4) s_e2[static_cast<size_t>(k + a)] += c3[a] * c3[a + 2];
                if (a + 3 < 4) s_e3[static_cast<size_t>(k + a)] += c3[a] * c3[a + 3];
            }
        }
        for (int i = 0; i < m - 1; ++i) A_e1[static_cast<size_t>(i)] = safe_lam * s_e1[static_cast<size_t>(i)];
        for (int i = 0; i < m - 2; ++i) A_e2[static_cast<size_t>(i)] = safe_lam * s_e2[static_cast<size_t>(i)];
        for (int i = 0; i < m - 3; ++i) A_e3[static_cast<size_t>(i)] = safe_lam * s_e3[static_cast<size_t>(i)];
    }

    const bool huber_mode = use_huber || (std::abs(asym_p - 0.5) < 0.005);
    const int iterations = huber_mode ? 4 : 8;
    const double p_asym = std::clamp(asym_p, 1e-4, 0.9999);
    w = base_w;

    for (int iter = 0; iter < iterations; ++iter) {
        for (int b = 0; b < m; ++b) {
            A_d[static_cast<size_t>(b)] = w[static_cast<size_t>(b)] + safe_lam * s_d[static_cast<size_t>(b)] + 1e-12;
            rhs[static_cast<size_t>(b)] = w[static_cast<size_t>(b)] * m_y[static_cast<size_t>(b)];
        }

        if (use_d2) {
            solve_pentadiagonal_symmetric(m, A_d.data(), A_e1.data(), A_e2.data(), rhs.data(), z.data());
        } else {
            solve_heptadiagonal_symmetric(m, A_d.data(), A_e1.data(), A_e2.data(), A_e3.data(), rhs.data(), z.data());
        }

        if (iter + 1 >= iterations) break;

        if (huber_mode) {
            res_scratch.clear();
            for (int b = 0; b < m; ++b) {
                if (base_w[static_cast<size_t>(b)] > 0.0) {
                    res_scratch.push_back(m_y[static_cast<size_t>(b)] - z[static_cast<size_t>(b)]);
                }
            }
            const double med_r = median_of_inplace(res_scratch);
            for (double& r : res_scratch) r = std::abs(r - med_r);
            const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(res_scratch));
            const double delta = 1.345 * mad_sigma;
            for (int b = 0; b < m; ++b) {
                if (base_w[static_cast<size_t>(b)] > 0.0) {
                    const double a = std::abs(m_y[static_cast<size_t>(b)] - z[static_cast<size_t>(b)]);
                    w[static_cast<size_t>(b)] = base_w[static_cast<size_t>(b)] * ((a <= delta) ? 1.0 : (delta / a));
                } else {
                    w[static_cast<size_t>(b)] = 0.0;
                }
            }
        } else {
            for (int b = 0; b < m; ++b) {
                if (base_w[static_cast<size_t>(b)] > 0.0) {
                    const double asym_w = (m_y[static_cast<size_t>(b)] > z[static_cast<size_t>(b)]) ? p_asym : (1.0 - p_asym);
                    w[static_cast<size_t>(b)] = base_w[static_cast<size_t>(b)] * 2.0 * asym_w;
                } else {
                    w[static_cast<size_t>(b)] = 0.0;
                }
            }
        }
    }

    // Tukey twicing pass: solve for the smooth residual on clean macro-bins so transfer function
    // becomes 2*H - H^2 = 1 - (1 - H)^2 (scaled by symmetry so lower-envelope p << 0.5 is preserved)
    const double twicing_scale = huber_mode ? 1.0 : std::clamp(1.0 - std::abs(1.0 - 2.0 * p_asym), 0.0, 1.0);
    if (twicing_scale > 1e-4) {
        for (int b = 0; b < m; ++b) {
            A_d[static_cast<size_t>(b)] = w[static_cast<size_t>(b)] + safe_lam * s_d[static_cast<size_t>(b)] + 1e-12;
            rhs[static_cast<size_t>(b)] = w[static_cast<size_t>(b)] * (m_y[static_cast<size_t>(b)] - z[static_cast<size_t>(b)]);
        }
        std::vector<double> dz(static_cast<size_t>(m), 0.0);
        if (use_d2) {
            solve_pentadiagonal_symmetric(m, A_d.data(), A_e1.data(), A_e2.data(), rhs.data(), dz.data());
        } else {
            solve_heptadiagonal_symmetric(m, A_d.data(), A_e1.data(), A_e2.data(), A_e3.data(), rhs.data(), dz.data());
        }
        for (int b = 0; b < m; ++b) {
            z[static_cast<size_t>(b)] += twicing_scale * dz[static_cast<size_t>(b)];
        }
    }

    // Upsample z[0..m-1] smoothly via Catmull-Rom cubic interpolation inside [first_pass, last_pass]
    // and smooth bounded C1 tangent continuation on outer skirts.
    const double* xk = m_center.data();
    const double s_left = (z[1] - z[0]) / std::max(1.0, xk[1] - xk[0]);
    const double s_right = (z[static_cast<size_t>(m - 1)] - z[static_cast<size_t>(m - 2)]) /
                           std::max(1.0, xk[m - 1] - xk[m - 2]);
    const double scale_left = std::max(16.0, 0.5 * std::max<double>(static_cast<double>(first_pass), static_cast<double>(n) / 16.0));
    const double scale_right = std::max(16.0, 0.5 * std::max<double>(static_cast<double>(n - 1 - last_pass), static_cast<double>(n) / 16.0));

    std::vector<double> out(static_cast<size_t>(n));
    int b = 0;
    for (int i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        if (x <= xk[0]) {
            out[static_cast<size_t>(i)] = smooth_tangent_extrap(z[0], s_left, xk[0] - x, -1.0, scale_left, 6.0);
        } else if (x >= xk[m - 1]) {
            out[static_cast<size_t>(i)] = smooth_tangent_extrap(z[static_cast<size_t>(m - 1)], s_right, x - xk[m - 1], +1.0, scale_right, 6.0);
        } else {
            while (b + 1 < m - 1 && x > xk[b + 1]) ++b;
            const double span = std::max(1e-9, xk[b + 1] - xk[b]);
            const double t = std::clamp((x - xk[b]) / span, 0.0, 1.0);
            const double p1 = z[static_cast<size_t>(b)];
            const double p2 = z[static_cast<size_t>(b + 1)];
            const double p0 = (b > 0) ? z[static_cast<size_t>(b - 1)] : (2.0 * p1 - p2);
            const double p3 = (b + 2 < m) ? z[static_cast<size_t>(b + 2)] : (2.0 * p2 - p1);
            const double t2 = t * t;
            const double t3 = t2 * t;
            out[static_cast<size_t>(i)] = 0.5 * (
                2.0 * p1 +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            );
        }
    }

    return out;
}

double compute_passband_span_khz(
    const std::vector<double>& freqs_mhz,
    int first_pass,
    int last_pass,
    int n,
    double sample_rate_hz
) {
    if (first_pass >= 0 && last_pass > first_pass &&
        static_cast<int>(freqs_mhz.size()) == n) {
        const double span = std::abs(freqs_mhz[static_cast<size_t>(last_pass)] -
                                     freqs_mhz[static_cast<size_t>(first_pass)]) * 1e3;
        if (span > 1e-3) return span;
    }
    const double valid_frac = static_cast<double>(std::max(1, last_pass - first_pass + 1)) /
                              static_cast<double>(std::max(1, n));
    return std::max(1.0, sample_rate_hz * 1e-3 * valid_frac);
}

// Helper for RippleFilter and SavitzkyGolay: build a curvature-inpainted continuum array where
// masked hi_window and outer skirt bins come from the D3^T D3 bridge, while clean passband bins
// stay intact (symmetric 3.5*sigma_MAD outliers replaced by the bridge).
std::vector<double> build_inpainted_work_spectrum(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    int first_pass = -1, last_pass = -1;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[static_cast<size_t>(i)]) {
            if (first_pass < 0) first_pass = i;
            last_pass = i;
        }
    }
    if (first_pass < 0 || last_pass - first_pass < 3) return p_db;

    const int valid_span = last_pass - first_pass + 1;
    const int m = compute_macro_bin_count(valid_span);
    const double passband_span_khz = compute_passband_span_khz(freqs_mhz, first_pass, last_pass, n, config.sample_rate_hz);
    const double bridge_cutoff_khz = std::clamp(
        std::max(3.0 * config.filter_smoothing_sigma_khz, 250.0),
        100.0,
        std::max(120.0, 0.85 * passband_span_khz)
    );
    const double lam3 = compute_whittaker_lambda(m, passband_span_khz, bridge_cutoff_khz, 3, 1.0);

    std::vector<double> bridge = solve_whittaker_d3_gap_bridge(
        p_db, passband_mask, hi_window, lam3, 0.5, true, 3
    );

    thread_local std::vector<double> res_clean;
    res_clean.clear();
    if (res_clean.capacity() < static_cast<size_t>(n)) res_clean.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (passband_mask[static_cast<size_t>(i)] && !hi_window[static_cast<size_t>(i)]) {
            res_clean.push_back(p_db[static_cast<size_t>(i)] - bridge[static_cast<size_t>(i)]);
        }
    }
    if (res_clean.empty()) return bridge;

    const double med_r = median_of_inplace(res_clean);
    for (double& r : res_clean) r = std::abs(r - med_r);
    const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(res_clean));
    const double clip_thresh = 3.5 * mad_sigma;

    std::vector<double> work(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const double b_val = bridge[idx] + med_r;
        if (passband_mask[idx] && !hi_window[idx]) {
            work[idx] = (std::abs(p_db[idx] - b_val) <= clip_thresh) ? p_db[idx] : b_val;
        } else {
            work[idx] = b_val;
        }
    }
    return work;
}

} // anonymous namespace

// =========================================================================
// Fast O(N) Symmetric Positive-Definite Pentadiagonal LDL^T Solver (BW=2)
// =========================================================================
void solve_pentadiagonal_symmetric(
    int n,
    const double* diag,   // A[i, i]
    const double* sub1,   // A[i+1, i]
    const double* sub2,   // A[i+2, i]
    const double* rhs,
    double* x
) {
    if (n <= 0) return;
    if (n == 1) {
        const double d0 = (std::abs(diag[0]) < 1e-15) ? 1e-15 : diag[0];
        x[0] = rhs[0] / d0;
        return;
    }

    thread_local std::vector<double> tl_D;
    thread_local std::vector<double> tl_invD;
    thread_local std::vector<double> tl_L1;
    thread_local std::vector<double> tl_L2;

    if (static_cast<int>(tl_D.size()) < n) {
        tl_D.resize(static_cast<size_t>(n));
        tl_invD.resize(static_cast<size_t>(n));
    }
    if (static_cast<int>(tl_L1.size()) < std::max(1, n - 1)) tl_L1.resize(static_cast<size_t>(std::max(1, n - 1)));
    if (static_cast<int>(tl_L2.size()) < std::max(1, n - 2)) tl_L2.resize(static_cast<size_t>(std::max(1, n - 2)));

    double* D = tl_D.data();
    double* invD = tl_invD.data();
    double* L1 = tl_L1.data();
    double* L2 = tl_L2.data();

    for (int i = 0; i < n; ++i) {
        double d_val = diag[i];
        if (i >= 1) d_val -= L1[i - 1] * L1[i - 1] * D[i - 1];
        if (i >= 2) d_val -= L2[i - 2] * L2[i - 2] * D[i - 2];
        if (d_val < 1e-15) d_val = 1e-15;
        D[i] = d_val;
        const double inv_d = 1.0 / d_val;
        invD[i] = inv_d;

        if (i < n - 1) {
            double num1 = sub1[i];
            if (i >= 1) num1 -= L2[i - 1] * L1[i - 1] * D[i - 1];
            L1[i] = num1 * inv_d;
        }
        if (i < n - 2) {
            L2[i] = sub2[i] * inv_d;
        }
    }

    for (int i = 0; i < n; ++i) {
        double val = rhs[i];
        if (i >= 1) val -= L1[i - 1] * x[i - 1];
        if (i >= 2) val -= L2[i - 2] * x[i - 2];
        x[i] = val;
    }

    for (int i = 0; i < n; ++i) {
        x[i] *= invD[i];
    }

    for (int i = n - 1; i >= 0; --i) {
        double val = x[i];
        if (i < n - 1) val -= L1[i] * x[i + 1];
        if (i < n - 2) val -= L2[i] * x[i + 2];
        x[i] = val;
    }
}

// =========================================================================
// Fast O(N) Symmetric Positive-Definite Heptadiagonal LDL^T Solver (BW=3)
// =========================================================================
void solve_heptadiagonal_symmetric(
    int n,
    const double* d,    // main diagonal [0..n-1]
    const double* e1,   // sub/super diagonal 1 [0..n-2]
    const double* e2,   // sub/super diagonal 2 [0..n-3]
    const double* e3,   // sub/super diagonal 3 [0..n-4]
    const double* rhs,  // right-hand side [0..n-1]
    double* x           // solution output [0..n-1]
) {
    if (n <= 0) return;
    if (n == 1) {
        const double d0 = (d[0] < 1e-14) ? 1e-14 : d[0];
        x[0] = rhs[0] / d0;
        return;
    }

    thread_local std::vector<double> tl_D;
    thread_local std::vector<double> tl_invD;
    thread_local std::vector<double> tl_L1;
    thread_local std::vector<double> tl_L2;
    thread_local std::vector<double> tl_L3;

    if (static_cast<int>(tl_D.size()) < n) {
        tl_D.resize(static_cast<size_t>(n));
        tl_invD.resize(static_cast<size_t>(n));
    }
    if (static_cast<int>(tl_L1.size()) < std::max(1, n - 1)) tl_L1.resize(static_cast<size_t>(std::max(1, n - 1)));
    if (static_cast<int>(tl_L2.size()) < std::max(1, n - 2)) tl_L2.resize(static_cast<size_t>(std::max(1, n - 2)));
    if (static_cast<int>(tl_L3.size()) < std::max(1, n - 3)) tl_L3.resize(static_cast<size_t>(std::max(1, n - 3)));

    double* D = tl_D.data();
    double* invD = tl_invD.data();
    double* L1 = tl_L1.data();
    double* L2 = tl_L2.data();
    double* L3 = tl_L3.data();

    // Factorization: A = L * D * L^T (bandwidth 3)
    for (int i = 0; i < n; ++i) {
        double d_val = d[i];
        if (i >= 1) d_val -= L1[i - 1] * L1[i - 1] * D[i - 1];
        if (i >= 2) d_val -= L2[i - 2] * L2[i - 2] * D[i - 2];
        if (i >= 3) d_val -= L3[i - 3] * L3[i - 3] * D[i - 3];
        if (d_val < 1e-14) d_val = 1e-14;
        D[i] = d_val;
        const double inv_d = 1.0 / d_val;
        invD[i] = inv_d;

        if (i < n - 1) {
            double num1 = e1[i];
            if (i >= 1) num1 -= L2[i - 1] * L1[i - 1] * D[i - 1];
            if (i >= 2) num1 -= L3[i - 2] * L2[i - 2] * D[i - 2];
            L1[i] = num1 * inv_d;
        }
        if (i < n - 2) {
            double num2 = e2[i];
            if (i >= 1) num2 -= L3[i - 1] * L1[i - 1] * D[i - 1];
            L2[i] = num2 * inv_d;
        }
        if (i < n - 3) {
            L3[i] = e3[i] * inv_d;
        }
    }

    // Forward substitution: L * y = rhs
    for (int i = 0; i < n; ++i) {
        double val = rhs[i];
        if (i >= 1) val -= L1[i - 1] * x[i - 1];
        if (i >= 2) val -= L2[i - 2] * x[i - 2];
        if (i >= 3) val -= L3[i - 3] * x[i - 3];
        x[i] = val;
    }

    // Diagonal scaling: z = D^-1 * y
    for (int i = 0; i < n; ++i) {
        x[i] *= invD[i];
    }

    // Back substitution: L^T * x = z
    for (int i = n - 1; i >= 0; --i) {
        double val = x[i];
        if (i < n - 1) val -= L1[i] * x[i + 1];
        if (i < n - 2) val -= L2[i] * x[i + 2];
        if (i < n - 3) val -= L3[i] * x[i + 3];
        x[i] = val;
    }
}

// =========================================================================
// Solver 1: Whittaker / ALS (Quintic D3^T D3 default, Cubic D2^T D2 option)
// =========================================================================
std::vector<double> BaselineFitter::fit_als(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    int first_pass = -1, last_pass = -1;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[static_cast<size_t>(i)]) {
            if (first_pass < 0) first_pass = i;
            last_pass = i;
        }
    }
    if (first_pass < 0 || last_pass - first_pass < 3) return p_db;

    const int valid_span = last_pass - first_pass + 1;
    const int m = compute_macro_bin_count(valid_span);
    const double passband_span_khz = compute_passband_span_khz(freqs_mhz, first_pass, last_pass, n, config.sample_rate_hz);

    const int diff_order = (config.als_diff_order == 2) ? 2 : 3;
    const double lam_scale = std::max(1e-3, config.als_lambda / 200.0);
    const double lam = compute_whittaker_lambda(m, passband_span_khz, config.als_cutoff_khz, diff_order, lam_scale);
    const bool use_huber = (std::abs(config.als_p - 0.5) < 0.005);

    return solve_whittaker_d3_gap_bridge(p_db, passband_mask, hi_window, lam, config.als_p, use_huber, diff_order);
}

// =========================================================================
// Solver 2: Orthogonal Chebyshev Polynomial via Householder QR + Huber IRLS
// =========================================================================
std::vector<double> BaselineFitter::fit_poly(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    const int deg = std::clamp(config.poly_degree, 2, 20);
    const int p = deg + 1;

    int first_pass = -1, last_pass = -1;
    int clean_count = 0;
    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (passband_mask[idx]) {
            if (first_pass < 0) first_pass = i;
            last_pass = i;
            if (!hi_window[idx]) ++clean_count;
        }
    }
    if (first_pass < 0 || clean_count < p + 2) return p_db;

    const bool use_freq = (static_cast<int>(freqs_mhz.size()) == n) &&
                          (std::abs(freqs_mhz[static_cast<size_t>(last_pass)] -
                                    freqs_mhz[static_cast<size_t>(first_pass)]) > 1e-9);
    const double x_lo = use_freq ? freqs_mhz[static_cast<size_t>(first_pass)] : static_cast<double>(first_pass);
    const double x_hi = use_freq ? freqs_mhz[static_cast<size_t>(last_pass)] : static_cast<double>(last_pass);
    const double x_mid = 0.5 * (x_lo + x_hi);
    const double inv_half = 1.0 / (0.5 * (x_hi - x_lo));

    thread_local std::vector<double> u_vec, weights, res_scratch;
    u_vec.resize(static_cast<size_t>(n));
    weights.assign(static_cast<size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const double xv = use_freq ? freqs_mhz[idx] : static_cast<double>(i);
        u_vec[idx] = (xv - x_mid) * inv_half;
        if (passband_mask[idx] && !hi_window[idx]) {
            weights[idx] = 1.0;
        }
    }

    thread_local ChunkedHouseholderQR qr;
    double coeffs[21] = {};
    double phi[21] = {};

    // 5-iteration symmetric Huber MAD IRLS with curvature-regularized Householder QR
    constexpr int kIters = 5;
    for (int iter = 0; iter < kIters; ++iter) {
        qr.reset(p);
        double sum_w = 0.0;
        for (int i = first_pass; i <= last_pass; ++i) {
            const size_t idx = static_cast<size_t>(i);
            const double wi = weights[idx];
            if (wi <= 0.0) continue;
            eval_chebyshev_basis(u_vec[idx], deg, phi);
            qr.add_row(phi, p_db[idx], std::sqrt(wi));
            sum_w += wi;
        }

        // Mild curvature-regularization pseudo-rows: sqrt(gamma) * k^2 * c_k = 0 (k >= 2)
        const double sqrt_gamma = std::sqrt(2e-9 * std::max(1.0, sum_w));
        for (int k = 2; k <= deg; ++k) {
            qr.add_reg_diag(k, sqrt_gamma * static_cast<double>(k * k));
        }
        qr.solve(coeffs);

        if (iter + 1 >= kIters) break;

        res_scratch.clear();
        if (res_scratch.capacity() < static_cast<size_t>(clean_count)) {
            res_scratch.reserve(static_cast<size_t>(clean_count));
        }
        for (int i = first_pass; i <= last_pass; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (passband_mask[idx] && !hi_window[idx]) {
                const double fitted = eval_chebyshev_clenshaw(u_vec[idx], coeffs, deg);
                res_scratch.push_back(p_db[idx] - fitted);
            }
        }
        const double med_r = median_of_inplace(res_scratch);
        for (double& r : res_scratch) r = std::abs(r - med_r);
        const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(res_scratch));
        const double delta = 1.345 * mad_sigma;

        for (int i = first_pass; i <= last_pass; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (passband_mask[idx] && !hi_window[idx]) {
                const double fitted = eval_chebyshev_clenshaw(u_vec[idx], coeffs, deg);
                const double a = std::abs(p_db[idx] - fitted);
                weights[idx] = (a <= delta) ? 1.0 : (delta / a);
            }
        }
    }

    // Evaluate smoothly: Clenshaw inside [-1, 1], bounded C1 tangent continuation on outer skirts
    const double y_left = eval_chebyshev_clenshaw(-1.0, coeffs, deg);
    const double y_right = eval_chebyshev_clenshaw(1.0, coeffs, deg);
    const double s_left = eval_chebyshev_endpoint_deriv(coeffs, deg, -1.0);
    const double s_right = eval_chebyshev_endpoint_deriv(coeffs, deg, 1.0);

    std::vector<double> baseline_db(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const double u = u_vec[idx];
        if (u < -1.0) {
            baseline_db[idx] = smooth_tangent_extrap(y_left, s_left, -1.0 - u, -1.0, 0.15, 6.0);
        } else if (u > 1.0) {
            baseline_db[idx] = smooth_tangent_extrap(y_right, s_right, u - 1.0, +1.0, 0.15, 6.0);
        } else {
            baseline_db[idx] = eval_chebyshev_clenshaw(u, coeffs, deg);
        }
    }

    return baseline_db;
}

// =========================================================================
// Solver 3: Ripple Filter (D3^T D3 Curvature Inpainting + Tukey Twicing)
// =========================================================================
std::vector<double> BaselineFitter::fit_filter(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    std::vector<double> work = build_inpainted_work_spectrum(p_db, freqs_mhz, passband_mask, hi_window, config);

    const double bin_res_hz = (config.sample_rate_hz > 0.0)
                                  ? (config.sample_rate_hz / static_cast<double>(n))
                                  : 1.0;
    const double sigma_bins = std::clamp(
        (config.filter_smoothing_sigma_khz * 1e3) / std::max(1e-6, bin_res_hz),
        3.0,
        std::max(3.0, 0.45 * static_cast<double>(n))
    );

    // Tukey twicing: B1 = G(work); B2 = B1 + G(work - B1)
    // Transfer function: 2*G - G^2 = 1 - (1 - G)^2 -> <0.2% attenuation on 0.75 MHz ripples
    std::vector<double> b1 = gaussian_smooth_1d(work, sigma_bins);
    std::vector<double> resid(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        resid[static_cast<size_t>(i)] = work[static_cast<size_t>(i)] - b1[static_cast<size_t>(i)];
    }
    std::vector<double> g_resid = gaussian_smooth_1d(resid, sigma_bins);
    std::vector<double> b2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        b2[static_cast<size_t>(i)] = b1[static_cast<size_t>(i)] + g_resid[static_cast<size_t>(i)];
    }
    return b2;
}

// =========================================================================
// Solver 4: Moment-Preserving Degree-4 Savitzky-Golay Filter (O(N log N))
// =========================================================================
std::vector<double> BaselineFitter::fit_savgol(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 8) return p_db;

    std::vector<double> work = build_inpainted_work_spectrum(p_db, freqs_mhz, passband_mask, hi_window, config);

    const double bin_res_hz = (config.sample_rate_hz > 0.0)
                                  ? (config.sample_rate_hz / static_cast<double>(n))
                                  : 1.0;
    const double sigma_bins = std::max(3.0, (config.filter_smoothing_sigma_khz * 1e3) / std::max(1e-6, bin_res_hz));
    const int H = std::clamp(static_cast<int>(std::round(2.4 * sigma_bins)), 5, std::max(5, n / 6));

    // Exact degree-4 Savitzky-Golay weights at k=0 over [-H..H] via 3x3 even-moment Vandermonde system
    // Normalized coordinate u_k = k / H in [-1, 1] avoids H^8 conditioning blowup for large H.
    double S[5] = {}; // S[r] = sum_{k=-H}^H (k/H)^(2*r), r = 0..4
    S[0] = 1.0;       // k = 0 term
    const double inv_H = 1.0 / static_cast<double>(H);
    for (int k = 1; k <= H; ++k) {
        const double u = static_cast<double>(k) * inv_H;
        const double u2 = u * u;
        double pow_u = 1.0;
        for (int r = 0; r < 5; ++r) {
            S[r] += 2.0 * pow_u;
            pow_u *= u2;
        }
    }

    const double M[3][3] = {
        {S[0], S[1], S[2]},
        {S[1], S[2], S[3]},
        {S[2], S[3], S[4]}
    };
    const double rhs_sg[3] = {1.0, 0.0, 0.0};
    double a_sg[3] = {1.0 / S[0], 0.0, 0.0};
    solve_3x3(M, rhs_sg, a_sg);

    std::vector<double> hk(static_cast<size_t>(H) + 1);
    for (int k = 0; k <= H; ++k) {
        const double u = static_cast<double>(k) * inv_H;
        const double u2 = u * u;
        hk[static_cast<size_t>(k)] = a_sg[0] + a_sg[1] * u2 + a_sg[2] * (u2 * u2);
    }

    return convolve_symmetric_kernel_1d(work, hk, true);
}

// =========================================================================
// Solver 5: Coaxial Standing-Wave Reflection Solver (WLS Periodogram + QR)
// =========================================================================
std::vector<double> BaselineFitter::fit_standing_wave(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config,
    int* out_components_used,
    double* out_period1_mhz,
    double* out_period2_mhz
) {
    if (out_components_used) *out_components_used = 0;
    if (out_period1_mhz) *out_period1_mhz = 0.0;
    if (out_period2_mhz) *out_period2_mhz = 0.0;

    const int n = static_cast<int>(p_db.size());
    if (n < 16) return p_db;

    const int d_c = std::clamp(config.poly_degree, 4, 6);
    const int p_poly = d_c + 1;

    int first_pass = -1, last_pass = -1;
    int hi_first = -1, hi_last = -1;
    int clean_count = 0;
    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (passband_mask[idx]) {
            if (first_pass < 0) first_pass = i;
            last_pass = i;
            if (hi_window[idx]) {
                if (hi_first < 0) hi_first = i;
                hi_last = i;
            } else {
                ++clean_count;
            }
        }
    }
    if (first_pass < 0 || clean_count < p_poly + 8) return p_db;

    const bool use_freq = (static_cast<int>(freqs_mhz.size()) == n) &&
                          (std::abs(freqs_mhz[static_cast<size_t>(last_pass)] -
                                    freqs_mhz[static_cast<size_t>(first_pass)]) > 1e-9);
    const double x_lo = use_freq ? freqs_mhz[static_cast<size_t>(first_pass)] : static_cast<double>(first_pass);
    const double x_hi = use_freq ? freqs_mhz[static_cast<size_t>(last_pass)] : static_cast<double>(last_pass);
    const double x_mid = 0.5 * (x_lo + x_hi);
    const double x_half = 0.5 * (x_hi - x_lo);
    const double inv_half = 1.0 / x_half;

    const double passband_span_mhz = compute_passband_span_khz(freqs_mhz, first_pass, last_pass, n, config.sample_rate_hz) * 1e-3;
    const double bin_width_mhz = passband_span_mhz / static_cast<double>(std::max(1, last_pass - first_pass));
    const double gap_mhz = (hi_first >= 0)
                               ? (use_freq
                                      ? (std::abs(freqs_mhz[static_cast<size_t>(hi_last)] -
                                                  freqs_mhz[static_cast<size_t>(hi_first)]) + bin_width_mhz)
                                      : (static_cast<double>(hi_last - hi_first + 1) * bin_width_mhz))
                               : 0.0;

    std::vector<double> u_vec(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double xv = use_freq ? freqs_mhz[static_cast<size_t>(i)] : static_cast<double>(i);
        u_vec[static_cast<size_t>(i)] = (xv - x_mid) * inv_half;
    }

    // Robust clean passband dynamic range for physical amplitude/gap-stability checks
    std::vector<double> clean_vals;
    clean_vals.reserve(static_cast<size_t>(clean_count));
    for (int i = first_pass; i <= last_pass; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (passband_mask[idx] && !hi_window[idx]) clean_vals.push_back(p_db[idx]);
    }
    std::sort(clean_vals.begin(), clean_vals.end());
    const double p05_db = clean_vals[clean_vals.size() / 20];
    const double p95_db = clean_vals[(clean_vals.size() * 19) / 20];
    const double clean_span_db = std::max(0.5, p95_db - p05_db);

    // Build clean macro-bins strictly on passband_mask && !hi_window (NO zero-filling of H I gap!)
    const int valid_span = last_pass - first_pass + 1;
    const int m_target = (valid_span <= 64) ? valid_span : std::clamp(valid_span / 16, 64, 240);
    std::vector<double> macro_u;
    std::vector<double> macro_y;
    std::vector<double> macro_w;
    macro_u.reserve(static_cast<size_t>(m_target));
    macro_y.reserve(static_cast<size_t>(m_target));
    macro_w.reserve(static_cast<size_t>(m_target));

    double total_macro_w = 0.0;
    for (int b = 0; b < m_target; ++b) {
        const int i0 = first_pass + static_cast<int>((static_cast<int64_t>(b) * valid_span) / m_target);
        int i1 = first_pass + static_cast<int>((static_cast<int64_t>(b + 1) * valid_span) / m_target);
        if (i1 <= i0) i1 = std::min(last_pass + 1, i0 + 1);

        double sum_u = 0.0, sum_y = 0.0;
        int cnt = 0;
        for (int i = i0; i < i1; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (passband_mask[idx] && !hi_window[idx]) {
                sum_u += u_vec[idx];
                sum_y += p_db[idx];
                ++cnt;
            }
        }
        if (cnt > 0) {
            macro_u.push_back(sum_u / static_cast<double>(cnt));
            macro_y.push_back(sum_y / static_cast<double>(cnt));
            const double wb = static_cast<double>(cnt);
            macro_w.push_back(wb);
            total_macro_w += wb;
        }
    }
    const int m_clean = static_cast<int>(macro_u.size());
    if (m_clean < p_poly + 6) {
        AstroConfig fallback_cfg = config;
        fallback_cfg.poly_degree = d_c;
        return fit_poly(p_db, freqs_mhz, passband_mask, hi_window, fallback_cfg);
    }

    // Helper: angular frequency in u-coordinates for delay frequency tau = 1/P [cycles/MHz]
    auto tau_to_omega_u = [&](double tau_cyc_per_mhz) -> double {
        return kPi * passband_span_mhz * tau_cyc_per_mhz;
    };

    // Helper: evaluate joint model (Chebyshev T_0..T_{d_c} + sinusoidal modes) at u
    auto eval_model = [&](double u, const double* coeffs, const std::vector<double>& taus) -> double {
        double val = eval_chebyshev_clenshaw(u, coeffs, d_c);
        for (size_t k = 0; k < taus.size(); ++k) {
            const double ang = tau_to_omega_u(taus[k]) * u;
            const int base = p_poly + 2 * static_cast<int>(k);
            val += coeffs[base] * std::cos(ang) + coeffs[base + 1] * std::sin(ang);
        }
        return val;
    };

    // Helper: solve joint QR on clean macro-bins for given set of mode frequencies
    thread_local ChunkedHouseholderQR qr;
    auto solve_macro_joint = [&](const std::vector<double>& taus, double* out_coeffs, double& out_macro_rss) {
        const int n_modes = static_cast<int>(taus.size());
        const int p_tot = p_poly + 2 * n_modes;
        double phi[16] = {};
        double omegas[4] = {};
        for (int k = 0; k < n_modes; ++k) omegas[k] = tau_to_omega_u(taus[k]);

        qr.reset(p_tot);
        for (int b = 0; b < m_clean; ++b) {
            const double ub = macro_u[static_cast<size_t>(b)];
            eval_chebyshev_basis(ub, d_c, phi);
            for (int k = 0; k < n_modes; ++k) {
                const double ang = omegas[k] * ub;
                phi[p_poly + 2 * k] = std::cos(ang);
                phi[p_poly + 2 * k + 1] = std::sin(ang);
            }
            qr.add_row(phi, macro_y[static_cast<size_t>(b)], std::sqrt(macro_w[static_cast<size_t>(b)]));
        }
        const double sqrt_gamma = std::sqrt(1e-8 * std::max(1.0, total_macro_w));
        for (int k = 2; k <= d_c; ++k) {
            qr.add_reg_diag(k, sqrt_gamma * static_cast<double>(k * k));
        }
        const double sin_reg = std::sqrt(1e-10 * std::max(1.0, total_macro_w));
        for (int k = 0; k < 2 * n_modes; ++k) {
            qr.add_reg_diag(p_poly + k, sin_reg);
        }
        qr.solve(out_coeffs, &out_macro_rss);
    };

    // Helper: compute full-resolution RSS on passband_mask && !hi_window for exact BIC gating
    auto compute_full_rss = [&](const double* coeffs, const std::vector<double>& taus) -> double {
        double omegas[4] = {};
        for (size_t k = 0; k < taus.size(); ++k) omegas[k] = tau_to_omega_u(taus[k]);
        double rss_full = 0.0;
        for (int i = first_pass; i <= last_pass; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (!passband_mask[idx] || hi_window[idx]) continue;
            const double u = u_vec[idx];
            double pred = eval_chebyshev_clenshaw(u, coeffs, d_c);
            for (size_t k = 0; k < taus.size(); ++k) {
                const double ang = omegas[k] * u;
                const int base = p_poly + 2 * static_cast<int>(k);
                pred += coeffs[base] * std::cos(ang) + coeffs[base + 1] * std::sin(ang);
            }
            const double r = p_db[idx] - pred;
            rss_full += r * r;
        }
        return rss_full;
    };

    // Stage A: Fit Chebyshev degree d_c via Householder QR strictly on clean bins
    std::vector<double> accepted_taus;
    double cur_coeffs[16] = {};
    double stage_a_macro_rss = 0.0;
    solve_macro_joint(accepted_taus, cur_coeffs, stage_a_macro_rss);
    (void)stage_a_macro_rss;
    double cur_full_rss = compute_full_rss(cur_coeffs, accepted_taus);

    // Stage B: Physical standing-wave period search & BIC + gap-stability gating
    const double P_min = std::max(std::max(0.02, config.standing_wave_min_period_mhz), 0.9 * gap_mhz);
    const double P_max = 0.85 * passband_span_mhz;
    const int max_comp = std::clamp(config.standing_wave_max_components, 1, 2);
    const double min_sep_tau = 0.75 / std::max(1e-6, passband_span_mhz);
    const double n_eff = std::max(10.0, static_cast<double>(clean_count) / 2.35);

    if (P_min < P_max && std::sqrt(cur_full_rss / static_cast<double>(clean_count)) > 1e-6) {
        const double tau_min = 1.0 / P_max;
        const double tau_max = 1.0 / P_min;
        const double d_tau = 1.0 / (12.0 * std::max(1e-6, passband_span_mhz));
        const int n_grid = std::clamp(static_cast<int>(std::ceil((tau_max - tau_min) / d_tau)) + 1, 16, 2000);
        const double step_tau = (n_grid > 1) ? ((tau_max - tau_min) / static_cast<double>(n_grid - 1)) : 0.0;

        std::vector<double> macro_res(static_cast<size_t>(m_clean));
        std::vector<double> grid_scores(static_cast<size_t>(n_grid), 0.0);

        for (int mode_idx = 0; mode_idx < max_comp; ++mode_idx) {
            // Compute clean macro-bin residuals from currently accepted model
            for (int b = 0; b < m_clean; ++b) {
                macro_res[static_cast<size_t>(b)] =
                    macro_y[static_cast<size_t>(b)] - eval_model(macro_u[static_cast<size_t>(b)], cur_coeffs, accepted_taus);
            }

            // Scan frequencies with exact 3x3 floating-mean WLS periodogram
            double W0 = 0.0, Wy = 0.0;
            for (int b = 0; b < m_clean; ++b) {
                W0 += macro_w[static_cast<size_t>(b)];
                Wy += macro_w[static_cast<size_t>(b)] * macro_res[static_cast<size_t>(b)];
            }
            const double mean_term = (W0 > 1e-12) ? (Wy * Wy / W0) : 0.0;

            for (int g = 0; g < n_grid; ++g) {
                const double tau = tau_min + static_cast<double>(g) * step_tau;
                bool too_close = false;
                for (double at : accepted_taus) {
                    if (std::abs(tau - at) < min_sep_tau) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) {
                    grid_scores[static_cast<size_t>(g)] = -1.0;
                    continue;
                }

                const double om = tau_to_omega_u(tau);
                double Wc = 0.0, Ws = 0.0, Wcc = 0.0, Wcs = 0.0, Wss = 0.0;
                double Wcy = 0.0, Wsy = 0.0;
                for (int b = 0; b < m_clean; ++b) {
                    const double wb = macro_w[static_cast<size_t>(b)];
                    const double ang = om * macro_u[static_cast<size_t>(b)];
                    const double cb = std::cos(ang);
                    const double sb = std::sin(ang);
                    const double rb = macro_res[static_cast<size_t>(b)];
                    Wc  += wb * cb;
                    Ws  += wb * sb;
                    Wcc += wb * cb * cb;
                    Wcs += wb * cb * sb;
                    Wss += wb * sb * sb;
                    Wcy += wb * cb * rb;
                    Wsy += wb * sb * rb;
                }
                const double reg = 1e-12 * std::max(1.0, W0);
                const double A3[3][3] = {
                    {W0 + reg, Wc,        Ws},
                    {Wc,       Wcc + reg, Wcs},
                    {Ws,       Wcs,       Wss + reg}
                };
                const double b3[3] = {Wy, Wcy, Wsy};
                double sol[3] = {};
                if (solve_3x3(A3, b3, sol)) {
                    grid_scores[static_cast<size_t>(g)] = std::max(0.0, sol[0] * Wy + sol[1] * Wcy + sol[2] * Wsy - mean_term);
                } else {
                    grid_scores[static_cast<size_t>(g)] = 0.0;
                }
            }

            // Pick up to top 3 well-separated periodogram peaks for joint golden-section refinement
            std::vector<int> top_peaks;
            for (int pk = 0; pk < 3; ++pk) {
                int best_g = -1;
                double best_s = 0.0;
                for (int g = 0; g < n_grid; ++g) {
                    if (grid_scores[static_cast<size_t>(g)] <= best_s) continue;
                    const double tau_g = tau_min + static_cast<double>(g) * step_tau;
                    bool near_prev = false;
                    for (int prev_g : top_peaks) {
                        const double tau_p = tau_min + static_cast<double>(prev_g) * step_tau;
                        if (std::abs(tau_g - tau_p) < 0.5 / std::max(1e-6, passband_span_mhz)) {
                            near_prev = true;
                            break;
                        }
                    }
                    if (!near_prev) {
                        best_s = grid_scores[static_cast<size_t>(g)];
                        best_g = g;
                    }
                }
                if (best_g < 0) break;
                top_peaks.push_back(best_g);
            }
            if (top_peaks.empty()) break;

            // 15-step golden-section search minimizing the JOINT macro-bin QR residual
            constexpr double kInvPhi = 0.61803398874989484820;
            constexpr double kInvPhi2 = 0.38196601125010515180;
            const double half_bracket = std::max(2.0 * step_tau, 0.30 / std::max(1e-6, passband_span_mhz));
            const double max_allowed_amp = std::max(1.5, 1.5 * clean_span_db);

            double best_cand_tau = -1.0;
            double best_cand_macro_rss = 1e300;
            double best_cand_coeffs[16] = {};

            for (int g_peak : top_peaks) {
                const double tau0 = tau_min + static_cast<double>(g_peak) * step_tau;
                double lo = std::max(tau_min, tau0 - half_bracket);
                double hi = std::min(tau_max, tau0 + half_bracket);
                for (double at : accepted_taus) {
                    if (at < tau0) lo = std::max(lo, at + min_sep_tau);
                    if (at > tau0) hi = std::min(hi, at - min_sep_tau);
                }
                if (hi <= lo) continue;

                auto eval_joint_obj = [&](double tau_val) -> double {
                    std::vector<double> trial_taus = accepted_taus;
                    trial_taus.push_back(tau_val);
                    double tmp_c[16] = {};
                    double tmp_rss = 1e300;
                    solve_macro_joint(trial_taus, tmp_c, tmp_rss);
                    return tmp_rss;
                };

                double c = lo + kInvPhi2 * (hi - lo);
                double d = lo + kInvPhi * (hi - lo);
                double fc = eval_joint_obj(c);
                double fd = eval_joint_obj(d);
                for (int gs = 0; gs < 15; ++gs) {
                    if (fc < fd) {
                        hi = d;
                        d = c;
                        fd = fc;
                        c = lo + kInvPhi2 * (hi - lo);
                        fc = eval_joint_obj(c);
                    } else {
                        lo = c;
                        c = d;
                        fc = fd;
                        d = lo + kInvPhi * (hi - lo);
                        fd = eval_joint_obj(d);
                    }
                }
                const double refined_tau = (fc < fd) ? c : d;
                std::vector<double> trial_taus = accepted_taus;
                trial_taus.push_back(refined_tau);
                double trial_c[16] = {};
                double trial_rss = 1e300;
                solve_macro_joint(trial_taus, trial_c, trial_rss);

                bool amps_physical = true;
                for (size_t k = 0; k < trial_taus.size(); ++k) {
                    const int base = p_poly + 2 * static_cast<int>(k);
                    const double amp = std::hypot(trial_c[base], trial_c[base + 1]);
                    if (!std::isfinite(amp) || amp > max_allowed_amp) {
                        amps_physical = false;
                        break;
                    }
                }
                if (amps_physical && trial_rss < best_cand_macro_rss) {
                    best_cand_macro_rss = trial_rss;
                    best_cand_tau = refined_tau;
                    std::memcpy(best_cand_coeffs, trial_c, sizeof(best_cand_coeffs));
                }
            }

            if (best_cand_tau <= 0.0) break;

            // Evaluate strict delta-BIC > 12.0 on full-resolution clean bins (N_eff = N_clean / 2.35)
            std::vector<double> cand_taus = accepted_taus;
            cand_taus.push_back(best_cand_tau);
            const double cand_full_rss = compute_full_rss(best_cand_coeffs, cand_taus);

            const double safe_rss0 = std::max(1e-30, cur_full_rss);
            const double safe_rss1 = std::max(1e-30, cand_full_rss);
            const double delta_bic = n_eff * std::log(safe_rss0 / safe_rss1) - 3.0 * std::log(n_eff);
            if (!(delta_bic > 12.0)) break;

            // Physicality & gap-stability gate: prevent destructive interference / huge cancellation inside H I gap
            bool physical_ok = true;
            for (size_t k = 0; k < cand_taus.size(); ++k) {
                const int base = p_poly + 2 * static_cast<int>(k);
                const double amp = std::hypot(best_cand_coeffs[base], best_cand_coeffs[base + 1]);
                if (!std::isfinite(amp) || amp > max_allowed_amp) {
                    physical_ok = false;
                    break;
                }
            }

            if (physical_ok && hi_first >= 0) {
                double max_abs_sin_clean = 0.0;
                double max_abs_sin_gap = 0.0;
                double max_clean_dev = 0.0;
                double max_gap_dev = 0.0;
                double min_clean_m = 1e300, max_clean_m = -1e300;
                double min_gap_m = 1e300, max_gap_m = -1e300;

                for (int i = first_pass; i <= last_pass; ++i) {
                    const size_t idx = static_cast<size_t>(i);
                    if (!passband_mask[idx]) continue;
                    const double u = u_vec[idx];
                    double sin_sum = 0.0;
                    for (size_t k = 0; k < cand_taus.size(); ++k) {
                        const double ang = tau_to_omega_u(cand_taus[k]) * u;
                        const int base = p_poly + 2 * static_cast<int>(k);
                        sin_sum += best_cand_coeffs[base] * std::cos(ang) + best_cand_coeffs[base + 1] * std::sin(ang);
                    }
                    const double m_new = eval_chebyshev_clenshaw(u, best_cand_coeffs, d_c) + sin_sum;
                    const double m_prev = eval_model(u, cur_coeffs, accepted_taus);
                    const double dev = std::abs(m_new - m_prev);

                    if (hi_window[idx]) {
                        max_abs_sin_gap = std::max(max_abs_sin_gap, std::abs(sin_sum));
                        max_gap_dev = std::max(max_gap_dev, dev);
                        min_gap_m = std::min(min_gap_m, m_new);
                        max_gap_m = std::max(max_gap_m, m_new);
                    } else {
                        max_abs_sin_clean = std::max(max_abs_sin_clean, std::abs(sin_sum));
                        max_clean_dev = std::max(max_clean_dev, dev);
                        min_clean_m = std::min(min_clean_m, m_new);
                        max_clean_m = std::max(max_clean_m, m_new);
                    }
                }

                const double env_pad = std::max(1.5, 0.75 * clean_span_db);
                if (max_abs_sin_gap > std::max(0.25, 1.8 * max_abs_sin_clean) ||
                    max_gap_dev > std::max(0.8, 3.0 * std::max(max_clean_dev, max_abs_sin_clean)) ||
                    min_gap_m < min_clean_m - env_pad ||
                    max_gap_m > max_clean_m + env_pad) {
                    physical_ok = false;
                }
            }

            if (!physical_ok) break;

            // Accept standing-wave mode
            accepted_taus = std::move(cand_taus);
            std::memcpy(cur_coeffs, best_cand_coeffs, sizeof(cur_coeffs));
            cur_full_rss = cand_full_rss;

            if (std::sqrt(cur_full_rss / static_cast<double>(clean_count)) <= 1e-6) break;
        }
    }

    // Stage C: Final full-resolution joint Householder QR + 3-iteration symmetric Huber IRLS
    auto solve_full_irls = [&](const std::vector<double>& taus, double* out_coeffs) {
        const int n_modes = static_cast<int>(taus.size());
        const int p_tot = p_poly + 2 * n_modes;
        double omegas[4] = {};
        for (int k = 0; k < n_modes; ++k) omegas[k] = tau_to_omega_u(taus[k]);

        std::vector<double> w_full(static_cast<size_t>(n), 0.0);
        for (int i = first_pass; i <= last_pass; ++i) {
            const size_t idx = static_cast<size_t>(i);
            if (passband_mask[idx] && !hi_window[idx]) w_full[idx] = 1.0;
        }

        std::vector<double> res_scratch;
        res_scratch.reserve(static_cast<size_t>(clean_count));
        double phi[16] = {};

        constexpr int kIters = 3;
        for (int iter = 0; iter < kIters; ++iter) {
            qr.reset(p_tot);
            double sum_w = 0.0;
            for (int i = first_pass; i <= last_pass; ++i) {
                const size_t idx = static_cast<size_t>(i);
                const double wi = w_full[idx];
                if (wi <= 0.0) continue;
                const double u = u_vec[idx];
                eval_chebyshev_basis(u, d_c, phi);
                for (int k = 0; k < n_modes; ++k) {
                    const double ang = omegas[k] * u;
                    phi[p_poly + 2 * k] = std::cos(ang);
                    phi[p_poly + 2 * k + 1] = std::sin(ang);
                }
                qr.add_row(phi, p_db[idx], std::sqrt(wi));
                sum_w += wi;
            }

            const double sqrt_gamma = std::sqrt(1e-8 * std::max(1.0, sum_w));
            for (int k = 2; k <= d_c; ++k) {
                qr.add_reg_diag(k, sqrt_gamma * static_cast<double>(k * k));
            }
            const double sin_reg = std::sqrt(1e-12 * std::max(1.0, sum_w));
            for (int k = 0; k < 2 * n_modes; ++k) {
                qr.add_reg_diag(p_poly + k, sin_reg);
            }
            qr.solve(out_coeffs);

            if (iter + 1 >= kIters) break;

            res_scratch.clear();
            for (int i = first_pass; i <= last_pass; ++i) {
                const size_t idx = static_cast<size_t>(i);
                if (passband_mask[idx] && !hi_window[idx]) {
                    res_scratch.push_back(p_db[idx] - eval_model(u_vec[idx], out_coeffs, taus));
                }
            }
            const double med_r = median_of_inplace(res_scratch);
            for (double& r : res_scratch) r = std::abs(r - med_r);
            const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(res_scratch));
            const double delta = 1.345 * mad_sigma;

            for (int i = first_pass; i <= last_pass; ++i) {
                const size_t idx = static_cast<size_t>(i);
                if (passband_mask[idx] && !hi_window[idx]) {
                    const double a = std::abs(p_db[idx] - eval_model(u_vec[idx], out_coeffs, taus));
                    w_full[idx] = (a <= delta) ? 1.0 : (delta / a);
                }
            }
        }
    };

    double final_coeffs[16] = {};
    solve_full_irls(accepted_taus, final_coeffs);

    // Final safety check on amplitudes after full-resolution IRLS
    bool final_ok = true;
    const double max_allowed_amp_final = std::max(1.5, 1.5 * clean_span_db);
    for (size_t k = 0; k < accepted_taus.size(); ++k) {
        const int base = p_poly + 2 * static_cast<int>(k);
        const double amp = std::hypot(final_coeffs[base], final_coeffs[base + 1]);
        if (!std::isfinite(amp) || amp > max_allowed_amp_final) {
            final_ok = false;
            break;
        }
    }
    if (!final_ok) {
        accepted_taus.clear();
        solve_full_irls(accepted_taus, final_coeffs);
    }

    // Populate telemetry
    if (out_components_used) *out_components_used = static_cast<int>(accepted_taus.size());
    if (out_period1_mhz && accepted_taus.size() >= 1) *out_period1_mhz = 1.0 / accepted_taus[0];
    if (out_period2_mhz && accepted_taus.size() >= 2) *out_period2_mhz = 1.0 / accepted_taus[1];

    // Evaluate smoothly with exact C1 bounded tangent continuation on outer skirts
    auto eval_endpoint_deriv = [&](double u_edge) -> double {
        double d = eval_chebyshev_endpoint_deriv(final_coeffs, d_c, u_edge);
        for (size_t k = 0; k < accepted_taus.size(); ++k) {
            const double om = tau_to_omega_u(accepted_taus[k]);
            const double ang = om * u_edge;
            const int base = p_poly + 2 * static_cast<int>(k);
            d += om * (-final_coeffs[base] * std::sin(ang) + final_coeffs[base + 1] * std::cos(ang));
        }
        return d;
    };

    const double y_left = eval_model(-1.0, final_coeffs, accepted_taus);
    const double y_right = eval_model(1.0, final_coeffs, accepted_taus);
    const double s_left = eval_endpoint_deriv(-1.0);
    const double s_right = eval_endpoint_deriv(1.0);

    std::vector<double> baseline_db(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const double u = u_vec[idx];
        if (u < -1.0) {
            baseline_db[idx] = smooth_tangent_extrap(y_left, s_left, -1.0 - u, -1.0, 0.15, 6.0);
        } else if (u > 1.0) {
            baseline_db[idx] = smooth_tangent_extrap(y_right, s_right, u - 1.0, +1.0, 0.15, 6.0);
        } else {
            baseline_db[idx] = eval_model(u, final_coeffs, accepted_taus);
        }
    }

    return baseline_db;
}

// =========================================================================
// Stage 1 Entry Point: Automatic IF Skirt Trimming, Solver Dispatch,
// Jensen-Unbiased Linear Re-Scaling & Continuum Quality Diagnostics
// =========================================================================
BaselineResults BaselineFitter::fit(
    const std::vector<double>& power,
    const std::vector<double>& freqs_mhz,
    const AstroConfig& config
) {
    const int nfft = static_cast<int>(power.size());
    BaselineResults res;
    res.baseline_power.resize(static_cast<size_t>(nfft), 1.0);
    res.baseline_db.resize(static_cast<size_t>(nfft), 0.0);
    res.effective_passband_mask.assign(static_cast<size_t>(nfft), 1);

    if (nfft == 0) return res;

    // 1. HI protection window
    const double hi_half_mhz = config.hi_protection_half_width_khz / 1e3;
    const double hi_lo = HYDROGEN_REST_FREQ_MHZ - hi_half_mhz;
    const double hi_hi = HYDROGEN_REST_FREQ_MHZ + hi_half_mhz;
    std::vector<uint8_t> hi_window(static_cast<size_t>(nfft), 0);
    if (static_cast<int>(freqs_mhz.size()) == nfft) {
        for (int i = 0; i < nfft; ++i) {
            const double f = freqs_mhz[static_cast<size_t>(i)];
            if (f >= hi_lo && f <= hi_hi) {
                hi_window[static_cast<size_t>(i)] = 1;
            }
        }
    }

    // 2. Vectorized conversion from raw power to dB
    std::vector<double> p_db(static_cast<size_t>(nfft));
    SimdDsp::power_to_db_f64(power.data(), p_db.data(), static_cast<size_t>(nfft), 1e-30);

    // 3. Automatic IF skirt trimming
    const int min_edge = static_cast<int>(static_cast<double>(nfft) * std::clamp(config.passband_edge_fraction, 0.0, 0.40));
    int edge_left = min_edge;
    int edge_right = min_edge;

    if (config.auto_skirt_trim && nfft >= 64) {
        // Compute robust block-median envelope across spectrum
        const int block_sz = std::clamp(nfft / 256, 4, 32);
        const int n_blocks = (nfft + block_sz - 1) / block_sz;
        std::vector<double> blk_med(static_cast<size_t>(n_blocks));
        std::vector<double> blk_ctr(static_cast<size_t>(n_blocks));
        std::vector<double> blk_tmp;
        blk_tmp.reserve(static_cast<size_t>(block_sz));

        for (int b = 0; b < n_blocks; ++b) {
            const int i0 = b * block_sz;
            const int i1 = std::min(nfft, i0 + block_sz);
            blk_tmp.clear();
            for (int i = i0; i < i1; ++i) blk_tmp.push_back(p_db[static_cast<size_t>(i)]);
            blk_med[static_cast<size_t>(b)] = median_of_inplace(blk_tmp);
            blk_ctr[static_cast<size_t>(b)] = 0.5 * static_cast<double>(i0 + i1 - 1);
        }

        std::vector<double> env(static_cast<size_t>(nfft));
        int b = 0;
        for (int i = 0; i < nfft; ++i) {
            const double x = static_cast<double>(i);
            while (b + 1 < n_blocks - 1 && x > blk_ctr[static_cast<size_t>(b + 1)]) ++b;
            if (x <= blk_ctr[0]) {
                env[static_cast<size_t>(i)] = blk_med[0];
            } else if (x >= blk_ctr[static_cast<size_t>(n_blocks - 1)]) {
                env[static_cast<size_t>(i)] = blk_med[static_cast<size_t>(n_blocks - 1)];
            } else {
                const double span = std::max(1e-9, blk_ctr[static_cast<size_t>(b + 1)] - blk_ctr[static_cast<size_t>(b)]);
                const double t = (x - blk_ctr[static_cast<size_t>(b)]) / span;
                env[static_cast<size_t>(i)] = blk_med[static_cast<size_t>(b)] +
                                              t * (blk_med[static_cast<size_t>(b + 1)] - blk_med[static_cast<size_t>(b)]);
            }
        }

        // Reference inner median over [nfft/4 .. 3*nfft/4] excluding hi_window
        std::vector<double> inner_vals;
        inner_vals.reserve(static_cast<size_t>(nfft / 2 + 1));
        for (int i = nfft / 4; i <= (3 * nfft) / 4 && i < nfft; ++i) {
            if (!hi_window[static_cast<size_t>(i)]) {
                inner_vals.push_back(env[static_cast<size_t>(i)]);
            }
        }
        if (inner_vals.empty()) {
            for (int i = min_edge; i < nfft - min_edge; ++i) {
                if (!hi_window[static_cast<size_t>(i)]) {
                    inner_vals.push_back(env[static_cast<size_t>(i)]);
                }
            }
        }

        if (!inner_vals.empty()) {
            const double inner_med = median_of_inplace(inner_vals);
            const int max_edge_cap = (nfft - std::max(16, nfft / 4)) / 2;
            const int max_edge = std::min(max_edge_cap, std::max(min_edge, static_cast<int>(static_cast<double>(nfft) * 0.22)));
            const int slope_step = std::max(block_sz, nfft / 64);

            // Independently scan inward on left edge
            while (edge_left < max_edge && !hi_window[static_cast<size_t>(edge_left)]) {
                const double cur = env[static_cast<size_t>(edge_left)];
                const double inward = env[static_cast<size_t>(std::min(nfft - 1, edge_left + slope_step))];
                const bool dropped_1db = (cur < inner_med - 1.0);
                const bool steep_rolloff = ((inward - cur) > 0.60) && (cur < inner_med - 0.35);
                if (dropped_1db || steep_rolloff) {
                    ++edge_left;
                } else {
                    break;
                }
            }

            // Independently scan inward on right edge
            while (edge_right < max_edge) {
                const int idx = nfft - 1 - edge_right;
                if (idx <= edge_left || hi_window[static_cast<size_t>(idx)]) break;
                const double cur = env[static_cast<size_t>(idx)];
                const double inward = env[static_cast<size_t>(std::max(0, idx - slope_step))];
                const bool dropped_1db = (cur < inner_med - 1.0);
                const bool steep_rolloff = ((inward - cur) > 0.60) && (cur < inner_med - 0.35);
                if (dropped_1db || steep_rolloff) {
                    ++edge_right;
                } else {
                    break;
                }
            }
        }
    }

    res.effective_edge_left_bins = edge_left;
    res.effective_edge_right_bins = edge_right;
    res.effective_passband_mask.assign(static_cast<size_t>(nfft), 1);
    for (int i = 0; i < edge_left && i < nfft; ++i) {
        res.effective_passband_mask[static_cast<size_t>(i)] = 0;
    }
    for (int i = std::max(0, nfft - edge_right); i < nfft; ++i) {
        res.effective_passband_mask[static_cast<size_t>(i)] = 0;
    }

    const auto& passband_mask = res.effective_passband_mask;

    // 4. Dispatch baseline solver
    if (config.baseline_method == BaselineMethod::Poly) {
        res.baseline_db = fit_poly(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::ALS) {
        res.baseline_db = fit_als(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::SavitzkyGolay) {
        res.baseline_db = fit_savgol(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::StandingWave) {
        res.baseline_db = fit_standing_wave(
            p_db, freqs_mhz, passband_mask, hi_window, config,
            &res.standing_wave_components_used,
            &res.standing_wave_period1_mhz,
            &res.standing_wave_period2_mhz
        );
    } else {
        res.baseline_db = fit_filter(p_db, freqs_mhz, passband_mask, hi_window, config);
    }

    // 5. Convert to linear power and eliminate Jensen log-median bias via 10%-trimmed linear ratio mean
    SimdDsp::db_to_power_f64(res.baseline_db.data(), res.baseline_power.data(), static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        if (!std::isfinite(res.baseline_power[static_cast<size_t>(i)]) ||
            res.baseline_power[static_cast<size_t>(i)] < 1e-30) {
            res.baseline_power[static_cast<size_t>(i)] = 1e-30;
        }
    }

    std::vector<double> lin_ratios;
    lin_ratios.reserve(static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (passband_mask[idx] && !hi_window[idx]) {
            const double r = power[idx] / res.baseline_power[idx];
            if (std::isfinite(r) && r > 0.0) {
                lin_ratios.push_back(r);
            }
        }
    }

    if (!lin_ratios.empty()) {
        std::sort(lin_ratios.begin(), lin_ratios.end());
        const size_t m_r = lin_ratios.size();
        const size_t trim = (m_r >= 10) ? (m_r / 10) : 0;
        const size_t lo = trim;
        const size_t hi = m_r - trim;
        double sum_r = 0.0;
        for (size_t k = lo; k < hi; ++k) sum_r += lin_ratios[k];
        double scale = sum_r / static_cast<double>(std::max<size_t>(1, hi - lo));
        if (!std::isfinite(scale) || scale <= 0.0) scale = 1.0;
        scale = std::clamp(scale, 0.5, 2.0);
        const double delta_db = 10.0 * std::log10(scale);

        for (int i = 0; i < nfft; ++i) {
            const size_t idx = static_cast<size_t>(i);
            res.baseline_power[idx] *= scale;
            res.baseline_db[idx] += delta_db;
        }
    }

    // 6. Populate off-line continuum flatness & quality diagnostics
    double sum_r2 = 0.0;
    int clean_cnt = 0;
    int clean_run = 0;
    std::vector<double> lag6_diffs;
    std::vector<double> lag1_diffs;
    lag6_diffs.reserve(static_cast<size_t>(nfft));
    lag1_diffs.reserve(static_cast<size_t>(nfft));

    double sum_x = 0.0, sum_y = 0.0;
    int pair_cnt = 0;
    std::vector<double> resid_db(static_cast<size_t>(nfft), 0.0);

    for (int i = 0; i < nfft; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (passband_mask[idx] && !hi_window[idx]) {
            const double r = p_db[idx] - res.baseline_db[idx];
            resid_db[idx] = r;
            sum_r2 += r * r;
            ++clean_cnt;
            ++clean_run;
            if (clean_run >= 2) {
                const double r_prev = resid_db[static_cast<size_t>(i - 1)];
                lag1_diffs.push_back(std::abs(r - r_prev));
                sum_x += r_prev;
                sum_y += r;
                ++pair_cnt;
            }
            if (clean_run >= 7) {
                lag6_diffs.push_back(std::abs(r - resid_db[static_cast<size_t>(i - 6)]));
            }
        } else {
            clean_run = 0;
        }
    }

    if (clean_cnt > 0) {
        res.off_line_rms_db = std::sqrt(sum_r2 / static_cast<double>(clean_cnt));
        constexpr double kMadToSigmaLag = 1.4826 / 1.4142135623730950488;
        if (!lag6_diffs.empty()) {
            res.thermal_rms_db = kMadToSigmaLag * median_of_inplace(lag6_diffs);
        } else if (!lag1_diffs.empty()) {
            res.thermal_rms_db = kMadToSigmaLag * median_of_inplace(lag1_diffs);
        } else {
            res.thermal_rms_db = res.off_line_rms_db;
        }

        if (res.thermal_rms_db < 1e-9 && res.off_line_rms_db < 1e-9) {
            res.baseline_quality_ratio = 1.0;
        } else {
            res.baseline_quality_ratio = res.off_line_rms_db / std::max(1e-6, res.thermal_rms_db);
        }

        if (pair_cnt >= 2) {
            const double mx = sum_x / static_cast<double>(pair_cnt);
            const double my = sum_y / static_cast<double>(pair_cnt);
            double num = 0.0, vx = 0.0, vy = 0.0;
            for (int i = 1; i < nfft; ++i) {
                const size_t idx = static_cast<size_t>(i);
                if (passband_mask[idx] && !hi_window[idx] &&
                    passband_mask[idx - 1] && !hi_window[idx - 1]) {
                    const double dx = resid_db[idx - 1] - mx;
                    const double dy = resid_db[idx] - my;
                    num += dx * dy;
                    vx += dx * dx;
                    vy += dy * dy;
                }
            }
            const double denom = std::sqrt(vx * vy);
            res.residual_lag1_autocorr = (denom > 1e-18) ? std::clamp(num / denom, -1.0, 1.0) : 0.0;
        }
    }

    return res;
}

// =========================================================================
// Stage 2: Local Line-Free Velocity Refinement (Multiplicative Huber IRLS)
// =========================================================================
void BaselineFitter::refine_local(
    const std::vector<double>& vel_kms,
    std::vector<double>& calibrated_k,
    std::vector<double>& calibrated_db,
    const AstroConfig& config,
    double& out_slope,
    double& out_intercept,
    const std::vector<uint8_t>* effective_passband_mask
) {
    out_slope = 0.0;
    out_intercept = 0.0;
    if (config.local_baseline_order < 0) return;

    const size_t n = vel_kms.size();
    if (n == 0 || n != calibrated_k.size() || n != calibrated_db.size()) return;

    const double hi_half_kms =
        (std::max(0.0, config.hi_protection_half_width_khz) * 1e-3 / HYDROGEN_REST_FREQ_MHZ) *
        SPEED_OF_LIGHT_KMS;
    const double v_inner = std::max(std::max(0.0, config.line_free_velocity_kms), hi_half_kms);
    const double v_outer = std::max(v_inner + 20.0, config.local_baseline_outer_kms);

    const bool has_mask = (effective_passband_mask != nullptr && effective_passband_mask->size() == n);
    const size_t fallback_edge = static_cast<size_t>(
        static_cast<double>(n) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)
    );

    std::vector<double> sel_u;
    std::vector<double> sel_y;
    sel_u.reserve(n / 2);
    sel_y.reserve(n / 2);

    const double inv_v_outer = 1.0 / v_outer;
    size_t neg_cnt = 0;
    size_t pos_cnt = 0;

    for (size_t i = 0; i < n; ++i) {
        const bool in_passband = has_mask
                                     ? ((*effective_passband_mask)[i] != 0)
                                     : (i >= fallback_edge && i + fallback_edge < n);
        if (!in_passband) continue;
        const double v = vel_kms[i];
        const double av = std::abs(v);
        if (av >= v_inner && av <= v_outer && std::isfinite(calibrated_k[i])) {
            sel_u.push_back(v * inv_v_outer);
            sel_y.push_back(calibrated_k[i]);
            if (v <= -v_inner) ++neg_cnt;
            if (v >= v_inner) ++pos_cnt;
        }
    }

    const size_t m_pts = sel_u.size();
    if (m_pts < 4) return;

    int order = std::clamp(config.local_baseline_order, 0, 2);
    if (order >= 1 && (neg_cnt < 10 || pos_cnt < 10)) {
        order = 0;
    }

    const int p = order + 1;
    std::vector<double> w(m_pts, 1.0);
    std::vector<double> res_scratch;
    res_scratch.reserve(m_pts);

    thread_local ChunkedHouseholderQR qr;
    double coeffs[3] = {0.0, 0.0, 0.0};

    // 3-iteration symmetric Huber IRLS in normalized velocity u = v / v_outer
    constexpr int kIters = 3;
    for (int iter = 0; iter < kIters; ++iter) {
        if (order == 0) {
            double sw = 0.0, swy = 0.0;
            for (size_t k = 0; k < m_pts; ++k) {
                sw += w[k];
                swy += w[k] * sel_y[k];
            }
            coeffs[0] = (sw > 1e-12) ? (swy / sw) : 0.0;
            coeffs[1] = 0.0;
            coeffs[2] = 0.0;
        } else {
            qr.reset(p);
            for (size_t k = 0; k < m_pts; ++k) {
                const double uk = sel_u[k];
                const double phi[3] = {1.0, uk, uk * uk};
                qr.add_row(phi, sel_y[k], std::sqrt(w[k]));
            }
            coeffs[0] = coeffs[1] = coeffs[2] = 0.0;
            qr.solve(coeffs);
        }

        if (iter + 1 >= kIters) break;

        res_scratch.clear();
        for (size_t k = 0; k < m_pts; ++k) {
            const double uk = sel_u[k];
            const double pred = coeffs[0] + coeffs[1] * uk + coeffs[2] * uk * uk;
            res_scratch.push_back(sel_y[k] - pred);
        }
        const double med_r = median_of_inplace(res_scratch);
        for (double& r : res_scratch) r = std::abs(r - med_r);
        const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(res_scratch));
        const double delta = 1.345 * mad_sigma;

        for (size_t k = 0; k < m_pts; ++k) {
            const double uk = sel_u[k];
            const double pred = coeffs[0] + coeffs[1] * uk + coeffs[2] * uk * uk;
            const double a = std::abs(sel_y[k] - pred);
            w[k] = (a <= delta) ? 1.0 : (delta / a);
        }
    }

    if (!std::isfinite(coeffs[0]) || !std::isfinite(coeffs[1]) || !std::isfinite(coeffs[2])) {
        return;
    }

    out_intercept = coeffs[0];
    out_slope = coeffs[1] * inv_v_outer;

    // Multiplicative gain refinement inside the local velocity band, with smooth raised-cosine
    // tapering outside v_outer so outer passband skirts never blow up.
    const double tsys = std::max(1.0, config.system_temp_k);
    const double inv_tsys = 1.0 / tsys;
    const double v_taper = std::max(40.0, 0.35 * v_outer);
    const double inv_v_taper = 1.0 / v_taper;

    thread_local std::vector<double> tl_ratio;
    tl_ratio.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const double v = vel_kms[i];
        const double av = std::abs(v);
        double local_k = 0.0;

        if (av <= v_outer) {
            const double u = v * inv_v_outer;
            local_k = coeffs[0] + coeffs[1] * u + coeffs[2] * u * u;
        } else if (av < v_outer + v_taper) {
            const double u_clamped = std::clamp(v * inv_v_outer, -1.0, 1.0);
            const double raw_k = coeffs[0] + coeffs[1] * u_clamped + coeffs[2] * u_clamped * u_clamped;
            const double t = (av - v_outer) * inv_v_taper;
            const double taper = 0.5 * (1.0 + std::cos(kPi * t));
            local_k = raw_k * taper;
        }

        const double denom = std::max(0.5, 1.0 + local_k * inv_tsys);
        const double ratio = std::max(1e-6, (1.0 + calibrated_k[i] * inv_tsys) / denom);
        calibrated_k[i] = tsys * (ratio - 1.0);
        tl_ratio[i] = ratio;
    }

    SimdDsp::power_to_db_f64(tl_ratio.data(), calibrated_db.data(), n, 1e-6);
}

} // namespace astro
