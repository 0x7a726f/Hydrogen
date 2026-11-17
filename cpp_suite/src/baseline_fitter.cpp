#include "baseline_fitter.h"
#include "simd_dsp.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace astro {

// Fast, stable, O(N) symmetric positive-definite pentadiagonal solver: A * x = rhs
// Factors A = L * D * L^T where L is unit lower triangular with bandwidth 2.
// Fully in-place solve in vector x, with persistent thread_local factorization buffers
// guaranteeing ZERO heap allocations in hot path.
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
        double d0 = (std::abs(diag[0]) < 1e-15) ? 1e-15 : diag[0];
        x[0] = rhs[0] / d0;
        return;
    }

    // Reusable thread-local factorization buffers: zero heap allocations in hot path
    thread_local std::vector<double> tl_D;
    thread_local std::vector<double> tl_invD;
    thread_local std::vector<double> tl_L1;
    thread_local std::vector<double> tl_L2;

    if (static_cast<int>(tl_D.size()) < n) {
        tl_D.resize(n);
        tl_invD.resize(n);
    }
    if (static_cast<int>(tl_L1.size()) < n - 1) tl_L1.resize(n - 1);
    if (static_cast<int>(tl_L2.size()) < n - 2) tl_L2.resize(n - 2);

    double* D = tl_D.data();
    double* invD = tl_invD.data();
    double* L1 = tl_L1.data();
    double* L2 = tl_L2.data();

    // Factorization: A = L * D * L^T (single reciprocal per pivot)
    for (int i = 0; i < n; ++i) {
        double d_val = diag[i];
        if (i >= 1) {
            d_val -= L1[i - 1] * L1[i - 1] * D[i - 1];
        }
        if (i >= 2) {
            d_val -= L2[i - 2] * L2[i - 2] * D[i - 2];
        }
        if (d_val < 1e-15) d_val = 1e-15;
        D[i] = d_val;
        const double inv_d = 1.0 / d_val;
        invD[i] = inv_d;

        if (i < n - 1) {
            double num1 = sub1[i];
            if (i >= 1) {
                num1 -= L2[i - 1] * L1[i - 1] * D[i - 1];
            }
            L1[i] = num1 * inv_d;
        }

        if (i < n - 2) {
            L2[i] = sub2[i] * inv_d;
        }
    }

    // Forward substitution in-place: L * y = rhs -> store y in x
    for (int i = 0; i < n; ++i) {
        double val = rhs[i];
        if (i >= 1) val -= L1[i - 1] * x[i - 1];
        if (i >= 2) val -= L2[i - 2] * x[i - 2];
        x[i] = val;
    }

    // Diagonal solve in-place using cached reciprocal: z = y * invD
    for (int i = 0; i < n; ++i) {
        x[i] *= invD[i];
    }

    // Back substitution in-place: L^T * x = z -> update x backwards
    for (int i = n - 1; i >= 0; --i) {
        double val = x[i];
        if (i < n - 1) val -= L1[i] * x[i + 1];
        if (i < n - 2) val -= L2[i] * x[i + 2];
        x[i] = val;
    }
}

static double median_of_inplace(std::vector<double>& v) {
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

        constexpr double kTwoPi = 6.283185307179586476925286766559;
        for (int s = 4; s < nfft; s <<= 1) {
            st_s.push_back(s);
            const double base_ang = kTwoPi / static_cast<double>(2 * s);
            for (int j = 0; j < s; ++j) {
                double ang = base_ang * static_cast<double>(j);
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

        // Stage 1: Radix-4 Stockham DIT (M >= 16 for n >= 64 -> unit-stride 4x AVX2)
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
                double x0r = in_re[0 * M + k], x0i = in_im[0 * M + k];
                double x1r = in_re[1 * M + k], x1i = in_im[1 * M + k];
                double x2r = in_re[2 * M + k], x2i = in_im[2 * M + k];
                double x3r = in_re[3 * M + k], x3i = in_im[3 * M + k];
                double t0r = x0r + x2r, t0i = x0i + x2i;
                double t1r = x0r - x2r, t1i = x0i - x2i;
                double t2r = x1r + x3r, t2i = x1i + x3i;
                double t3r = x1r - x3r, t3i = x1i - x3i;
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
                    double ar = cur_r[in_off + j], ai = cur_i[in_off + j];
                    double br0 = cur_r[in_off + half_n + j], bi0 = cur_i[in_off + half_n + j];
                    double wr = wr_ptr[j], wi = wi_ptr[j];
                    double br = inverse ? (br0 * wr - bi0 * wi) : (br0 * wr + bi0 * wi);
                    double bi = inverse ? (br0 * wi + bi0 * wr) : (bi0 * wr - br0 * wi);
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

// Symmetric AVX2+FMA 1D Gaussian smoothing with edge replication (O(N log N) FFT for radius > 24)
static std::vector<double> gaussian_smooth_1d(const std::vector<double>& input, double sigma) {
    if (sigma <= 0.0 || input.size() < 3) return input;
    const int n = static_cast<int>(input.size());
    std::vector<double> output(n);

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    thread_local std::vector<double> tl_half_kernel;
    tl_half_kernel.resize(static_cast<size_t>(radius) + 1);

    const double inv_2sig2 = -0.5 / (sigma * sigma);
    double sum = 1.0; // k = 0 weight is 1.0
    tl_half_kernel[0] = 1.0;
    for (int k = 1; k <= radius; ++k) {
        double g = std::exp(inv_2sig2 * static_cast<double>(k) * static_cast<double>(k));
        tl_half_kernel[k] = g;
        sum += 2.0 * g;
    }
    const double inv_sum = 1.0 / sum;
    for (int k = 0; k <= radius; ++k) tl_half_kernel[k] *= inv_sum;

    const double first_v = input.front();
    const double last_v = input.back();

    // O(N log N) AVX2+FMA double FFT convolution for wide kernels
    if (radius > 24) {
        const int L = n + 2 * radius;
        int nfft = 64;
        while (nfft < L) nfft <<= 1;

        thread_local DoubleFftPlan plan;
        thread_local std::vector<double> k_spec_re, k_spec_im;
        thread_local std::vector<double> sig_re, sig_im, s_re, s_im, out_re, out_im;
        thread_local int cached_nfft = 0, cached_radius = 0;
        thread_local double cached_sigma = -1.0;

        plan.prepare(nfft);
        sig_re.resize(static_cast<size_t>(nfft));
        sig_im.resize(static_cast<size_t>(nfft));
        s_re.resize(static_cast<size_t>(nfft));
        s_im.resize(static_cast<size_t>(nfft));
        out_re.resize(static_cast<size_t>(nfft));
        out_im.resize(static_cast<size_t>(nfft));

        if (cached_nfft != nfft || cached_radius != radius || cached_sigma != sigma) {
            cached_nfft = nfft;
            cached_radius = radius;
            cached_sigma = sigma;
            k_spec_re.resize(static_cast<size_t>(nfft));
            k_spec_im.resize(static_cast<size_t>(nfft));

            const double inv_nfft = 1.0 / static_cast<double>(nfft);
            std::fill(sig_re.begin(), sig_re.end(), 0.0);
            std::fill(sig_im.begin(), sig_im.end(), 0.0);
            sig_re[radius] = tl_half_kernel[0] * inv_nfft;
            for (int m = 1; m <= radius; ++m) {
                double w = tl_half_kernel[m] * inv_nfft;
                sig_re[radius - m] = w;
                sig_re[radius + m] = w;
            }
            plan.run(sig_re.data(), sig_im.data(), k_spec_re.data(), k_spec_im.data(), false);
        }

        // Edge-replicated signal padded with zeros to nfft
        std::fill_n(sig_re.data(), radius, first_v);
        std::memcpy(sig_re.data() + radius, input.data(), static_cast<size_t>(n) * sizeof(double));
        std::fill_n(sig_re.data() + radius + n, radius, last_v);
        std::fill_n(sig_re.data() + L, nfft - L, 0.0);
        std::fill_n(sig_im.data(), nfft, 0.0);

        plan.run(sig_re.data(), sig_im.data(), s_re.data(), s_im.data(), false);

        // Pointwise complex multiply: sig = S * K_spec
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
            double ar = s_re[idx], ai = s_im[idx];
            double br = k_spec_re[idx], bi = k_spec_im[idx];
            sig_re[idx] = ar * br - ai * bi;
            sig_im[idx] = ar * bi + ai * br;
        }

        plan.run(sig_re.data(), sig_im.data(), out_re.data(), out_im.data(), true);
        std::memcpy(output.data(), out_re.data() + 2 * radius, static_cast<size_t>(n) * sizeof(double));
        return output;
    }

    // Direct symmetric AVX2+FMA convolution for small kernels (radius <= 24)
    thread_local std::vector<double> tl_padded;
    const size_t pad_len = static_cast<size_t>(n) + 2ULL * static_cast<size_t>(radius);
    if (tl_padded.size() < pad_len) tl_padded.resize(pad_len);
    double* p = tl_padded.data();
    std::fill_n(p, radius, first_v);
    std::memcpy(p + radius, input.data(), static_cast<size_t>(n) * sizeof(double));
    std::fill_n(p + radius + n, radius, last_v);

    const double* hk = tl_half_kernel.data();
    const double* center_base = p + radius;
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

std::vector<double> BaselineFitter::fit_als(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    (void)freqs_mhz;
    const int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    // Solve ALS on at most 256 macro-bins so als_lambda (tuned for ~256 points) maintains a
    // consistent fractional passband smoothing scale and well-conditioned LDL^T factorization
    // across large FFT sizes (e.g. 8192..65536).
    const int m = (n <= 256) ? n : 256;
    std::vector<double> m_y(m, 0.0);
    std::vector<double> m_center(m, 0.0);
    std::vector<uint8_t> m_pass(m, 1);
    std::vector<uint8_t> m_hi(m, 0);

    if (m == n) {
        for (int i = 0; i < n; ++i) {
            m_y[i] = p_db[i];
            m_center[i] = static_cast<double>(i);
            m_pass[i] = passband_mask[i];
            m_hi[i] = hi_window[i];
        }
    } else {
        for (int b = 0; b < m; ++b) {
            const int i0 = static_cast<int>((static_cast<int64_t>(b) * n) / m);
            const int i1 = static_cast<int>((static_cast<int64_t>(b + 1) * n) / m);
            m_center[b] = 0.5 * static_cast<double>(i0 + i1 - 1);

            double sum_clean = 0.0;
            int cnt_clean = 0;
            double sum_pass = 0.0;
            int cnt_pass = 0;
            double sum_all = 0.0;
            int cnt_all = 0;
            int cnt_hi = 0;
            int cnt_out_pass = 0;

            for (int i = i0; i < i1; ++i) {
                sum_all += p_db[i];
                ++cnt_all;
                if (passband_mask[i]) {
                    sum_pass += p_db[i];
                    ++cnt_pass;
                    if (!hi_window[i]) {
                        sum_clean += p_db[i];
                        ++cnt_clean;
                    }
                } else {
                    ++cnt_out_pass;
                }
                if (hi_window[i]) ++cnt_hi;
            }

            if (cnt_clean > 0) {
                m_y[b] = sum_clean / static_cast<double>(cnt_clean);
                m_pass[b] = 1;
                m_hi[b] = (cnt_hi > cnt_all / 2) ? 1 : 0;
            } else if (cnt_pass > 0) {
                // Entire macro-bin is inside H I window
                m_y[b] = sum_pass / static_cast<double>(cnt_pass);
                m_pass[b] = 1;
                m_hi[b] = 1;
            } else {
                // Outer skirt macro-bin
                m_y[b] = (cnt_all > 0) ? (sum_all / static_cast<double>(cnt_all)) : 0.0;
                m_pass[b] = (cnt_out_pass > 0) ? 0 : 1;
                m_hi[b] = (cnt_hi > 0) ? 1 : 0;
            }
        }
    }

    thread_local std::vector<double> w, s_diag, A_diag, A_sub1, A_sub2, rhs, z, clean_residuals;
    w.resize(m);
    s_diag.resize(m);
    A_diag.resize(m);
    A_sub1.resize(m - 1);
    A_sub2.resize(m - 2);
    rhs.resize(m);
    z.assign(m, 0.0);

    for (int i = 0; i < m; ++i) {
        double wi = 1.0;
        if (!m_pass[i]) wi = 0.02;
        if (m_hi[i]) wi = 1e-9;
        w[i] = wi;
    }

    const double lam = std::max(1.0, config.als_lambda);
    const double p = std::clamp(config.als_p, 1e-4, 0.9999);

    // Second difference operator D has D^T D entries:
    // S_diag: [1, 5, 6, 6, ..., 6, 5, 1]
    // S_sub1: [-2, -4, -4, ..., -4, -2]
    // S_sub2: [1, 1, ..., 1]
    std::fill_n(s_diag.data(), m, 6.0);
    s_diag[0] = 1.0;
    s_diag[1] = 5.0;
    s_diag[m - 2] = 5.0;
    s_diag[m - 1] = 1.0;

    std::fill_n(A_sub1.data(), m - 1, lam * -4.0);
    A_sub1[0] = lam * -2.0;
    A_sub1[m - 2] = lam * -2.0;
    std::fill_n(A_sub2.data(), m - 2, lam);

    const int iterations = 8;
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < m; ++i) {
            A_diag[i] = w[i] + lam * s_diag[i];
            rhs[i] = w[i] * m_y[i];
        }

        solve_pentadiagonal_symmetric(m, A_diag.data(), A_sub1.data(), A_sub2.data(), rhs.data(), z.data());

        for (int i = 0; i < m; ++i) {
            double wi = (m_y[i] > z[i]) ? p : (1.0 - p);
            if (!m_pass[i]) wi = 0.02;
            if (m_hi[i]) wi = 1e-9;
            w[i] = wi;
        }
    }

    std::vector<double> baseline_db(n);
    if (m == n) {
        for (int i = 0; i < n; ++i) baseline_db[i] = z[i];
    } else {
        int b = 0;
        for (int i = 0; i < n; ++i) {
            const double x = static_cast<double>(i);
            while (b + 1 < m - 1 && x > m_center[b + 1]) ++b;
            if (x <= m_center[0]) {
                baseline_db[i] = z[0];
            } else if (x >= m_center[m - 1]) {
                baseline_db[i] = z[m - 1];
            } else {
                const double span = std::max(1e-9, m_center[b + 1] - m_center[b]);
                const double t = (x - m_center[b]) / span;
                baseline_db[i] = z[b] + t * (z[b + 1] - z[b]);
            }
        }
    }

    // Center baseline on the median of clean passband noise residuals at full resolution
    clean_residuals.clear();
    if (clean_residuals.capacity() < static_cast<size_t>(n)) clean_residuals.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            clean_residuals.push_back(p_db[i] - baseline_db[i]);
        }
    }

    double noise_offset = median_of_inplace(clean_residuals);
    for (int i = 0; i < n; ++i) {
        baseline_db[i] += noise_offset;
    }
    return baseline_db;
}

// Polynomial fitting via Hankel normal equations + adaptive Tikhonov regularization
std::vector<double> BaselineFitter::fit_poly(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    int n = static_cast<int>(p_db.size());
    if (n < 4) return p_db;

    int deg = std::clamp(config.poly_degree, 2, 14);
    int m = deg + 1; // number of coefficients (<= 15)

    // Normalize frequencies on passband
    double mean_f = 0.0;
    double std_f = 0.0;
    int count_pb = 0;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i]) {
            mean_f += freqs_mhz[i];
            count_pb++;
        }
    }
    if (count_pb < m) return p_db;
    mean_f /= count_pb;

    for (int i = 0; i < n; ++i) {
        if (passband_mask[i]) {
            double df = freqs_mhz[i] - mean_f;
            std_f += df * df;
        }
    }
    std_f = (count_pb > 1) ? std::sqrt(std_f / (count_pb - 1)) : 1.0;
    if (std_f < 1e-6) std_f = 1.0;
    const double inv_std_f = 1.0 / std_f;

    thread_local std::vector<double> x_norm, weights, tl_res, scratch_res;
    x_norm.resize(n);
    weights.resize(n);
    tl_res.resize(n);
    for (int i = 0; i < n; ++i) {
        x_norm[i] = (freqs_mhz[i] - mean_f) * inv_std_f;
        weights[i] = (passband_mask[i] && !hi_window[i]) ? 1.0 : 0.0;
    }

    double coeffs[16] = {0.0};
    if (scratch_res.capacity() < static_cast<size_t>(n)) scratch_res.reserve(n);

    const int max_mom = 2 * deg; // <= 28
    for (int iter = 0; iter < 6; ++iter) {
        double mom[32] = {0.0};
        double XTWy[16] = {0.0};
        double XTWX_orig[16 * 16] = {0.0};

        int i = 0;
#if defined(__AVX2__)
        __m256d vmom[32];
        __m256d vrhs[16];
        for (int k = 0; k <= max_mom; ++k) vmom[k] = _mm256_setzero_pd();
        for (int r = 0; r < m; ++r) vrhs[r] = _mm256_setzero_pd();

        for (; i + 4 <= n; i += 4) {
            __m256d px = _mm256_loadu_pd(weights.data() + i);
            __m256d xi = _mm256_loadu_pd(x_norm.data() + i);
            __m256d yi = _mm256_loadu_pd(p_db.data() + i);
            vmom[0] = _mm256_add_pd(vmom[0], px);
#if SIMD_HAS_FMA
            vrhs[0] = _mm256_fmadd_pd(px, yi, vrhs[0]);
#else
            vrhs[0] = _mm256_add_pd(vrhs[0], _mm256_mul_pd(px, yi));
#endif
            for (int k = 1; k < m; ++k) {
                px = _mm256_mul_pd(px, xi);
                vmom[k] = _mm256_add_pd(vmom[k], px);
#if SIMD_HAS_FMA
                vrhs[k] = _mm256_fmadd_pd(px, yi, vrhs[k]);
#else
                vrhs[k] = _mm256_add_pd(vrhs[k], _mm256_mul_pd(px, yi));
#endif
            }
            for (int k = m; k <= max_mom; ++k) {
                px = _mm256_mul_pd(px, xi);
                vmom[k] = _mm256_add_pd(vmom[k], px);
            }
        }
        auto hsum_pd = [](__m256d v) {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, v);
            return tmp[0] + tmp[1] + tmp[2] + tmp[3];
        };
        for (int k = 0; k <= max_mom; ++k) mom[k] = hsum_pd(vmom[k]);
        for (int r = 0; r < m; ++r) XTWy[r] = hsum_pd(vrhs[r]);
#endif
        for (; i < n; ++i) {
            double px = weights[i];
            double xi = x_norm[i];
            double yi = p_db[i];
            mom[0] += px;
            XTWy[0] += px * yi;
            for (int k = 1; k < m; ++k) {
                px *= xi;
                mom[k] += px;
                XTWy[k] += px * yi;
            }
            for (int k = m; k <= max_mom; ++k) {
                px *= xi;
                mom[k] += px;
            }
        }

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                XTWX_orig[r * m + c] = mom[r + c];
            }
        }

        // Adaptive Cholesky solve with ridge damping
        bool cholesky_ok = false;
        double lambda = 1e-6;
        double A[16 * 16];
        double L[16 * 16];

        for (int attempt = 0; attempt < 5 && !cholesky_ok; ++attempt) {
            std::memcpy(A, XTWX_orig, static_cast<size_t>(m * m) * sizeof(double));
            for (int r = 0; r < m; ++r) {
                A[r * m + r] += lambda * (1.0 + XTWX_orig[r * m + r]);
            }

            std::memset(L, 0, static_cast<size_t>(m * m) * sizeof(double));
            cholesky_ok = true;
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col <= row; ++col) {
                    double sum = 0.0;
                    for (int k = 0; k < col; ++k) sum += L[row * m + k] * L[col * m + k];
                    if (row == col) {
                        double val = A[row * m + row] - sum;
                        if (val <= 1e-12) { cholesky_ok = false; break; }
                        L[row * m + col] = std::sqrt(val);
                    } else {
                        L[row * m + col] = (A[row * m + col] - sum) / L[col * m + col];
                    }
                }
                if (!cholesky_ok) break;
            }
            if (!cholesky_ok) lambda *= 10.0;
        }

        if (cholesky_ok) {
            double y_sol[16];
            for (int row = 0; row < m; ++row) {
                double sum = 0.0;
                for (int k = 0; k < row; ++k) sum += L[row * m + k] * y_sol[k];
                y_sol[row] = (XTWy[row] - sum) / L[row * m + row];
            }
            for (int row = m - 1; row >= 0; --row) {
                double sum = 0.0;
                for (int k = row + 1; k < m; ++k) sum += L[k * m + row] * coeffs[k];
                coeffs[row] = (y_sol[row] - sum) / L[row * m + row];
            }
        }

        // Vectorized Horner residual evaluation across full grid
        int eval_i = 0;
#if defined(__AVX2__)
        const __m256d c_top = _mm256_set1_pd(coeffs[m - 1]);
        for (; eval_i + 4 <= n; eval_i += 4) {
            __m256d xi = _mm256_loadu_pd(x_norm.data() + eval_i);
            __m256d p_eval = c_top;
            for (int k = m - 2; k >= 0; --k) {
                __m256d ck = _mm256_set1_pd(coeffs[k]);
#if SIMD_HAS_FMA
                p_eval = _mm256_fmadd_pd(p_eval, xi, ck);
#else
                p_eval = _mm256_add_pd(_mm256_mul_pd(p_eval, xi), ck);
#endif
            }
            __m256d yi = _mm256_loadu_pd(p_db.data() + eval_i);
            _mm256_storeu_pd(tl_res.data() + eval_i, _mm256_sub_pd(yi, p_eval));
        }
#endif
        for (; eval_i < n; ++eval_i) {
            double p_eval = coeffs[m - 1];
            double xi = x_norm[eval_i];
            for (int k = m - 2; k >= 0; --k) p_eval = p_eval * xi + coeffs[k];
            tl_res[eval_i] = p_db[eval_i] - p_eval;
        }

        scratch_res.clear();
        for (int idx = 0; idx < n; ++idx) {
            if (passband_mask[idx] && !hi_window[idx]) {
                scratch_res.push_back(tl_res[idx]);
            }
        }
        const size_t active_cnt = scratch_res.size();
        double med_r = median_of_inplace(scratch_res);
        double sum_res_sq = 0.0;
        for (double r : scratch_res) sum_res_sq += (r - med_r) * (r - med_r);
        double std_r = (active_cnt > 0) ? std::sqrt(sum_res_sq / active_cnt) : 0.05;
        if (std_r < 1e-4) std_r = 0.05;

        const double hi_thresh = 1.5 * std_r;
        const double lo_thresh = -3.0 * std_r;
        for (int idx = 0; idx < n; ++idx) {
            if (passband_mask[idx] && !hi_window[idx]) {
                double diff = tl_res[idx] - med_r;
                weights[idx] = (diff < hi_thresh && diff > lo_thresh) ? 1.0 : 0.0;
            }
        }
    }

    std::vector<double> baseline_db(n);
    int out_i = 0;
#if defined(__AVX2__)
    const __m256d c_top = _mm256_set1_pd(coeffs[m - 1]);
    for (; out_i + 4 <= n; out_i += 4) {
        __m256d xi = _mm256_loadu_pd(x_norm.data() + out_i);
        __m256d p_eval = c_top;
        for (int k = m - 2; k >= 0; --k) {
            __m256d ck = _mm256_set1_pd(coeffs[k]);
#if SIMD_HAS_FMA
            p_eval = _mm256_fmadd_pd(p_eval, xi, ck);
#else
            p_eval = _mm256_add_pd(_mm256_mul_pd(p_eval, xi), ck);
#endif
        }
        _mm256_storeu_pd(baseline_db.data() + out_i, p_eval);
    }
#endif
    for (; out_i < n; ++out_i) {
        double p_eval = coeffs[m - 1];
        double xi = x_norm[out_i];
        for (int k = m - 2; k >= 0; --k) p_eval = p_eval * xi + coeffs[k];
        baseline_db[out_i] = p_eval;
    }

    scratch_res.clear();
    for (int idx = 0; idx < n; ++idx) {
        if (passband_mask[idx] && !hi_window[idx]) {
            scratch_res.push_back(p_db[idx] - baseline_db[idx]);
        }
    }

    double offset = median_of_inplace(scratch_res);
#if defined(__AVX2__)
    const __m256d v_off = _mm256_set1_pd(offset);
    int off_i = 0;
    for (; off_i + 4 <= n; off_i += 4) {
        _mm256_storeu_pd(baseline_db.data() + off_i,
                         _mm256_add_pd(_mm256_loadu_pd(baseline_db.data() + off_i), v_off));
    }
    for (; off_i < n; ++off_i) baseline_db[off_i] += offset;
#else
    for (int idx = 0; idx < n; ++idx) baseline_db[idx] += offset;
#endif
    return baseline_db;
}

// Ripple filter: Zero-copy gap interpolation across HI window + Gaussian smoothing
std::vector<double> BaselineFitter::fit_filter(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    (void)freqs_mhz;
    int n = static_cast<int>(p_db.size());
    if (n <= 0) return {};

    int first_valid = -1;
    int last_valid = -1;
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            if (first_valid < 0) first_valid = i;
            last_valid = i;
            valid_count++;
        }
    }
    if (valid_count < 2) return p_db;

    auto local_valid_median = [&](int center, int half_span) {
        std::vector<double> vals;
        vals.reserve(static_cast<size_t>(2 * half_span + 1));
        const int lo = std::max(0, center - half_span);
        const int hi = std::min(n - 1, center + half_span);
        for (int idx = lo; idx <= hi; ++idx) {
            if (passband_mask[idx] && !hi_window[idx]) {
                vals.push_back(p_db[idx]);
            }
        }
        if (vals.empty()) return p_db[center];
        return median_of_inplace(vals);
    };

    const int anchor_span = std::clamp(n / 128, 3, 24);
    thread_local std::vector<double> interpolated;
    interpolated.resize(n);
    double* dst = interpolated.data();

    const double left_anchor = local_valid_median(first_valid, anchor_span);
    const double right_anchor = local_valid_median(last_valid, anchor_span);
    std::fill_n(dst, first_valid, left_anchor);
    dst[first_valid] = p_db[first_valid];

    int prev_i = first_valid;
    for (int i = first_valid + 1; i <= last_valid; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            int gap = i - prev_i;
            if (gap == 1) {
                dst[i] = p_db[i];
            } else {
                const double y0 = local_valid_median(prev_i, anchor_span);
                const double y1 = local_valid_median(i, anchor_span);
                const double step = (y1 - y0) / static_cast<double>(gap);
                for (int j = 1; j < gap; ++j) {
                    dst[prev_i + j] = y0 + static_cast<double>(j) * step;
                }
                dst[i] = p_db[i];
            }
            prev_i = i;
        }
    }
    // Right edge clamp
    if (last_valid + 1 < n) {
        std::fill_n(dst + last_valid + 1, n - (last_valid + 1), right_anchor);
    }

    double bin_res_hz = config.sample_rate_hz / static_cast<double>(n);
    double sigma_bins = std::max(3.0, (config.filter_smoothing_sigma_khz * 1e3) / bin_res_hz);
    std::vector<double> smoothed = gaussian_smooth_1d(interpolated, sigma_bins);

    thread_local std::vector<double> clean_res;
    clean_res.clear();
    if (clean_res.capacity() < static_cast<size_t>(n)) clean_res.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            clean_res.push_back(p_db[i] - smoothed[i]);
        }
    }
    double offset = median_of_inplace(clean_res);
    for (int i = 0; i < n; ++i) smoothed[i] += offset;
    return smoothed;
}

// Moment-preserving degree-2 Savitzky-Golay passband baseline filter
// Exact analytical weights: h[k] = 3 * (3*H^2 + 3*H - 1 - 5*k^2) / ((2*H - 1)*(2*H + 1)*(2*H + 3))
// Preserves 0th, 1st, and 2nd passband moments (quadratic curvature) with zero attenuation.
std::vector<double> BaselineFitter::fit_savgol(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    (void)freqs_mhz;
    const int n = static_cast<int>(p_db.size());
    if (n < 8) return p_db;

    int first_valid = -1;
    int last_valid = -1;
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            if (first_valid < 0) first_valid = i;
            last_valid = i;
            ++valid_count;
        }
    }
    if (valid_count < 4) return p_db;

    // Anchor gap interpolation using robust local medians around gap boundaries so single boundary
    // noise spikes do not tilt the H I window bridge.
    auto local_valid_median = [&](int center, int half_span) {
        std::vector<double> vals;
        vals.reserve(static_cast<size_t>(2 * half_span + 1));
        const int lo = std::max(0, center - half_span);
        const int hi = std::min(n - 1, center + half_span);
        for (int idx = lo; idx <= hi; ++idx) {
            if (passband_mask[idx] && !hi_window[idx]) {
                vals.push_back(p_db[idx]);
            }
        }
        if (vals.empty()) return p_db[center];
        return median_of_inplace(vals);
    };

    const int anchor_span = std::clamp(n / 128, 3, 24);
    std::vector<double> work(n);
    const double left_anchor = local_valid_median(first_valid, anchor_span);
    const double right_anchor = local_valid_median(last_valid, anchor_span);
    std::fill_n(work.data(), first_valid + 1, left_anchor);

    int prev_i = first_valid;
    for (int i = first_valid + 1; i <= last_valid; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            const int gap = i - prev_i;
            if (gap == 1) {
                work[i] = p_db[i];
            } else {
                const double y0 = local_valid_median(prev_i, anchor_span);
                const double y1 = local_valid_median(i, anchor_span);
                const double step = (y1 - y0) / static_cast<double>(gap);
                for (int j = 0; j <= gap; ++j) {
                    work[prev_i + j] = y0 + static_cast<double>(j) * step;
                }
            }
            prev_i = i;
        }
    }
    if (last_valid + 1 < n) {
        std::fill_n(work.data() + last_valid + 1, n - (last_valid + 1), right_anchor);
    }

    const double bin_res_hz = config.sample_rate_hz / static_cast<double>(n);
    const double sigma_bins = std::max(3.0, (config.filter_smoothing_sigma_khz * 1e3) / bin_res_hz);
    const int H = std::clamp(static_cast<int>(std::round(1.8 * sigma_bins)), 5, std::max(5, n / 6));

    const double hd = static_cast<double>(H);
    const double denom = (2.0 * hd - 1.0) * (2.0 * hd + 1.0) * (2.0 * hd + 3.0);
    std::vector<double> hk(static_cast<size_t>(H) + 1);
    for (int k = 0; k <= H; ++k) {
        const double kd = static_cast<double>(k);
        hk[static_cast<size_t>(k)] = 3.0 * (3.0 * hd * hd + 3.0 * hd - 1.0 - 5.0 * kd * kd) / denom;
    }

    auto apply_sg = [&](const std::vector<double>& in_vec) {
        std::vector<double> out_vec(n);
        const double fv = in_vec.front();
        const double lv = in_vec.back();
        for (int i = 0; i < n; ++i) {
            double acc = hk[0] * in_vec[static_cast<size_t>(i)];
            for (int k = 1; k <= H; ++k) {
                const double vl = (i - k >= 0) ? in_vec[static_cast<size_t>(i - k)] : fv;
                const double vr = (i + k < n) ? in_vec[static_cast<size_t>(i + k)] : lv;
                acc += hk[static_cast<size_t>(k)] * (vl + vr);
            }
            out_vec[static_cast<size_t>(i)] = acc;
        }
        return out_vec;
    };

    // Pass 1 Savitzky-Golay + outlier clipping on clean passband
    std::vector<double> smoothed = apply_sg(work);
    std::vector<double> clean_res;
    clean_res.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            clean_res.push_back(p_db[i] - smoothed[i]);
        }
    }
    const double med_r = median_of_inplace(clean_res);
    for (double& r : clean_res) r = std::abs(r - med_r);
    const double mad_sigma = std::max(1e-4, 1.4826 * median_of_inplace(clean_res));
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            if (std::abs((p_db[i] - smoothed[i]) - med_r) > 2.2 * mad_sigma) {
                work[i] = smoothed[i] + med_r;
            }
        } else {
            work[i] = smoothed[i] + med_r;
        }
    }

    // Pass 2 Savitzky-Golay on outlier-cleaned continuum
    smoothed = apply_sg(work);

    clean_res.clear();
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            clean_res.push_back(p_db[i] - smoothed[i]);
        }
    }
    const double offset = median_of_inplace(clean_res);
    for (int i = 0; i < n; ++i) smoothed[i] += offset;
    return smoothed;
}

// Coaxial Standing-Wave Reflection Sinusoid + Continuum Solver:
// 1. Estimates broadband continuum B0(f) via smooth gap-interpolated filter.
// 2. Transforms line-free passband residuals into the cable-delay (lag) domain via DoubleFftPlan.
// 3. Identifies the top K <= 2 standing-wave ripple frequencies and solves their sinusoidal
//    amplitudes (cos/sin) strictly on line-free channels (!hi_window) via linear least squares,
//    analytically bridging across the 21-cm H I window with zero line absorption.
std::vector<double> BaselineFitter::fit_standing_wave(
    const std::vector<double>& p_db,
    const std::vector<double>& freqs_mhz,
    const std::vector<uint8_t>& passband_mask,
    const std::vector<uint8_t>& hi_window,
    const AstroConfig& config
) {
    const int n = static_cast<int>(p_db.size());
    if (n < 16) return p_db;

    // 1. Broad continuum baseline B0(f) via degree-4 Chebyshev polynomial on line-free channels
    AstroConfig poly_cfg = config;
    poly_cfg.poly_degree = std::clamp(config.poly_degree, 3, 4);
    std::vector<double> base_db = fit_poly(p_db, freqs_mhz, passband_mask, hi_window, poly_cfg);

    // 2. 4x zero-padded delay-domain FFT of line-free residuals to detect dominant standing-wave frequencies
    int n_pow2 = 64;
    while (n_pow2 < n) n_pow2 <<= 1;
    const int nfft_sw = n_pow2 * 4;

    thread_local DoubleFftPlan sw_plan;
    sw_plan.prepare(nfft_sw);
    std::vector<double> in_re(static_cast<size_t>(nfft_sw), 0.0);
    std::vector<double> in_im(static_cast<size_t>(nfft_sw), 0.0);
    std::vector<double> out_re(static_cast<size_t>(nfft_sw), 0.0);
    std::vector<double> out_im(static_cast<size_t>(nfft_sw), 0.0);

    int valid_cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            const double w_hann = 0.5 * (1.0 - std::cos(6.28318530717958647692 * static_cast<double>(i) / static_cast<double>(std::max(1, n - 1))));
            in_re[static_cast<size_t>(i)] = (p_db[i] - base_db[i]) * w_hann;
            ++valid_cnt;
        }
    }
    if (valid_cnt < 16) return base_db;

    sw_plan.run(in_re.data(), in_im.data(), out_re.data(), out_im.data(), false);

    // Search fundamental ripple cycles 2 .. min(n/8, 96) in 4x oversampled bins
    const int min_bin = 8;
    const int max_bin = std::min(nfft_sw / 8, 96 * 4);
    if (max_bin <= min_bin + 4) return base_db;

    std::vector<double> mag_sq(static_cast<size_t>(max_bin + 1), 0.0);
    for (int m = min_bin; m <= max_bin; ++m) {
        const double r = out_re[static_cast<size_t>(m)];
        const double im = out_im[static_cast<size_t>(m)];
        mag_sq[static_cast<size_t>(m)] = r * r + im * im;
    }

    // Pick top 2 well-separated delay peaks with sub-bin parabolic frequency refinement
    int m1 = min_bin;
    for (int m = min_bin + 1; m <= max_bin; ++m) {
        if (mag_sq[static_cast<size_t>(m)] > mag_sq[static_cast<size_t>(m1)]) m1 = m;
    }
    int m2 = -1;
    double best_m2 = -1.0;
    for (int m = min_bin; m <= max_bin; ++m) {
        if (std::abs(m - m1) <= 8) continue;
        if (mag_sq[static_cast<size_t>(m)] > best_m2) {
            best_m2 = mag_sq[static_cast<size_t>(m)];
            m2 = m;
        }
    }

    auto refine_freq = [&](int m_bin) {
        double delta = 0.0;
        if (m_bin > min_bin && m_bin < max_bin) {
            const double y_l = std::sqrt(mag_sq[static_cast<size_t>(m_bin - 1)]);
            const double y_0 = std::sqrt(mag_sq[static_cast<size_t>(m_bin)]);
            const double y_r = std::sqrt(mag_sq[static_cast<size_t>(m_bin + 1)]);
            const double den = y_l - 2.0 * y_0 + y_r;
            if (std::abs(den) > 1e-12) {
                delta = std::clamp(0.5 * (y_l - y_r) / den, -0.5, 0.5);
            }
        }
        return 6.28318530717958647692 * (static_cast<double>(m_bin) + delta) / static_cast<double>(nfft_sw);
    };

    const double w1 = refine_freq(m1);
    const double w2 = (m2 > 0) ? refine_freq(m2) : (2.0 * w1);

    // 3. Solve joint 8x8 normal equations on line-free channels:
    //    [1, u, u^2, u^3, cos(w1*i), sin(w1*i), cos(w2*i), sin(w2*i)]
    constexpr int K = 8;
    double ATA[K][K] = {};
    double ATb[K] = {};
    const double inv_half_n = 2.0 / static_cast<double>(std::max(1, n - 1));
    for (int i = 0; i < n; ++i) {
        if (!passband_mask[i] || hi_window[i]) continue;
        const double x = static_cast<double>(i);
        const double u = x * inv_half_n - 1.0;
        const double u2 = u * u;
        const double phi[K] = {
            1.0,
            u,
            u2,
            u2 * u,
            std::cos(w1 * x),
            std::sin(w1 * x),
            std::cos(w2 * x),
            std::sin(w2 * x)
        };
        const double y = p_db[i] - base_db[i];
        for (int r = 0; r < K; ++r) {
            ATb[r] += phi[r] * y;
            for (int c = r; c < K; ++c) {
                ATA[r][c] += phi[r] * phi[c];
            }
        }
    }
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < r; ++c) ATA[r][c] = ATA[c][r];
        ATA[r][r] += 1e-8 * (1.0 + ATA[r][r]); // Mild Tikhonov regularization
    }

    // Gaussian elimination with partial pivoting (8x8)
    double M[K][K + 1] = {};
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < K; ++c) M[r][c] = ATA[r][c];
        M[r][K] = ATb[r];
    }
    bool ok = true;
    for (int col = 0; col < K; ++col) {
        int piv = col;
        for (int r = col + 1; r < K; ++r) {
            if (std::abs(M[r][col]) > std::abs(M[piv][col])) piv = r;
        }
        if (std::abs(M[piv][col]) < 1e-12) { ok = false; break; }
        if (piv != col) {
            for (int c = col; c <= K; ++c) std::swap(M[col][c], M[piv][c]);
        }
        for (int r = col + 1; r < K; ++r) {
            const double f = M[r][col] / M[col][col];
            for (int c = col; c <= K; ++c) M[r][c] -= f * M[col][c];
        }
    }
    double coeff[K] = {};
    if (ok) {
        for (int r = K - 1; r >= 0; --r) {
            double s = M[r][K];
            for (int c = r + 1; c < K; ++c) s -= M[r][c] * coeff[c];
            coeff[r] = s / M[r][r];
        }
        for (int i = 0; i < n; ++i) {
            const double x = static_cast<double>(i);
            const double u = x * inv_half_n - 1.0;
            const double u2 = u * u;
            base_db[i] += coeff[0] + coeff[1] * u + coeff[2] * u2 + coeff[3] * (u2 * u)
                        + coeff[4] * std::cos(w1 * x) + coeff[5] * std::sin(w1 * x)
                        + coeff[6] * std::cos(w2 * x) + coeff[7] * std::sin(w2 * x);
        }
    }

    std::vector<double> clean_res;
    clean_res.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (passband_mask[i] && !hi_window[i]) {
            clean_res.push_back(p_db[i] - base_db[i]);
        }
    }
    const double offset = median_of_inplace(clean_res);
    for (int i = 0; i < n; ++i) base_db[i] += offset;
    return base_db;
}

BaselineResults BaselineFitter::fit(
    const std::vector<double>& power,
    const std::vector<double>& freqs_mhz,
    const AstroConfig& config
) {
    int nfft = static_cast<int>(power.size());
    BaselineResults res;
    res.baseline_power.resize(nfft);
    res.baseline_db.resize(nfft);

    if (nfft == 0) return res;

    // 1. Passband mask (inner ~87%)
    int edge_bins = static_cast<int>(nfft * std::clamp(config.passband_edge_fraction, 0.0, 0.45));
    std::vector<uint8_t> passband_mask(nfft, 1);
    for (int i = 0; i < edge_bins; ++i) passband_mask[i] = 0;
    for (int i = nfft - edge_bins; i < nfft; ++i) passband_mask[i] = 0;

    // 2. HI protection window
    double hi_half_mhz = config.hi_protection_half_width_khz / 1e3;
    std::vector<uint8_t> hi_window(nfft, 0);
    for (int i = 0; i < nfft; ++i) {
        if (freqs_mhz[i] >= (HYDROGEN_REST_FREQ_MHZ - hi_half_mhz) &&
            freqs_mhz[i] <= (HYDROGEN_REST_FREQ_MHZ + hi_half_mhz)) {
            hi_window[i] = 1;
        }
    }

    // Vectorized conversion from raw power to dB
    std::vector<double> p_db(nfft);
    SimdDsp::power_to_db_f64(power.data(), p_db.data(), nfft, 1e-12);

    if (config.baseline_method == BaselineMethod::Poly) {
        res.baseline_db = fit_poly(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::ALS) {
        res.baseline_db = fit_als(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::SavitzkyGolay) {
        res.baseline_db = fit_savgol(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else if (config.baseline_method == BaselineMethod::StandingWave) {
        res.baseline_db = fit_standing_wave(p_db, freqs_mhz, passband_mask, hi_window, config);
    } else {
        res.baseline_db = fit_filter(p_db, freqs_mhz, passband_mask, hi_window, config);
    }

    // Vectorized conversion from baseline dB back to linear power
    SimdDsp::db_to_power_f64(res.baseline_db.data(), res.baseline_power.data(), nfft);
    for (int i = 0; i < nfft; ++i) {
        if (res.baseline_power[i] < 1e-12) res.baseline_power[i] = 1e-12;
    }

    return res;
}

void BaselineFitter::refine_local(
    const std::vector<double>& vel_kms,
    std::vector<double>& calibrated_k,
    std::vector<double>& calibrated_db,
    const AstroConfig& config,
    double& out_slope,
    double& out_intercept
) {
    size_t n = vel_kms.size();
    out_slope = 0.0;
    out_intercept = 0.0;
    if (n != calibrated_k.size() || n != calibrated_db.size()) return;

    const size_t edge_bins = static_cast<size_t>(
        static_cast<double>(n) * std::clamp(config.passband_edge_fraction, 0.0, 0.45));
    const double v_thresh = config.line_free_velocity_kms;
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    size_t m = 0;
    for (size_t i = edge_bins; i + edge_bins < n; ++i) {
        const double v = vel_kms[i];
        if (v >= -200.0 && v <= 200.0 && std::abs(v) >= v_thresh) {
            // Fit in linear Kelvin space to keep E[T_A] = 0 without log/Jensen bias
            const double y = calibrated_k[i];
            sum_x += v;
            sum_y += y;
            sum_xx += v * v;
            sum_xy += v * y;
            m++;
        }
    }

    if (m >= 10) {
        double denom = (static_cast<double>(m) * sum_xx - sum_x * sum_x);
        if (std::abs(denom) > 1e-12) {
            out_slope = (static_cast<double>(m) * sum_xy - sum_x * sum_y) / denom;
            out_intercept = (sum_y - out_slope * sum_x) / static_cast<double>(m);
        }

        const double inv_tsys = 1.0 / std::max(1.0, config.system_temp_k);
        thread_local std::vector<double> tl_ratio;
        tl_ratio.resize(n);

        size_t i = 0;
#if defined(__AVX2__)
        const __m256d v_slope = _mm256_set1_pd(out_slope);
        const __m256d v_inter = _mm256_set1_pd(out_intercept);
        const __m256d v_inv_t = _mm256_set1_pd(inv_tsys);
        const __m256d v_one   = _mm256_set1_pd(1.0);
        const __m256d v_min_r = _mm256_set1_pd(1e-6);
        for (; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(vel_kms.data() + i);
            __m256d tk = _mm256_loadu_pd(calibrated_k.data() + i);
#if SIMD_HAS_FMA
            __m256d corr = _mm256_fmadd_pd(v_slope, v, v_inter);
            __m256d tk_clean = _mm256_sub_pd(tk, corr);
            __m256d ratio = _mm256_fmadd_pd(tk_clean, v_inv_t, v_one);
#else
            __m256d corr = _mm256_add_pd(_mm256_mul_pd(v_slope, v), v_inter);
            __m256d tk_clean = _mm256_sub_pd(tk, corr);
            __m256d ratio = _mm256_add_pd(_mm256_mul_pd(tk_clean, v_inv_t), v_one);
#endif
            _mm256_storeu_pd(calibrated_k.data() + i, tk_clean);
            _mm256_storeu_pd(tl_ratio.data() + i, _mm256_max_pd(v_min_r, ratio));
        }
#endif
        for (; i < n; ++i) {
            double tk_clean = calibrated_k[i] - (out_slope * vel_kms[i] + out_intercept);
            calibrated_k[i] = tk_clean;
            tl_ratio[i] = std::max(1e-6, 1.0 + tk_clean * inv_tsys);
        }

        SimdDsp::power_to_db_f64(tl_ratio.data(), calibrated_db.data(), n, 1e-6);
    }
}

} // namespace astro
