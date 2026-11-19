#include "virtual_sdr.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <thread>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

struct Phasor {
    double re;
    double im;
    double dre;
    double dim;

    void init(double phase, double dphi) {
        re = std::cos(phase);
        im = std::sin(phase);
        dre = std::cos(dphi);
        dim = std::sin(dphi);
    }

    inline void step() {
        double next_re = re * dre - im * dim;
        double next_im = re * dim + im * dre;
        re = next_re;
        im = next_im;
    }

    inline void renorm() {
        double mag2 = re * re + im * im;
        if (mag2 > 0.0) {
            double inv = 1.5 - 0.5 * mag2;
            re *= inv;
            im *= inv;
        }
    }

    inline double final_phase() const {
        return std::atan2(im, re);
    }
};

// Fast SplitMix64 PRNG and Marsaglia polar complex Gaussian generator
inline uint64_t next_u64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline float next_uniform01(uint64_t& state) {
    return static_cast<float>((next_u64(state) >> 40) * (1.0 / 16777216.0));
}

inline void next_complex_gaussian(uint64_t& state, float sigma, float& out_i, float& out_q) {
    float u, v, s;
    do {
        uint64_t r = next_u64(state);
        u = static_cast<float>(static_cast<int32_t>(r & 0xFFFFFF) - 0x800000) * (1.0f / 8388608.0f);
        v = static_cast<float>(static_cast<int32_t>((r >> 24) & 0xFFFFFF) - 0x800000) * (1.0f / 8388608.0f);
        s = u * u + v * v;
    } while (s >= 1.0f || s <= 1e-12f);

    float scale = sigma * std::sqrt(-2.0f * std::log(s) / s);
    out_i = u * scale;
    out_q = v * scale;
}

inline uint8_t clamp_u8(float x) {
    int v = static_cast<int>(x + 0.5f);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

} // namespace

VirtualSdrDevice::VirtualSdrDevice() {
    rng_state_ = static_cast<uint64_t>(std::random_device{}()) ^ 0xA0761D6478BD642FULL;
    if (rng_state_ == 0) rng_state_ = 0x9E3779B97F4A7C15ULL;
}

bool VirtualSdrDevice::open(uint32_t /*index*/) {
    is_open_ = true;
    next_chunk_time_ = std::chrono::steady_clock::now();
    return true;
}

void VirtualSdrDevice::close() {
    is_open_ = false;
}

std::vector<float> VirtualSdrDevice::get_valid_gains() {
    return {0.0f, 0.9f, 1.4f, 2.7f, 3.7f, 7.7f, 8.7f, 12.5f, 14.4f, 15.7f, 16.6f,
            19.7f, 20.7f, 22.9f, 25.4f, 28.0f, 29.7f, 32.8f, 33.8f, 36.4f, 37.2f,
            38.6f, 40.2f, 42.1f, 43.4f, 43.9f, 44.5f, 48.0f, 49.6f};
}

bool VirtualSdrDevice::set_sample_rate(uint32_t rate_hz) {
    if (rate_hz == 0) return false;
    sample_rate_hz_.store(rate_hz);
    return true;
}

bool VirtualSdrDevice::set_frequency(uint64_t freq_hz) {
    center_freq_hz_.store(freq_hz);
    return true;
}

bool VirtualSdrDevice::set_gain(float gain_db, bool auto_gain) {
    gain_db_.store(gain_db);
    auto_gain_.store(auto_gain);
    return true;
}

bool VirtualSdrDevice::set_bias_tee(bool enabled) {
    bias_tee_.store(enabled);
    return true;
}

bool VirtualSdrDevice::set_ppm(int ppm) {
    ppm_.store(ppm);
    return true;
}

void VirtualSdrDevice::reset_buffer() {
    // No-op for virtual SDR
}

int VirtualSdrDevice::read_samples(uint8_t* buffer, int length, int* bytes_read) {
    if (bytes_read) *bytes_read = 0;
    if (!is_open_ || !buffer || length < 2) {
        return -1;
    }

    const int chunk_samples = length / 2;
    const uint32_t fs = std::max(1u, sample_rate_hz_.load());
    const uint64_t fc = center_freq_hz_.load();
    const double span = static_cast<double>(fs);

    // Real-time pacing with drift compensation
    auto now = std::chrono::steady_clock::now();
    if (next_chunk_time_ > now) {
        std::this_thread::sleep_until(next_chunk_time_);
        now = std::chrono::steady_clock::now();
    }
    double chunk_duration_sec = static_cast<double>(chunk_samples) / static_cast<double>(fs);
    auto chunk_duration = std::chrono::microseconds(static_cast<int64_t>(chunk_duration_sec * 1000000.0));
    if (next_chunk_time_ + chunk_duration * 2 < now) {
        next_chunk_time_ = now + chunk_duration;
    } else {
        next_chunk_time_ += chunk_duration;
    }

    // Multi-frequency RF simulation with phase continuity (hoisted band selection + complex phasor recurrence)
    const double dt = 1.0 / static_cast<double>(fs);
    const double two_pi_dt = 2.0 * M_PI * dt;
    const float noise_sigma = 14.0f;
    const float dc_offset_i = 10.0f + 128.0f;
    const float dc_offset_q = 128.0f;

    if (std::abs(static_cast<double>(fc) - SdrConstants::HI_REST_FREQ_HZ) < span * 0.48) {
        // Simulate Doppler-shifted (+7.4 km/s) narrowband thermal 21cm H I emission (~12 km/s FWHM)
        const double hi_line_freq_hz = SdrConstants::HI_REST_FREQ_HZ - 35000.0;
        const double hi_offset = hi_line_freq_hz - static_cast<double>(fc);
        Phasor hi_osc, test_spur_osc;
        hi_osc.init(hi_phase_, two_pi_dt * hi_offset);
        // Inject weak CW tone at -710 kHz (conjugate spur outside the HI rest window) for RFI/waterfall checks
        test_spur_osc.init(ripple_phase_, two_pi_dt * 710000.0);

        // Two cascaded 1-pole IIR filters shape white complex Gaussian excitation into a narrowband peak.
        // Solve |H_1(w)|^4 = 0.5 at w = 2*pi*f_hwhm/Fs (~28.43 kHz HWHM for 12 km/s FWHM).
        constexpr double kHiHwhmHz = 28427.79;
        const double w_hwhm = two_pi_dt * kHiHwhmHz;
        const double k_bw = 2.0 * (1.0 - std::cos(w_hwhm)) / (std::sqrt(2.0) - 1.0);
        const double alpha_exact = 0.5 * (-k_bw + std::sqrt(k_bw * k_bw + 4.0 * k_bw));
        const float lp_alpha = static_cast<float>(std::clamp(alpha_exact, 0.02, 0.5));
        const float lp_decay = 1.0f - lp_alpha;
        const float hi_sigma = noise_sigma * 0.50f; // +25% excess PSD (~+25 K at T_sys = 100 K)
        float lp1_i = hi_lp1_i_, lp1_q = hi_lp1_q_;
        float lp2_i = hi_lp2_i_, lp2_q = hi_lp2_q_;

        for (int n = 0; n < chunk_samples; ++n) {
            float ni, nq, h_ni, h_nq;
            next_complex_gaussian(rng_state_, noise_sigma, ni, nq);
            next_complex_gaussian(rng_state_, hi_sigma, h_ni, h_nq);

            lp1_i = lp_decay * lp1_i + lp_alpha * h_ni;
            lp1_q = lp_decay * lp1_q + lp_alpha * h_nq;
            lp2_i = lp_decay * lp2_i + lp_alpha * lp1_i;
            lp2_q = lp_decay * lp2_q + lp_alpha * lp1_q;

            hi_osc.step();
            test_spur_osc.step();
            if ((n & 1023) == 1023) {
                hi_osc.renorm();
                test_spur_osc.renorm();
            }

            float sig_i = lp2_i * static_cast<float>(hi_osc.re) - lp2_q * static_cast<float>(hi_osc.im)
                        + 3.5f * static_cast<float>(test_spur_osc.im);
            float sig_q = lp2_i * static_cast<float>(hi_osc.im) + lp2_q * static_cast<float>(hi_osc.re)
                        + 3.5f * static_cast<float>(test_spur_osc.re);

            buffer[2 * n]     = clamp_u8(ni + sig_i + dc_offset_i);
            buffer[2 * n + 1] = clamp_u8(nq + sig_q + dc_offset_q);
        }
        hi_phase_ = hi_osc.final_phase();
        ripple_phase_ = test_spur_osc.final_phase();
        hi_lp1_i_ = lp1_i; hi_lp1_q_ = lp1_q;
        hi_lp2_i_ = lp2_i; hi_lp2_q_ = lp2_q;
    } else if (fc >= 80000000ULL && fc <= 115000000ULL) {
        static const double fm_stations[] = {
            88.5e6, 89.2e6, 92.6e6, 96.9e6, 98.6e6, 100.3e6, 103.7e6, 105.8e6
        };
        Phasor active_osc[8];
        int active_idx[8];
        int num_active = 0;
        for (int s = 0; s < 8; ++s) {
            double offset = fm_stations[s] - static_cast<double>(fc);
            if (std::abs(offset) < (span * 0.48)) {
                active_osc[num_active].init(fm_phase_[s], two_pi_dt * offset);
                active_idx[num_active] = s;
                num_active++;
            }
        }

        for (int n = 0; n < chunk_samples; ++n) {
            float ni, nq;
            next_complex_gaussian(rng_state_, noise_sigma, ni, nq);
            float sig_i = 0.0f, sig_q = 0.0f;
            for (int k = 0; k < num_active; ++k) {
                active_osc[k].step();
                sig_i += 38.0f * static_cast<float>(active_osc[k].re);
                sig_q += 38.0f * static_cast<float>(active_osc[k].im);
            }
            if ((n & 1023) == 1023) {
                for (int k = 0; k < num_active; ++k) active_osc[k].renorm();
            }
            buffer[2 * n]     = clamp_u8(ni + sig_i + dc_offset_i);
            buffer[2 * n + 1] = clamp_u8(nq + sig_q + dc_offset_q);
        }
        for (int k = 0; k < num_active; ++k) {
            fm_phase_[active_idx[k]] = active_osc[k].final_phase();
        }
    } else if (fc >= 1080000000ULL && fc <= 1100000000ULL) {
        double offset = 1090.0e6 - static_cast<double>(fc);
        bool in_band = std::abs(offset) < (span * 0.48);
        Phasor osc;
        osc.init(adsb_phase_, two_pi_dt * offset);

        for (int n = 0; n < chunk_samples; ++n) {
            float ni, nq;
            next_complex_gaussian(rng_state_, noise_sigma, ni, nq);
            float sig_i = 0.0f, sig_q = 0.0f;
            if (in_band) {
                osc.step();
                if ((n & 1023) == 1023) osc.renorm();
                if (next_uniform01(rng_state_) < 0.03f) {
                    sig_i = 45.0f * static_cast<float>(osc.re);
                    sig_q = 45.0f * static_cast<float>(osc.im);
                }
            }
            buffer[2 * n]     = clamp_u8(ni + sig_i + dc_offset_i);
            buffer[2 * n + 1] = clamp_u8(nq + sig_q + dc_offset_q);
        }
        if (in_band) adsb_phase_ = osc.final_phase();
    } else {
        double* phase_ptr = &gen_phase_;
        double offset = span * 0.15;
        float amp = 22.0f;
        bool in_band = true;

        if (fc >= 118000000ULL && fc <= 137000000ULL) {
            phase_ptr = &am_phase_;
            offset = 121.5e6 - static_cast<double>(fc);
            amp = 24.0f;
            in_band = std::abs(offset) < (span * 0.48);
        } else if (fc >= 144000000ULL && fc <= 148000000ULL) {
            phase_ptr = &ham_phase_;
            offset = 144.8e6 - static_cast<double>(fc);
            amp = 28.0f;
            in_band = std::abs(offset) < (span * 0.48);
        } else if (fc >= 430000000ULL && fc <= 440000000ULL) {
            phase_ptr = &ism_phase_;
            offset = 433.92e6 - static_cast<double>(fc);
            amp = 30.0f;
            in_band = std::abs(offset) < (span * 0.48);
        }

        Phasor osc;
        osc.init(*phase_ptr, two_pi_dt * offset);
        for (int n = 0; n < chunk_samples; ++n) {
            float ni, nq;
            next_complex_gaussian(rng_state_, noise_sigma, ni, nq);
            float sig_i = 0.0f, sig_q = 0.0f;
            if (in_band) {
                osc.step();
                if ((n & 1023) == 1023) osc.renorm();
                sig_i = amp * static_cast<float>(osc.re);
                sig_q = amp * static_cast<float>(osc.im);
            }
            buffer[2 * n]     = clamp_u8(ni + sig_i + dc_offset_i);
            buffer[2 * n + 1] = clamp_u8(nq + sig_q + dc_offset_q);
        }
        if (in_band) *phase_ptr = osc.final_phase();
    }

    if (bytes_read) *bytes_read = chunk_samples * 2;
    return 0;
}
