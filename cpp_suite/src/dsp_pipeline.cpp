#include "dsp_pipeline.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DspPipeline::DspPipeline() {
    std::lock_guard<std::mutex> lock(dsp_mutex_);
    rebuild_plans();
}

DspPipeline::~DspPipeline() {
    std::lock_guard<std::mutex> lock(dsp_mutex_);
    if (fft_cfg_) {
        free(fft_cfg_);
        fft_cfg_ = nullptr;
    }
}

void DspPipeline::set_fft_size(int size) {
    // M8: Clamp before rounding up to even to prevent signed int overflow at INT_MAX
    size = std::clamp(size, 16, 1 << 20);
    size = (size + 1) & ~1;
    std::lock_guard<std::mutex> dsp_lock(dsp_mutex_);
    if (size != fft_size_.load(std::memory_order_relaxed)) {
        fft_size_.store(size, std::memory_order_release);
        rebuild_plans();
        {
            std::lock_guard<std::mutex> sample_lock(sample_mutex_);
            sample_buffer_.clear();
            sample_buffer_.reserve(static_cast<size_t>(size) * 2);
            has_new_samples_ = false;
        }
        flush();
    }
}

void DspPipeline::set_averaging(float factor) {
    if (!std::isfinite(factor)) return;
    avg_factor_.store(std::clamp(factor, 0.0f, 0.95f), std::memory_order_relaxed);
}

void DspPipeline::flush() {
    need_flush_.store(true, std::memory_order_release);
}

void DspPipeline::rebuild_plans() {
    if (fft_cfg_) {
        free(fft_cfg_);
        fft_cfg_ = nullptr;
    }

    const int n_pts = fft_size_.load(std::memory_order_relaxed);
    fft_cfg_ = kiss_fft_alloc(n_pts, 0, nullptr, nullptr);
    fft_in_.resize(n_pts);
    fft_out_.resize(n_pts);
    power_raw_.resize(n_pts);
    fft_mag_linear_.resize(n_pts);
    power_linear_raw_.resize(n_pts, 0.0f);
    power_linear_smooth_.assign(n_pts, 1e-12f);
    power_smooth_.assign(n_pts, -120.0f);
    scratch_u8_.resize(static_cast<size_t>(n_pts) * 2);
    telemetry_scratch_.reserve(static_cast<size_t>(n_pts) / 16 + 2);
    has_smooth_ = false;

    update_window();
}

void DspPipeline::update_window() {
    const int n_pts = fft_size_.load(std::memory_order_relaxed);
    window_scaled_.resize(n_pts);
    const double inv_n_minus_1 = 2.0 * M_PI / static_cast<double>(std::max(1, n_pts - 1));
    const float inv_128 = 1.0f / 128.0f;
    double sum_w = 0.0;

    for (int n = 0; n < n_pts; ++n) {
        // Hann window: 0.5 * (1 - cos(2*pi*n / (N-1)))
        float w = 0.5f * (1.0f - static_cast<float>(std::cos(static_cast<double>(n) * inv_n_minus_1)));
        sum_w += static_cast<double>(w);
        // Pre-scale by 1/128 for direct AVX multiplication with (u8 - 127.5)
        window_scaled_[n] = w * inv_128;
    }
    // Coherent window sum so full-scale complex tone (|I|,|Q| -> 1.0) maps to 0.0 dBFS
    coh_gain_sum_ = std::max(1.0, sum_w);
}

void DspPipeline::feed_samples(const uint8_t* data, size_t length) {
    // M8: Enforce even I/Q byte alignment so odd transfers never swap I and Q channels
    length &= ~size_t(1);
    if (!data || length == 0) return;

    std::lock_guard<std::mutex> lock(sample_mutex_);
    const size_t needed_bytes = static_cast<size_t>(fft_size_.load(std::memory_order_acquire)) * 2;
    if (needed_bytes == 0) return;

    if (length >= needed_bytes) {
        // Keep the latest full FFT frame for minimum display latency
        sample_buffer_.assign(data + (length - needed_bytes), data + length);
        has_new_samples_ = true;
    } else {
        // Accumulate small chunks into a rolling buffer bounded to needed_bytes
        sample_buffer_.insert(sample_buffer_.end(), data, data + length);
        if (sample_buffer_.size() > needed_bytes) {
            const size_t excess = (sample_buffer_.size() - needed_bytes) & ~size_t(1);
            const size_t keep = sample_buffer_.size() - excess;
            std::memmove(sample_buffer_.data(), sample_buffer_.data() + excess, keep);
            sample_buffer_.resize(keep);
        }
        if (sample_buffer_.size() >= needed_bytes) {
            has_new_samples_ = true;
        }
    }
}

void DspPipeline::compute_telemetry(const std::vector<float>& power, float peak_db) {
    if (power.empty()) return;

    // Estimate noise floor using the 50th-percentile (median) of decimated bins
    telemetry_scratch_.clear();
    for (size_t i = 0; i < power.size(); i += 16) {
        telemetry_scratch_.push_back(power[i]);
    }
    const size_t mid_idx = telemetry_scratch_.size() / 2;
    std::nth_element(telemetry_scratch_.begin(), telemetry_scratch_.begin() + mid_idx, telemetry_scratch_.end());
    float noise_floor = telemetry_scratch_[mid_idx];

    noise_floor_db_.store(noise_floor, std::memory_order_relaxed);
    float snr = std::max(0.0f, peak_db - noise_floor);
    snr_db_.store(snr, std::memory_order_relaxed);
}

bool DspPipeline::get_power_spectrum(std::vector<float>& out_power, float& out_peak_freq, float& out_peak_db,
                                     uint64_t center_freq_hz, uint32_t sample_rate_hz) {
    double peak_freq_hz = 0.0;
    if (!get_power_spectrum(out_power, peak_freq_hz, out_peak_db, center_freq_hz, sample_rate_hz)) {
        return false;
    }
    out_peak_freq = static_cast<float>(peak_freq_hz);
    return true;
}

bool DspPipeline::get_power_spectrum(std::vector<float>& out_power, double& out_peak_freq, float& out_peak_db,
                                     uint64_t center_freq_hz, uint32_t sample_rate_hz) {
    std::lock_guard<std::mutex> dsp_lock(dsp_mutex_);

    const int n_pts = fft_size_.load(std::memory_order_acquire);
    const size_t needed_bytes = static_cast<size_t>(n_pts) * 2;
    {
        std::lock_guard<std::mutex> sample_lock(sample_mutex_);
        if (!has_new_samples_ || sample_buffer_.size() < needed_bytes) {
            return false;
        }
        if (scratch_u8_.size() < needed_bytes) {
            scratch_u8_.resize(needed_bytes);
        }
        const size_t offset = sample_buffer_.size() - needed_bytes;
        std::memcpy(scratch_u8_.data(), sample_buffer_.data() + offset, needed_bytes);
        has_new_samples_ = false;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const float inv_norm_sq = static_cast<float>(1.0 / (coh_gain_sum_ * coh_gain_sum_));

    // 1. Vectorized Normalization & Hann Windowing (AVX2 / SSE2)
    SimdDsp::apply_hann_window_complex_fast(
        scratch_u8_.data(),
        window_scaled_.data(),
        reinterpret_cast<float*>(fft_in_.data()),
        n_pts
    );

    // 2. High-speed complex FFT (runtime-verified AVX2+FMA Stockham for power-of-2 >= 8, KissFFT fallback)
    const float* fft_power_src = nullptr;
    if (SimdDsp::detect_cpu_simd() >= SimdDsp::CpuSimdLevel::AVX2 &&
        SimdDsp::fft_cpx_f32_avx2(reinterpret_cast<float*>(fft_in_.data()), n_pts)) {
        fft_power_src = reinterpret_cast<const float*>(fft_in_.data());
    } else {
        if (!fft_cfg_) {
            return false;
        }
        kiss_fft(fft_cfg_, fft_in_.data(), fft_out_.data());
        fft_power_src = reinterpret_cast<const float*>(fft_out_.data());
    }

    // 3. Vectorized FFT Shift, Instantaneous Magnitude & Fast dBFS
    if (fft_mag_linear_.size() != static_cast<size_t>(n_pts)) {
        fft_mag_linear_.resize(n_pts);
    }
    if (power_linear_raw_.size() != static_cast<size_t>(n_pts)) {
        power_linear_raw_.resize(n_pts);
        power_linear_smooth_.assign(n_pts, 1e-12f);
    }
    SimdDsp::compute_power_dbfs_avx2(
        fft_power_src,
        power_raw_.data(),
        n_pts,
        inv_norm_sq,
        fft_mag_linear_.data()
    );

    // Track instantaneous FFT peak alongside smoothed Power Spectrum peak
    int fft_peak_idx = 0;
    float max_fft_db = -999.0f;
    for (int k = 0; k < n_pts; ++k) {
        power_linear_raw_[k] = fft_mag_linear_[k] * fft_mag_linear_[k];
        if (power_raw_[k] > max_fft_db) {
            max_fft_db = power_raw_[k];
            fft_peak_idx = k;
        }
    }
    fft_peak_db_.store(max_fft_db, std::memory_order_relaxed);
    fft_peak_mag_.store(fft_mag_linear_[fft_peak_idx], std::memory_order_relaxed);

    // 4. Exponential Smoothing in linear power (unbiased noise PSD estimator) -> dBFS
    const float cur_avg = avg_factor_.load(std::memory_order_relaxed);
    bool do_flush = need_flush_.exchange(false, std::memory_order_acq_rel);
    if (do_flush || !has_smooth_) {
        std::memcpy(power_linear_smooth_.data(), power_linear_raw_.data(), static_cast<size_t>(n_pts) * sizeof(float));
        std::memcpy(power_smooth_.data(), power_raw_.data(), static_cast<size_t>(n_pts) * sizeof(float));
    } else {
        SimdDsp::apply_smoothing_avx2(power_linear_smooth_.data(), power_linear_raw_.data(), n_pts, cur_avg);
        int k = 0;
#if defined(__AVX2__)
        const __m256 v_eps = _mm256_set1_ps(1e-14f);
        for (; k + 8 <= n_pts; k += 8) {
            __m256 pwr = _mm256_add_ps(_mm256_loadu_ps(power_linear_smooth_.data() + k), v_eps);
            _mm256_storeu_ps(power_smooth_.data() + k, SimdDsp::fast_10log10_avx2(pwr));
        }
#endif
        for (; k < n_pts; ++k) {
            power_smooth_[k] = SimdDsp::fast_10log10_scalar(power_linear_smooth_[k] + 1e-14f);
        }
    }
    has_smooth_ = true;

    // 5. Peak Detection (Smoothed Power Spectrum)
    int peak_idx = 0;
    float max_pwr = -999.0f;
    for (int k = 0; k < n_pts; ++k) {
        if (power_smooth_[k] > max_pwr) {
            max_pwr = power_smooth_[k];
            peak_idx = k;
        }
    }

    out_power = power_smooth_;
    out_peak_db = max_pwr;

    const double start_f = static_cast<double>(center_freq_hz) - (static_cast<double>(sample_rate_hz) / 2.0);
    const double step_f = static_cast<double>(sample_rate_hz) / static_cast<double>(n_pts);
    out_peak_freq = start_f + static_cast<double>(peak_idx) * step_f;
    fft_peak_freq_.store(start_f + static_cast<double>(fft_peak_idx) * step_f, std::memory_order_relaxed);

    // Compute Telemetry (SNR, Noise floor)
    compute_telemetry(power_smooth_, max_pwr);

    // Measure latency & throughput
    auto t_end = std::chrono::high_resolution_clock::now();
    double latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    last_latency_us_.store(latency_us, std::memory_order_relaxed);

    if (latency_us > 0.0) {
        double msps = (static_cast<double>(n_pts) / (latency_us * 1e-6)) / 1e6;
        dsp_throughput_msps_.store(msps, std::memory_order_relaxed);
    }

    return true;
}
