#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "astrophysics.h"
#include "baseline_fitter.h"
#include "config.h"
#include "constants.h"
#include "cs16_writer.h"
#include "dc_excision.h"
#include "dsp_pipeline.h"
#include "kiss_fft.h"
#include "lockfree_ringbuffer.h"
#include "rfi_filter.h"
#include "sidecar_reader.h"
#include "simd_dsp.h"
#include "streaming_processor.h"
#include "waterfall.h"

namespace fs = std::filesystem;

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
    } else { \
        std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    } \
} while(0)

static void test_simd_math_and_cs16_kernels() {
    std::cout << "[Test 1] SimdDsp AVX2 log10/exp10, cs16 window & max-pool waterfall...\n";

    // Unaligned + odd-length test to verify unaligned AVX2 loadu/storeu safety
    constexpr size_t N = 263;
    std::vector<double> p(N + 1), db(N + 1), rt(N + 1);
    for (size_t i = 1; i < N + 1; ++i) {
        p[i] = std::pow(10.0, -10.0 + 14.0 * static_cast<double>(i) / static_cast<double>(N));
    }
    SimdDsp::power_to_db_f64(p.data() + 1, db.data() + 1, N);
    SimdDsp::db_to_power_f64(db.data() + 1, rt.data() + 1, N);

    double max_db_err = 0.0;
    double max_rt_rel_err = 0.0;
    for (size_t i = 1; i < N + 1; ++i) {
        double exact_db = 10.0 * std::log10(p[i]);
        max_db_err = std::max(max_db_err, std::abs(db[i] - exact_db));
        max_rt_rel_err = std::max(max_rt_rel_err, std::abs(rt[i] - p[i]) / p[i]);
    }
    TEST_ASSERT(max_db_err < 1e-10, "AVX2 power_to_db_f64 accuracy within 1e-10 dB");
    TEST_ASSERT(max_rt_rel_err < 1e-10, "AVX2 db_to_power_f64 roundtrip accuracy within 1e-10 rel");

    // CS16 conversion accuracy vs scalar (win_interleaved has 2*M floats [w0, w0, w1, w1, ...])
    constexpr size_t M = 1031;
    std::vector<int16_t> iq(2 * M);
    std::vector<float> win_interleaved(2 * M);
    std::vector<kiss_fft_cpx> cpx_fast(M);
    for (size_t i = 0; i < M; ++i) {
        iq[2 * i] = static_cast<int16_t>((i * 137) % 30000 - 15000);
        iq[2 * i + 1] = static_cast<int16_t>((i * 251) % 30000 - 15000);
        float w = (0.5f * (1.0f - std::cos(2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(M - 1)))) / 32768.0f;
        win_interleaved[2 * i] = w;
        win_interleaved[2 * i + 1] = w;
    }
    SimdDsp::convert_cs16_window_float_fast(iq.data(), win_interleaved.data(), reinterpret_cast<float*>(cpx_fast.data()), M);
    double max_iq_err = 0.0;
    for (size_t i = 0; i < M; ++i) {
        float ref_r = static_cast<float>(iq[2 * i]) * win_interleaved[2 * i];
        float ref_i = static_cast<float>(iq[2 * i + 1]) * win_interleaved[2 * i + 1];
        max_iq_err = std::max(max_iq_err, static_cast<double>(std::abs(cpx_fast[i].r - ref_r)));
        max_iq_err = std::max(max_iq_err, static_cast<double>(std::abs(cpx_fast[i].i - ref_i)));
    }
    TEST_ASSERT(max_iq_err < 1e-6, "convert_cs16_window_float_fast matches scalar float32 reference");

    // Peak-preserving max-pool waterfall mapping when FFT bins >> display pixels
    std::vector<float> spec(4096, -100.0f);
    spec[2050] = -40.0f; // Narrow single-bin peak
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<uint32_t>(i);
    std::vector<uint32_t> out_px(400, 0);
    SimdDsp::map_waterfall_line_fast(spec.data(), 4096, -100.0f, -40.0f, lut.data(), out_px.data(), 400);
    uint32_t max_idx = 0;
    for (uint32_t px : out_px) max_idx = std::max(max_idx, px);
    TEST_ASSERT(max_idx == 255, "map_waterfall_line_fast must preserve narrow single-bin peak");
}

static void test_ring_buffer_wraparound() {
    std::cout << "[Test 2] LockFreeSpscRingBuffer wraparound & read/write invariants...\n";
    LockFreeSpscRingBuffer<uint8_t> rb(1024);
    std::vector<uint8_t> blk1(768, 0xA5);
    std::vector<uint8_t> blk2(768, 0x5A);
    std::vector<uint8_t> out1(512, 0), out2(768, 0);

    TEST_ASSERT(rb.write(blk1.data(), blk1.size()) == 768, "Initial write within capacity succeeds");
    TEST_ASSERT(rb.available_read() == 768, "available_read reflects written bytes");
    TEST_ASSERT(rb.available_write() <= 1024, "available_write bounded by capacity");
    TEST_ASSERT(rb.read(out1.data(), 512) == 512, "Read first 512 bytes succeeds");
    TEST_ASSERT(rb.write(blk2.data(), blk2.size()) == 768, "Wraparound write succeeds without underflow");
    TEST_ASSERT(rb.available_read() == 1024, "Wraparound available_read equals full capacity (1024)");
    TEST_ASSERT(rb.read(out2.data(), 256) == 256 && out2[0] == 0xA5, "First wrap segment matches blk1");
    TEST_ASSERT(rb.read(out2.data(), 768) == 768 && out2[0] == 0x5A && out2[767] == 0x5A, "Second wrap segment matches blk2");
    TEST_ASSERT(rb.available_read() == 0, "Drained ring buffer has 0 readable bytes");
}

static void test_waterfall_history_resize() {
    std::cout << "[Test 3] Waterfall::resize_preserving_history...\n";
    Waterfall wf(400, 120);
    std::vector<float> hot(400, -40.0f);
    std::vector<float> cold(400, -100.0f);
    wf.add_line(cold, -100.0f, -40.0f);
    wf.add_line(hot, -100.0f, -40.0f); // Row 0 is hot

    uint32_t top_before = wf.get_pixel_data()[200];
    wf.resize_preserving_history(800, 200);
    TEST_ASSERT(wf.get_width() == 800 && wf.get_height() == 200, "Upscaled dimensions updated");
    TEST_ASSERT(wf.get_pixel_data()[400] == top_before, "Row 0 pixel color preserved across horizontal scale");

    // Padded bottom row should match Obsidian #0D1117 (ABGR: 0xFF17110D)
    uint32_t bottom_row_px = wf.get_pixel_data()[199 * 800 + 10];
    TEST_ASSERT((bottom_row_px & 0x00FFFFFFu) == 0x0017110Du, "New padded waterfall rows use Obsidian #0D1117");
}

static void test_scientific_pipeline_solvers() {
    std::cout << "[Test 4] DCExcision, RFIFilter, BaselineFitter (Ripple/ALS/Poly) & Gaussian LM fit...\n";
    astro::AstroConfig cfg;
    cfg.center_freq_hz = 1420.5e6;
    cfg.sample_rate_hz = 2.4e6;
    cfg.fft_size = 4096;
    cfg.system_temp_k = 100.0;

    const int n = cfg.fft_size;
    std::vector<double> freqs_hz(n), freqs_mhz(n), power(n);
    const double df = cfg.sample_rate_hz / static_cast<double>(n);
    const double f0 = cfg.center_freq_hz - 0.5 * cfg.sample_rate_hz;

    for (int i = 0; i < n; ++i) {
        freqs_hz[i] = f0 + i * df;
        freqs_mhz[i] = freqs_hz[i] / 1e6;
        double norm_f = (freqs_mhz[i] - 1420.5) / 1.2;
        double bandpass = 100.0 * (1.0 - 0.20 * norm_f * norm_f + 0.008 * std::cos(8.0 * norm_f));
        double hi_dv = astro::SPEED_OF_LIGHT_KMS * (astro::HYDROGEN_REST_FREQ_HZ - freqs_hz[i]) / astro::HYDROGEN_REST_FREQ_HZ;
        double hi_signal = 15.0 * std::exp(-0.5 * std::pow((hi_dv - 6.0) / 4.5, 2.0));
        power[i] = bandpass * (1.0 + hi_signal / cfg.system_temp_k);
    }

    // Inject DC spike at center and narrow RFI spike outside HI zone
    power[n / 2] *= 10.0;
    power[400] *= 25.0;

    astro::DCExcisionInfo dc_info;
    auto dc_clean = astro::DCExcision::excise(power, cfg, dc_info);
    TEST_ASSERT(dc_clean.size() == static_cast<size_t>(n) && dc_info.cleared_bins > 0, "DCExcision excised center spike");
    TEST_ASSERT(dc_clean[n / 2] < power[n / 2] * 0.5, "DC bin power suppressed");

    int num_rfi = 0;
    auto rfi_clean = astro::RFIFilter::mitigate(dc_clean, freqs_mhz, cfg, num_rfi);
    TEST_ASSERT(num_rfi >= 1, "RFIFilter caught narrow RFI spike");
    TEST_ASSERT(rfi_clean[400] < dc_clean[400] * 0.2, "RFI spike interpolated out");

    for (auto method : {astro::BaselineMethod::RippleFilter, astro::BaselineMethod::ALS, astro::BaselineMethod::Poly}) {
        astro::AstroConfig mcfg = cfg;
        mcfg.baseline_method = method;
        auto base = astro::BaselineFitter::fit(rfi_clean, freqs_mhz, mcfg);
        bool all_valid = (base.baseline_power.size() == static_cast<size_t>(n));
        for (double bp : base.baseline_power) {
            if (!std::isfinite(bp) || bp <= 0.0) { all_valid = false; break; }
        }
        TEST_ASSERT(all_valid, "BaselineFitter produced finite positive baseline for solver");
    }

    auto base = astro::BaselineFitter::fit(rfi_clean, freqs_mhz, cfg);
    std::vector<double> cal_k(n), cal_db(n);
    for (int i = 0; i < n; ++i) {
        double ratio = rfi_clean[i] / base.baseline_power[i];
        cal_k[i] = cfg.system_temp_k * (ratio - 1.0);
        cal_db[i] = 10.0 * std::log10(std::max(ratio, 1e-12));
    }
    auto vel = astro::Astrophysics::compute_doppler_velocity(freqs_hz);
    double slope = 0.0, intercept = 0.0;
    astro::BaselineFitter::refine_local(vel, cal_k, cal_db, cfg, slope, intercept);

    std::vector<uint8_t> win_u8(n, 0);
    std::vector<bool> win_b(n, false);
    for (int i = 0; i < n; ++i) {
        if (std::abs(vel[i]) <= 60.0) { win_u8[i] = 1; win_b[i] = true; }
    }
    auto fit_u8 = astro::Astrophysics::fit_gaussian(vel, cal_k, win_u8);
    auto fit_b  = astro::Astrophysics::fit_gaussian(vel, cal_k, win_b);
    TEST_ASSERT(fit_u8.success && fit_b.success, "Gaussian LM fit succeeded");
    TEST_ASSERT(std::abs(fit_u8.v_center_kms - fit_b.v_center_kms) < 1e-6, "uint8 & bool fit_gaussian overloads match");
    TEST_ASSERT(std::abs(fit_u8.v_center_kms - 6.0) < 1.5, "Recovered HI centroid velocity within 1.5 km/s");
    std::cout << "  [Info] Test 4 Gaussian fit: amp=" << fit_u8.amplitude_k
              << " K, v_center=" << fit_u8.v_center_kms
              << " km/s, fwhm=" << fit_u8.fwhm_kms << " km/s\n";
    TEST_ASSERT(std::abs(fit_u8.amplitude_k - 15.0) < 2.0, "Recovered HI brightness temp within 2 K of 15 K");

    // Multi-method line profile analysis verification (DoubleGaussian, VoigtProfile, GaussHermite, SpectralMoments)
    for (auto lm : {astro::LineAnalysisMethod::GaussianLS,
                    astro::LineAnalysisMethod::DoubleGaussian,
                    astro::LineAnalysisMethod::VoigtProfile,
                    astro::LineAnalysisMethod::GaussHermite,
                    astro::LineAnalysisMethod::SpectralMoments}) {
        auto mfit = astro::Astrophysics::fit_line_profile(vel, cal_k, win_u8, lm);
        TEST_ASSERT(mfit.success && mfit.fitted_curve.size() == static_cast<size_t>(n),
                    "Multi-method fit_line_profile succeeded and populated full curve");
        TEST_ASSERT(std::abs(mfit.v_center_kms - 6.0) < 2.5,
                    "Multi-method fit_line_profile recovered H I velocity centroid within 2.5 km/s");
        TEST_ASSERT(mfit.fwhm_kms > 5.0 && mfit.fwhm_kms < 22.0,
                    "Multi-method fit_line_profile recovered physically valid FWHM");
    }
}

static void test_streaming_fft_parity_and_sidecar_dedup() {
    std::cout << "[Test 5] AVX2 Stockham FFT tone parity, SidecarReader dedup & Cs16Writer auto-stop...\n";
    fs::path tmp_dir = fs::path("tests") / "test_tmp_opt";
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);

    // 1. Write synthetic .cs16 with pure +300 kHz complex sinusoid at Fs = 2.4 MHz
    fs::path tone_path = tmp_dir / "tone_test.cs16";
    const double fs_hz = 2.4e6;
    const double f_tone = 300000.0;
    const size_t total_cpx = 4096 * 40; // Multiple blocks triggering 8-block AVX2 Stockham batching
    {
        std::ofstream ofs(tone_path, std::ios::binary);
        std::vector<int16_t> buf(2 * total_cpx);
        const double w = 2.0 * 3.14159265358979323846 * f_tone / fs_hz;
        for (size_t i = 0; i < total_cpx; ++i) {
            double ang = w * static_cast<double>(i);
            buf[2 * i]     = static_cast<int16_t>(12000.0 * std::cos(ang));
            buf[2 * i + 1] = static_cast<int16_t>(12000.0 * std::sin(ang));
        }
        ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(int16_t)));
    }

    // Power-of-two AVX2 batched FFT (4096) vs non-power-of-two KissFFT fallback (3072)
    auto find_peak_hz = [&](int nfft) -> double {
        astro::AstroConfig cfg;
        cfg.center_freq_hz = 1420.5e6;
        cfg.sample_rate_hz = fs_hz;
        cfg.fft_size = nfft;
        cfg.chunk_blocks = 8;
        cfg.verbose = false;
        astro::StreamingProcessor proc(tone_path.string(), cfg);
        auto res = proc.process();
        size_t best_bin = 0;
        for (size_t i = 1; i < res.averaged_power.size(); ++i) {
            if (res.averaged_power[i] > res.averaged_power[best_bin]) best_bin = i;
        }
        return res.freq_hz[best_bin];
    };

    double peak_avx2_hz = find_peak_hz(4096);
    double peak_kiss_hz = find_peak_hz(3072);
    double expected_hz  = 1420.5e6 + f_tone;
    TEST_ASSERT(std::abs(peak_avx2_hz - expected_hz) < 1500.0, "AVX2 Stockham FFT locates +300 kHz tone cleanly");
    TEST_ASSERT(std::abs(peak_kiss_hz - expected_hz) < 1500.0, "KissFFT fallback locates +300 kHz tone cleanly");

    // 2. SidecarReader canonical deduplication + .txt .768 precision check
    fs::path sub_dot = tmp_dir / ".";
    auto deduped = astro::SidecarReader::scan_cs16_files({tmp_dir.string(), sub_dot.string()});
    TEST_ASSERT(deduped.size() == 1, "SidecarReader deduplicates overlapping canonical paths");

    // 3. Cs16Writer non-blocking max_duration_s auto-stop + sidecar check
    fs::path cap_path = tmp_dir / "autostop_cap.cs16";
    Cs16Writer writer(tmp_dir.string());
    std::string started = writer.start_recording(2400000U, 1420500000ULL, "SimTest", 25.4f, false, cap_path.string(), 0.25);
    TEST_ASSERT(!started.empty(), "Cs16Writer started timed recording");
    std::vector<uint8_t> chunk(65536, 128);
    for (int k = 0; k < 15 && writer.is_recording(); ++k) {
        writer.feed_samples(chunk.data(), chunk.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    TEST_ASSERT(!writer.is_recording(), "Cs16Writer auto-stopped non-blockingly after max_duration_s");
    TEST_ASSERT(fs::exists(tmp_dir / "autostop_cap.json") && fs::exists(tmp_dir / "autostop_cap.txt"),
                "Cs16Writer flushed both .json and .txt sidecars on auto-stop");

    fs::remove_all(tmp_dir);
}

static void test_simd_fft_and_rfi_kernels() {
    std::cout << "[Test 6] AVX2 exp/FFT, Batcher median-15 & O(N log N) Filter...\n";

#ifdef __AVX2__
    // 1. fast_exp_pd_avx2 precision & underflow handling
    alignas(32) const double exp_in[8] = {-800.0, -708.1, -50.0, -0.5, 0.0, 0.5, 10.0, 300.0};
    alignas(32) double exp_out[8] = {};
    _mm256_storeu_pd(exp_out,     SimdDsp::fast_exp_pd_avx2(_mm256_loadu_pd(exp_in)));
    _mm256_storeu_pd(exp_out + 4, SimdDsp::fast_exp_pd_avx2(_mm256_loadu_pd(exp_in + 4)));
    TEST_ASSERT(exp_out[0] == 0.0 && exp_out[1] == 0.0, "fast_exp_pd_avx2 flushes <= -708 safely to 0");
    double max_exp_rel_err = 0.0;
    for (int k = 2; k < 8; ++k) {
        double ref = std::exp(exp_in[k]);
        max_exp_rel_err = std::max(max_exp_rel_err, std::abs(exp_out[k] - ref) / ref);
    }
    TEST_ASSERT(max_exp_rel_err < 1e-13, "fast_exp_pd_avx2 matches std::exp to < 1e-13 relative error");
#endif

    // 2. SimdDsp::fft_cpx_f32_avx2 parity against KissFFT for sizes 8..4096
    const int fft_sizes[] = {8, 16, 32, 64, 512, 4096};
    for (int nfft : fft_sizes) {
        std::vector<kiss_fft_cpx> in_kiss(nfft), out_kiss(nfft);
        std::vector<float> buf_avx(static_cast<size_t>(nfft) * 2);
        for (int i = 0; i < nfft; ++i) {
            float re = std::sin(0.137f * static_cast<float>(i)) + 0.5f * std::cos(0.41f * static_cast<float>(i));
            float im = std::cos(0.219f * static_cast<float>(i)) - 0.3f * std::sin(0.07f * static_cast<float>(i));
            in_kiss[i] = {re, im};
            buf_avx[2 * i] = re;
            buf_avx[2 * i + 1] = im;
        }
        kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, nullptr, nullptr);
        kiss_fft(cfg, in_kiss.data(), out_kiss.data());
        free(cfg);

        bool ok = SimdDsp::fft_cpx_f32_avx2(buf_avx.data(), nfft);
#ifdef __AVX2__
        TEST_ASSERT(ok, "fft_cpx_f32_avx2 succeeds for power-of-2 >= 8");
        double max_rel = 0.0;
        double norm_scale = static_cast<double>(nfft);
        for (int i = 0; i < nfft; ++i) {
            double dr = static_cast<double>(buf_avx[2 * i])     - static_cast<double>(out_kiss[i].r);
            double di = static_cast<double>(buf_avx[2 * i + 1]) - static_cast<double>(out_kiss[i].i);
            max_rel = std::max(max_rel, std::hypot(dr, di) / norm_scale);
        }
        TEST_ASSERT(max_rel < 1e-4, "fft_cpx_f32_avx2 matches KissFFT forward DFT");
#else
        (void)ok;
#endif
    }

    // 3. Batcher Odd-Even Merge median-15 RFI filter parity across edge & 4k+remainder sizes
    const int rfi_sizes[] = {10, 15, 67, 515};
    for (int nfft : rfi_sizes) {
        std::vector<double> pwr(nfft), freqs(nfft);
        for (int i = 0; i < nfft; ++i) {
            double x = static_cast<double>(i);
            freqs[i] = 1419.5 + 2.4 * x / static_cast<double>(nfft);
            pwr[i] = 10.0 + ((i % 3 == 0) ? 0.02 : -0.02) + 0.1 * std::sin(0.31 * x);
        }
        // Duplicate ties + interior/boundary RFI spikes
        pwr[2] = pwr[1];
        if (nfft > 20) {
            pwr[12] = 55.0;
            pwr[nfft / 2 + 5] = 80.0;
            pwr[nfft - 3] = 60.0;
        }
        astro::AstroConfig cfg;
        cfg.sample_rate_hz = 2.4e6;
        cfg.center_freq_hz = 1420.5e6;
        cfg.rfi_threshold_mad = 5.0;

        std::vector<double> ref_out = pwr;
        int ref_cnt = 0;
        if (nfft >= 32) {
            int edge_bins = static_cast<int>(nfft * std::clamp(cfg.passband_edge_fraction, 0.0, 0.45));
            double hi_half = cfg.hi_protection_half_width_khz / 1e3;
            std::vector<uint8_t> valid(nfft, 1);
            for (int i = 0; i < nfft; ++i) {
                if (i < edge_bins || i >= nfft - edge_bins) valid[i] = 0;
                if (freqs[i] >= astro::HYDROGEN_REST_FREQ_MHZ - hi_half &&
                    freqs[i] <= astro::HYDROGEN_REST_FREQ_MHZ + hi_half) valid[i] = 0;
            }
            std::vector<double> ref_res(nfft), ref_valid;
            for (int i = 0; i < nfft; ++i) {
                double win[15] = {};
                int lo = std::max(0, i - 7);
                int hi = std::min(nfft - 1, i + 7);
                int count = 0;
                for (int j = lo; j <= hi; ++j) win[count++] = pwr[j];
                int mid_w = count / 2;
                std::nth_element(win, win + mid_w, win + count);
                double med_w = (count % 2) ? win[mid_w] : 0.5 * (*std::max_element(win, win + mid_w) + win[mid_w]);
                ref_res[i] = pwr[i] - med_w;
                if (valid[i]) ref_valid.push_back(ref_res[i]);
            }
            auto med_fn = [](std::vector<double>& v) {
                size_t m = v.size() / 2;
                std::nth_element(v.begin(), v.begin() + m, v.end());
                return (v.size() % 2) ? v[m] : 0.5 * (*std::max_element(v.begin(), v.begin() + m) + v[m]);
            };
            double ref_med = med_fn(ref_valid);
            for (double& r : ref_valid) r = std::abs(r - ref_med);
            double ref_sig = std::max(1e-12, 1.4826 * med_fn(ref_valid));
            double ref_thr = cfg.rfi_threshold_mad * ref_sig;
            std::vector<uint8_t> is_rfi(nfft, 0);
            std::vector<int> rfi_idx;
            for (int i = 0; i < nfft; ++i) {
                if (valid[i] && (ref_res[i] - ref_med) > ref_thr) {
                    is_rfi[i] = 1;
                    rfi_idx.push_back(i);
                }
            }
            ref_cnt = static_cast<int>(rfi_idx.size());
            for (size_t k = 0; k < rfi_idx.size();) {
                const int run_start = rfi_idx[k];
                int run_end = run_start;
                while (k + 1 < rfi_idx.size() && rfi_idx[k + 1] == run_end + 1) {
                    run_end = rfi_idx[++k];
                }
                ++k;
                int l_idx = run_start - 1;
                while (l_idx >= 0 && is_rfi[l_idx]) --l_idx;
                int r_idx = run_end + 1;
                while (r_idx < nfft && is_rfi[r_idx]) ++r_idx;
                if (l_idx < 0 && r_idx >= nfft) continue;
                const double v_l = (l_idx >= 0) ? pwr[l_idx] : pwr[r_idx];
                const double v_r = (r_idx < nfft) ? pwr[r_idx] : pwr[l_idx];
                if (l_idx < 0) {
                    for (int idx = run_start; idx <= run_end; ++idx) ref_out[idx] = v_r;
                } else if (r_idx >= nfft) {
                    for (int idx = run_start; idx <= run_end; ++idx) ref_out[idx] = v_l;
                } else {
                    const int span = std::max(1, r_idx - l_idx);
                    for (int idx = run_start; idx <= run_end; ++idx) {
                        const double t = static_cast<double>(idx - l_idx) / static_cast<double>(span);
                        ref_out[idx] = v_l + t * (v_r - v_l);
                    }
                }
            }
        }

        int got_cnt = 0;
        auto got_out = astro::RFIFilter::mitigate(pwr, freqs, cfg, got_cnt);
        TEST_ASSERT(got_cnt == ref_cnt, "RfiFilter Batcher median-15 matches scalar RFI count");
        double max_diff = 0.0;
        for (int i = 0; i < nfft; ++i) max_diff = std::max(max_diff, std::abs(got_out[i] - ref_out[i]));
        TEST_ASSERT(max_diff < 1e-12, "RfiFilter Batcher median-15 output matches scalar reference identically");
    }

    // Boundary-index RFI spike test (idx=0 and idx=nfft-1 with edge_fraction=0.0) -> must not read OOB
    {
        constexpr int nfft = 64;
        std::vector<double> pwr(nfft, 10.0), freqs(nfft);
        for (int i = 0; i < nfft; ++i) {
            freqs[i] = 1419.5 + 2.4 * static_cast<double>(i) / nfft;
            pwr[i] += 0.01 * std::sin(0.5 * i);
        }
        pwr[0] = 500.0;
        pwr[nfft - 1] = 500.0;
        astro::AstroConfig cfg;
        cfg.passband_edge_fraction = 0.0;
        int cnt = 0;
        auto out = astro::RFIFilter::mitigate(pwr, freqs, cfg, cnt);
        TEST_ASSERT(cnt >= 2 && out[0] < 20.0 && out[nfft - 1] < 20.0,
                    "RFIFilter replaces boundary index (0 and nfft-1) RFI safely without OOB read");
    }

    // 4. BaselineFitter Filter O(N log N) FFT Gaussian convolution boundary & accuracy check
    {
        constexpr int nfft = 512;
        std::vector<double> pwr(nfft), freqs(nfft);
        for (int i = 0; i < nfft; ++i) {
            freqs[i] = 1419.5 + 2.4 * static_cast<double>(i) / nfft;
            double lin_db = -62.0 + 1.5 * std::cos(6.283185307 * static_cast<double>(i) / nfft);
            if (std::abs(freqs[i] - astro::HYDROGEN_REST_FREQ_MHZ) < 0.06) {
                lin_db += 8.0 * std::exp(-0.5 * std::pow((freqs[i] - astro::HYDROGEN_REST_FREQ_MHZ) / 0.015, 2));
            }
            pwr[i] = std::pow(10.0, lin_db * 0.1);
        }
        astro::AstroConfig cfg;
        cfg.sample_rate_hz = 2.4e6;
        cfg.center_freq_hz = 1420.5e6;
        cfg.baseline_method = astro::BaselineMethod::RippleFilter;
        cfg.filter_smoothing_sigma_khz = 160.0; // radius > 24 -> exercises DoubleFftPlan convolution
        auto bres = astro::BaselineFitter::fit(pwr, freqs, cfg);
        bool finite_ok = (bres.baseline_db.size() == static_cast<size_t>(nfft));
        for (int i = 0; i < nfft; ++i) {
            if (!std::isfinite(bres.baseline_db[i]) || std::abs(bres.baseline_db[i] + 62.0) > 2.5) {
                finite_ok = false;
                break;
            }
        }
        TEST_ASSERT(finite_ok, "Filter baseline O(N log N) FFT convolution produces accurate smooth baseline");
    }
}

static void test_dsp_pipeline_dbfs_and_instant_fft() {
    std::cout << "[Test 7] DspPipeline coherent full-scale dBFS & instantaneous FFT vs smoothed PSD...\n";
    constexpr int N = 4096;
    constexpr uint32_t FS = 2400000;
    constexpr uint64_t FC = 1420500000ULL;

    DspPipeline dsp;
    dsp.set_fft_size(N);
    dsp.set_averaging(0.15f); // Heavy smoothing on PSD

    // Full-scale coherent complex sinusoid at bin +512 (+300 kHz)
    std::vector<uint8_t> iq_tone1(2 * N);
    const double two_pi = 6.28318530717958647692;
    for (int i = 0; i < N; ++i) {
        double ph = two_pi * 512.0 * static_cast<double>(i) / static_cast<double>(N);
        int iv = static_cast<int>(std::lround(128.0 + 127.0 * std::cos(ph)));
        int qv = static_cast<int>(std::lround(128.0 + 127.0 * std::sin(ph)));
        iq_tone1[2 * i] = static_cast<uint8_t>(std::clamp(iv, 0, 255));
        iq_tone1[2 * i + 1] = static_cast<uint8_t>(std::clamp(qv, 0, 255));
    }

    dsp.feed_samples(iq_tone1.data(), iq_tone1.size());
    std::vector<float> psd_out;
    float psd_pk_freq = 0.0f, psd_pk_db = 0.0f;
    TEST_ASSERT(dsp.get_power_spectrum(psd_out, psd_pk_freq, psd_pk_db, FC, FS),
                "DspPipeline computed initial spectrum");

    const auto& fft_db = dsp.get_instant_fft_db();
    const auto& fft_mag = dsp.get_instant_fft_mag();
    TEST_ASSERT(fft_db.size() == static_cast<size_t>(N) && fft_mag.size() == static_cast<size_t>(N),
                "Instantaneous FFT dBFS and linear |X| buffers have full FFT size");
    TEST_ASSERT(dsp.get_instant_fft_peak_db() > -0.35f && dsp.get_instant_fft_peak_db() < 0.05f,
                "Coherent full-scale complex tone normalizes near 0.0 dBFS");
    TEST_ASSERT(dsp.get_instant_fft_peak_mag() > 0.96f && dsp.get_instant_fft_peak_mag() < 1.01f,
                "Instantaneous linear |X| normalizes near 1.0 FS");
    TEST_ASSERT(std::abs(dsp.get_instant_fft_peak_freq() * 1e-6f - 1420.800f) < 0.005f,
                "Instantaneous FFT peak frequency aligns with +300 kHz tone");

    // Step immediately to bin -512 (-300 kHz): instant FFT must follow immediately while heavy-smoothed PSD still retains prior tone power
    std::vector<uint8_t> iq_tone2(2 * N);
    for (int i = 0; i < N; ++i) {
        double ph = two_pi * (-512.0) * static_cast<double>(i) / static_cast<double>(N);
        int iv = static_cast<int>(std::lround(128.0 + 127.0 * std::cos(ph)));
        int qv = static_cast<int>(std::lround(128.0 + 127.0 * std::sin(ph)));
        iq_tone2[2 * i] = static_cast<uint8_t>(std::clamp(iv, 0, 255));
        iq_tone2[2 * i + 1] = static_cast<uint8_t>(std::clamp(qv, 0, 255));
    }
    dsp.feed_samples(iq_tone2.data(), iq_tone2.size());
    TEST_ASSERT(dsp.get_power_spectrum(psd_out, psd_pk_freq, psd_pk_db, FC, FS),
                "DspPipeline computed step-change spectrum");
    TEST_ASSERT(std::abs(dsp.get_instant_fft_peak_freq() * 1e-6f - 1420.200f) < 0.005f,
                "Instantaneous FFT jumps immediately to -300 kHz tone without lag");
    TEST_ASSERT(psd_out[2048 + 512] > fft_db[2048 + 512] + 5.0f,
                "Smoothed PSD retains memory at prior +300 kHz bin while instant FFT drops");
}

static void test_math_accuracy_and_solvers() {
    std::cout << "[Test 8] Math accuracy, noise unbiasedness, odd-NFFT shift & parametric recovery...\n";

    // 1. fast_10log10_scalar & AVX2 log2 power-of-two boundary continuity (< 0.002 dB)
    {
        float max_err_db = 0.0f;
        for (int exp2 = -36; exp2 <= 10; ++exp2) {
            for (int sub = -3; sub <= 3; ++sub) {
                float x = std::ldexp(1.0f + 0.0002f * static_cast<float>(sub), exp2);
                if (x <= 0.0f) continue;
                float approx = SimdDsp::fast_10log10_scalar(x);
                float exact = 10.0f * std::log10(x);
                max_err_db = std::max(max_err_db, std::abs(approx - exact));
            }
        }
        TEST_ASSERT(max_err_db < 0.002f, "fast_10log10_scalar accurate & continuous across 2^k (< 0.002 dB)");
    }

    // 2. DspPipeline linear-power EMA unbiasedness (no -2.506 dB Rayleigh log-averaging bias)
    {
        constexpr int N = 1024;
        DspPipeline dsp_raw, dsp_smooth;
        dsp_raw.set_fft_size(N);
        dsp_raw.set_averaging(1.0f);   // No EMA smoothing (gamma=1): single block noise floor
        dsp_smooth.set_fft_size(N);
        dsp_smooth.set_averaging(0.05f); // Heavy EMA smoothing: in linear power, E[P_smooth] == E[P_raw]

        // Deterministic PRNG white Gaussian-like I/Q noise (box-muller / LCG)
        uint32_t rng = 0x12345678u;
        auto next_u01 = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return (static_cast<double>(rng >> 8) + 0.5) / 16777216.0;
        };
        std::vector<uint8_t> iq_noise(2 * N * 64);
        for (size_t i = 0; i < N * 64; ++i) {
            double u1 = std::max(1e-7, next_u01());
            double u2 = next_u01();
            double r = 24.0 * std::sqrt(-2.0 * std::log(u1));
            double a = 6.28318530717958647692 * u2;
            int iv = static_cast<int>(std::lround(128.0 + r * std::cos(a)));
            int qv = static_cast<int>(std::lround(128.0 + r * std::sin(a)));
            iq_noise[2 * i] = static_cast<uint8_t>(std::clamp(iv, 1, 254));
            iq_noise[2 * i + 1] = static_cast<uint8_t>(std::clamp(qv, 1, 254));
        }

        // Accumulate mean linear power of raw single-block PSD vs heavily EMA-smoothed PSD
        double sum_lin_raw = 0.0;
        int cnt_raw = 0;
        std::vector<float> spec;
        float pk_f = 0.0f, pk_d = 0.0f;
        for (int b = 0; b < 64; ++b) {
            const uint8_t* blk = iq_noise.data() + static_cast<size_t>(b) * 2 * N;
            dsp_raw.feed_samples(blk, 2 * N);
            dsp_smooth.feed_samples(blk, 2 * N);
            if (dsp_raw.get_power_spectrum(spec, pk_f, pk_d, 1420500000ULL, 2400000U)) {
                for (float dbv : spec) sum_lin_raw += std::pow(10.0, static_cast<double>(dbv) * 0.1);
                cnt_raw += static_cast<int>(spec.size());
            }
            dsp_smooth.get_power_spectrum(spec, pk_f, pk_d, 1420500000ULL, 2400000U);
        }
        double mean_lin_raw_db = 10.0 * std::log10(sum_lin_raw / std::max(1, cnt_raw));
        double sum_lin_smooth = 0.0;
        for (float dbv : spec) sum_lin_smooth += std::pow(10.0, static_cast<double>(dbv) * 0.1);
        double mean_lin_smooth_db = 10.0 * std::log10(sum_lin_smooth / std::max<size_t>(1, spec.size()));
        TEST_ASSERT(std::abs(mean_lin_smooth_db - mean_lin_raw_db) < 0.35,
                    "DspPipeline linear-power EMA preserves unbiased mean noise power (no -2.51 dB log bias)");
    }

    // 3. Odd-NFFT StreamingProcessor FFT shift DC-bin placement (DC must map to nfft / 2 without OOB)
    {
        fs::path odd_tmp = fs::path("tests") / "test_odd_nfft.cs16";
        constexpr int odd_nfft = 1023;
        constexpr size_t total_cpx = odd_nfft * 8;
        {
            std::ofstream ofs(odd_tmp, std::ios::binary);
            std::vector<int16_t> iq(2 * total_cpx);
            for (size_t i = 0; i < total_cpx; ++i) {
                iq[2 * i] = 10000; // Pure DC offset (+0 Hz)
                iq[2 * i + 1] = 0;
            }
            ofs.write(reinterpret_cast<const char*>(iq.data()), static_cast<std::streamsize>(iq.size() * sizeof(int16_t)));
        }
        astro::AstroConfig cfg;
        cfg.center_freq_hz = 1420.5e6;
        cfg.sample_rate_hz = 2.4e6;
        cfg.fft_size = odd_nfft;
        cfg.verbose = false;
        astro::StreamingProcessor proc(odd_tmp.string(), cfg);
        auto res = proc.process();
        size_t max_bin = 0;
        for (size_t i = 1; i < res.averaged_power.size(); ++i) {
            if (res.averaged_power[i] > res.averaged_power[max_bin]) max_bin = i;
        }
        TEST_ASSERT(res.averaged_power.size() == static_cast<size_t>(odd_nfft) &&
                    max_bin == static_cast<size_t>(odd_nfft / 2),
                    "Odd-NFFT StreamingProcessor places DC accurately at nfft/2 without OOB");
        std::error_code ec;
        fs::remove(odd_tmp, ec);
    }

    // 4. Zero-mean pure noise baseline/metrics unbiasedness (no positive Jensen bias)
    {
        constexpr int n = 1024;
        std::vector<double> freq_mhz(n), vel_kms(n), cal_k(n), cal_db(n);
        uint32_t rng = 0x9E3779B9u;
        auto next_u01 = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return (static_cast<double>(rng >> 8) + 0.5) / 16777216.0;
        };
        astro::AstroConfig cfg;
        cfg.center_freq_hz = 1420.5e6;
        cfg.sample_rate_hz = 2.4e6;
        cfg.system_temp_k = 100.0;
        for (int i = 0; i < n; ++i) {
            double f_hz = (cfg.center_freq_hz - 0.5 * cfg.sample_rate_hz) +
                          cfg.sample_rate_hz * static_cast<double>(i) / static_cast<double>(n);
            freq_mhz[i] = f_hz * 1e-6;
            vel_kms[i] = astro::SPEED_OF_LIGHT_KMS * (astro::HYDROGEN_REST_FREQ_MHZ - freq_mhz[i]) / astro::HYDROGEN_REST_FREQ_MHZ;
            double u1 = std::max(1e-7, next_u01());
            double u2 = next_u01();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
            cal_k[i] = 1.0 * z + 0.2; // Add small 0.2 K offset for refine_local to remove
            cal_db[i] = 10.0 * std::log10(std::max(1e-6, 1.0 + cal_k[i] / cfg.system_temp_k));
        }
        double slope = 0.0, intercept = 0.0;
        astro::BaselineFitter::refine_local(vel_kms, cal_k, cal_db, cfg, slope, intercept);
        double mean_hi_k = 0.0;
        int cnt_hi = 0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(vel_kms[i]) <= 60.0) { mean_hi_k += cal_k[i]; ++cnt_hi; }
        }
        mean_hi_k /= std::max(1, cnt_hi);
        auto m = astro::Astrophysics::calculate_metrics(freq_mhz, vel_kms, cal_k, cal_db, cfg);
        TEST_ASSERT(std::abs(mean_hi_k) < 0.25, "refine_local linear-space subtraction is unbiased on pure noise");
        TEST_ASSERT(std::abs(m.integrated_flux_k_kms) < 15.0 && m.column_density_cm2 >= 0.0,
                    "Pure noise metrics are zero-mean without positive clamping bias");
    }

    // 5. Parametric recovery for DoubleGaussian, VoigtProfile, GaussHermite (h3, h4>0, h4<0), SpectralMoments
    {
        constexpr int n = 401;
        std::vector<double> vel(n), y(n);
        std::vector<uint8_t> win(n, 0);
        for (int i = 0; i < n; ++i) {
            vel[i] = -100.0 + 0.5 * static_cast<double>(i);
            if (std::abs(vel[i]) <= 80.0) win[i] = 1;
        }

        // 5a. DoubleGaussian: two distinct components at -22 km/s (12 K, sig=5) and +16 km/s (8 K, sig=6)
        for (int i = 0; i < n; ++i) {
            double c1 = 12.0 * std::exp(-0.5 * std::pow((vel[i] + 22.0) / 5.0, 2));
            double c2 =  8.0 * std::exp(-0.5 * std::pow((vel[i] - 16.0) / 6.0, 2));
            y[i] = c1 + c2;
        }
        auto dg = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::DoubleGaussian);
        TEST_ASSERT(dg.success && dg.has_secondary, "DoubleGaussian converged and identified two arms");
        double v_lo = std::min(dg.v_center_kms, dg.v2_center_kms);
        double v_hi = std::max(dg.v_center_kms, dg.v2_center_kms);
        TEST_ASSERT(std::abs(v_lo + 22.0) < 1.5 && std::abs(v_hi - 16.0) < 1.5 && dg.rmse_k < 0.2,
                    "DoubleGaussian recovered both arm velocities accurately");

        // 5b. VoigtProfile: eta = 0.55, sigma_g = 4.0, matched HWHM gamma_l = sigma_g * sqrt(2*ln(2))
        const double eta_true = 0.55;
        const double gamma_true = 4.0 * std::sqrt(2.0 * std::log(2.0));
        for (int i = 0; i < n; ++i) {
            double dv = vel[i] - 5.0;
            double g = std::exp(-0.5 * std::pow(dv / 4.0, 2));
            double l = 1.0 / (1.0 + std::pow(dv / gamma_true, 2));
            y[i] = 18.0 * ((1.0 - eta_true) * g + eta_true * l);
        }
        auto vp = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::VoigtProfile);
        TEST_ASSERT(vp.success && std::abs(vp.voigt_eta - eta_true) < 0.15 && vp.rmse_k < 0.2,
                    "VoigtProfile recovered mixing eta and profile accurately");

        // 5c. GaussHermite: test h3 asymmetry + h4 > 0 (narrower than Gaussian FWHM) & h4 < 0 (wider)
        auto eval_gh = [](double v, double A, double v0, double sig, double h3, double h4) {
            double w = (v - v0) / sig;
            double H3 = (2.0 * std::sqrt(2.0) * w * w * w - 3.0 * std::sqrt(2.0) * w) / std::sqrt(6.0);
            double w2 = w * w;
            double H4 = (4.0 * w2 * w2 - 12.0 * w2 + 3.0) / std::sqrt(24.0);
            return A * std::exp(-0.5 * w2) * (1.0 + h3 * H3 + h4 * H4);
        };
        for (int i = 0; i < n; ++i) y[i] = eval_gh(vel[i], 20.0, 4.0, 8.0, 0.12, 0.15);
        auto gh_pos = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::GaussHermite);
        const double gauss_fwhm_ref = 2.354820045 * gh_pos.sigma_kms;
        TEST_ASSERT(gh_pos.success && std::abs(gh_pos.gh_h3 - 0.12) < 0.05 && std::abs(gh_pos.gh_h4 - 0.15) < 0.05,
                    "GaussHermite recovered h3 > 0 and h4 > 0 accurately");
        TEST_ASSERT(gh_pos.fwhm_kms < gauss_fwhm_ref,
                    "GaussHermite h4 > 0 produces strictly narrower true numerical FWHM than 2.355*sigma");

        for (int i = 0; i < n; ++i) y[i] = eval_gh(vel[i], 20.0, 4.0, 8.0, -0.05, -0.14);
        auto gh_neg = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::GaussHermite);
        const double gauss_fwhm_neg_ref = 2.354820045 * gh_neg.sigma_kms;
        TEST_ASSERT(gh_neg.success && std::abs(gh_neg.gh_h4 + 0.14) < 0.05 && gh_neg.fwhm_kms > gauss_fwhm_neg_ref,
                    "GaussHermite h4 < 0 produces strictly wider true numerical FWHM than 2.355*sigma");

        // 5d. SpectralMoments: exact known Gaussian A=16, v0=9.5, sigma=6.0 -> M0 = 16*6*sqrt(2*pi) ~ 240.63
        for (int i = 0; i < n; ++i) y[i] = 16.0 * std::exp(-0.5 * std::pow((vel[i] - 9.5) / 6.0, 2));
        auto sm = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::SpectralMoments);
        double expected_m0 = 16.0 * 6.0 * std::sqrt(2.0 * 3.14159265358979323846);
        TEST_ASSERT(sm.success && std::abs(sm.moment0_k_kms - expected_m0) < 3.0 &&
                    std::abs(sm.moment1_kms - 9.5) < 0.3 &&
                    std::abs(sm.moment2_sigma_kms - 6.0) < 0.4,
                    "SpectralMoments recovered exact analytical M0, M1, and sqrt(M2)");

        // 5e. TripleGaussian: 3 distinct Galactic spiral arms at -30, +2, and +34 km/s
        for (int i = 0; i < n; ++i) {
            double c1 = 14.0 * std::exp(-0.5 * std::pow((vel[i] + 30.0) / 4.5, 2));
            double c2 = 10.0 * std::exp(-0.5 * std::pow((vel[i] -  2.0) / 5.0, 2));
            double c3 =  6.5 * std::exp(-0.5 * std::pow((vel[i] - 34.0) / 4.5, 2));
            y[i] = c1 + c2 + c3;
        }
        auto tg = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::TripleGaussian);
        TEST_ASSERT(tg.success && tg.has_secondary && tg.has_tertiary,
                    "TripleGaussian converged and identified all three kinematic arms");
        std::vector<double> arm_v = { tg.v_center_kms, tg.v2_center_kms, tg.v3_center_kms };
        std::sort(arm_v.begin(), arm_v.end());
        TEST_ASSERT(std::abs(arm_v[0] + 30.0) < 1.5 &&
                    std::abs(arm_v[1] -  2.0) < 1.5 &&
                    std::abs(arm_v[2] - 34.0) < 1.5 &&
                    tg.rmse_k < 0.25,
                    "TripleGaussian recovered all three spiral arm velocities and amplitudes accurately");

        // 5f. BusyFunction (Westmeier et al. 2014): double-horned profile with central trough
        for (int i = 0; i < n; ++i) {
            double dv = vel[i] - 5.0;
            double e1 = std::erf(0.35 * (22.0 + dv)) + 1.0;
            double e2 = std::erf(0.35 * (22.0 - dv)) + 1.0;
            y[i] = 0.25 * 10.0 * e1 * e2 * (1.0 + 0.0025 * dv * dv);
        }
        auto bf = astro::Astrophysics::fit_line_profile(vel, y, win, astro::LineAnalysisMethod::BusyFunction);
        TEST_ASSERT(bf.success &&
                    std::abs(bf.v_center_kms - 5.0) < 1.0 &&
                    std::abs(bf.busy_half_width_kms - 22.0) < 2.0 &&
                    bf.busy_trough_c > 0.001 &&
                    bf.rmse_k < 0.25,
                    "BusyFunction recovered double-horned half-width, centroid, and trough curvature");
    }

    // 6. SavitzkyGolay & StandingWave Continuum Baselines + Derived 21-cm Physical Metrics
    {
        constexpr int n = 2048;
        astro::AstroConfig cfg;
        cfg.center_freq_hz = 1420.405751768e6;
        cfg.sample_rate_hz = 2.4e6;
        cfg.fft_size = n;
        cfg.system_temp_k = 100.0;
        cfg.spin_temp_k = 150.0;

        std::vector<double> freqs_hz(n), freqs_mhz(n), true_base_db(n), raw_power(n);
        for (int i = 0; i < n; ++i) {
            freqs_hz[i] = (cfg.center_freq_hz - 0.5 * cfg.sample_rate_hz) +
                          cfg.sample_rate_hz * static_cast<double>(i) / static_cast<double>(n);
            freqs_mhz[i] = freqs_hz[i] * 1e-6;
            double x = (static_cast<double>(i) - 0.5 * n) / (0.5 * n);
            // Smooth quadratic passband + sinusoidal coaxial standing wave (14 cycles across passband)
            double ripple_db = 0.35 * std::cos(2.0 * 3.14159265358979323846 * 14.0 * i / n + 0.4)
                             + 0.12 * std::sin(2.0 * 3.14159265358979323846 * 23.0 * i / n - 0.7);
            true_base_db[i] = -20.0 - 0.8 * x * x + 0.15 * x + ripple_db;
        }
        auto vel_kms = astro::Astrophysics::compute_doppler_velocity(freqs_hz);
        for (int i = 0; i < n; ++i) {
            double hi_temp_k = 30.0 * std::exp(-0.5 * std::pow((vel_kms[i] + 10.0) / 8.0, 2));
            double base_lin = std::pow(10.0, true_base_db[i] * 0.1);
            raw_power[i] = base_lin * (1.0 + hi_temp_k / cfg.system_temp_k);
        }

        // Test StandingWave baseline recovery across the protected H I window
        cfg.baseline_method = astro::BaselineMethod::StandingWave;
        auto sw_res = astro::BaselineFitter::fit(raw_power, freqs_mhz, cfg);
        double sw_hi_rmse_db = 0.0;
        int hi_cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(freqs_mhz[i] - astro::HYDROGEN_REST_FREQ_MHZ) <= 0.19) {
                double err = sw_res.baseline_db[i] - true_base_db[i];
                sw_hi_rmse_db += err * err;
                ++hi_cnt;
            }
        }
        sw_hi_rmse_db = std::sqrt(sw_hi_rmse_db / std::max(1, hi_cnt));
        TEST_ASSERT(sw_hi_rmse_db < 0.08,
                    "StandingWave baseline models 2-tone sinusoidal coax ripple across H I window (< 0.08 dB)");

        // Test SavitzkyGolay baseline recovery on smooth quadratic continuum
        std::vector<double> sg_true_db(n), sg_raw_power(n);
        for (int i = 0; i < n; ++i) {
            double x = (static_cast<double>(i) - 0.5 * n) / (0.5 * n);
            sg_true_db[i] = -20.0 - 1.2 * x * x + 0.25 * x;
            double hi_temp_k = 25.0 * std::exp(-0.5 * std::pow((vel_kms[i] - 5.0) / 7.0, 2));
            double base_lin = std::pow(10.0, sg_true_db[i] * 0.1);
            sg_raw_power[i] = base_lin * (1.0 + hi_temp_k / cfg.system_temp_k);
        }
        cfg.baseline_method = astro::BaselineMethod::SavitzkyGolay;
        auto sg_res = astro::BaselineFitter::fit(sg_raw_power, freqs_mhz, cfg);
        double sg_hi_rmse_db = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(freqs_mhz[i] - astro::HYDROGEN_REST_FREQ_MHZ) <= 0.19) {
                double err = sg_res.baseline_db[i] - sg_true_db[i];
                sg_hi_rmse_db += err * err;
            }
        }
        sg_hi_rmse_db = std::sqrt(sg_hi_rmse_db / std::max(1, hi_cnt));
        TEST_ASSERT(sg_hi_rmse_db < 0.05,
                    "SavitzkyGolay degree-2 baseline preserves quadratic continuum across H I window (< 0.05 dB)");

        // Test derived 21-cm physical & kinematic metrics on exact synthetic profile (A=30 K, sigma=8 km/s, Ts=150 K)
        std::vector<double> cal_k(n), cal_db(n);
        for (int i = 0; i < n; ++i) {
            cal_k[i] = 30.0 * std::exp(-0.5 * std::pow((vel_kms[i] + 10.0) / 8.0, 2));
            cal_db[i] = 10.0 * std::log10(1.0 + cal_k[i] / cfg.system_temp_k);
        }
        cfg.line_method = astro::LineAnalysisMethod::GaussianLS;
        auto m = astro::Astrophysics::calculate_metrics(freqs_mhz, vel_kms, cal_k, cal_db, cfg);
        const double fwhm_true = astro::GAUSSIAN_SIGMA_TO_FWHM * 8.0; // 18.83856 km/s
        const double tk_expected = astro::HI_THERMAL_TEMP_FACTOR_K * fwhm_true * fwhm_true; // ~7756.3 K
        const double tau_expected = -std::log(1.0 - 30.0 / 150.0); // 0.22314355
        const double weq_expected = std::sqrt(2.0 * 3.14159265358979323846) * 8.0; // 20.053 km/s
        const double w20_w50_ratio = m.w20_kms / std::max(1e-6, m.w50_kms);

        TEST_ASSERT(std::abs(m.doppler_temp_max_k - tk_expected) / tk_expected < 0.015,
                    "Derived max kinetic temperature T_k,max matches theoretical 21.855 * FWHM^2");
        TEST_ASSERT(std::abs(m.peak_optical_depth - tau_expected) < 0.002,
                    "Derived peak optical depth tau_max matches -ln(1 - T_peak / T_spin)");
        TEST_ASSERT(m.column_density_thick_cm2 > m.column_density_cm2 * 1.07 &&
                    m.column_density_thick_cm2 < m.column_density_cm2 * 1.10,
                    "Optically-thick column density N_HI,thick accurately corrects for tau=0.223 self-absorption");
        TEST_ASSERT(std::abs(m.equivalent_width_kms - weq_expected) < 0.3,
                    "Equivalent width W_eq matches analytical sqrt(2*pi)*sigma");
        TEST_ASSERT(std::abs(m.w50_kms - fwhm_true) < 0.25 &&
                    std::abs(w20_w50_ratio - std::sqrt(std::log(5.0) / std::log(2.0))) < 0.02,
                    "Velocity widths W50 and W20 match theoretical Gaussian crossing ratio sqrt(ln 5 / ln 2)");
    }
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  RUNNING TEST SUITE: test_dsp_optimizations\n";
    std::cout << "======================================================================\n";

    test_simd_math_and_cs16_kernels();
    test_ring_buffer_wraparound();
    test_waterfall_history_resize();
    test_scientific_pipeline_solvers();
    test_streaming_fft_parity_and_sidecar_dedup();
    test_simd_fft_and_rfi_kernels();
    test_dsp_pipeline_dbfs_and_instant_fft();
    test_math_accuracy_and_solvers();

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "RESULTS: " << g_tests_passed << " / " << g_tests_run << " assertions PASSED.\n";
    if (g_tests_passed == g_tests_run) {
        std::cout << "[SUCCESS] test_dsp_optimizations PASSED (100%)\n";
        return 0;
    }
    std::cerr << "[FAILURE] Some DSP optimization assertions failed!\n";
    return 1;
}
