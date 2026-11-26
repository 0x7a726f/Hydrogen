#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "constants.h"
#include "config.h"
#include "sdr_controller.h"
#include "cs16_writer.h"
#include "sidecar_reader.h"
#include "streaming_processor.h"
#include "dc_excision.h"
#include "rfi_filter.h"
#include "baseline_fitter.h"
#include "astrophysics.h"
#include "exporter.h"
#include "utf8_path.h"
#include <exception>

static std::atomic<bool> g_interrupted{false};

#ifdef _WIN32
static BOOL WINAPI ConsoleCtrlHandler(DWORD ctrlType) {
    if (ctrlType == CTRL_C_EVENT || ctrlType == CTRL_BREAK_EVENT || ctrlType == CTRL_CLOSE_EVENT) {
        g_interrupted.store(true);
        return TRUE;
    }
    return FALSE;
}
#else
static void PosixSignalHandler(int) {
    g_interrupted.store(true);
}
#endif

static bool parse_strict_int(const std::string& s, int& out) {
    if (s.empty()) return false;
    try {
        size_t pos = 0;
        long long v = std::stoll(s, &pos, 10);
        if (pos != s.size()) return false;
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_strict_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    try {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size() || !std::isfinite(v)) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool has_cs16_extension(const std::string& path) {
    if (path.size() < 5) return false;
    std::string tail = path.substr(path.size() - 5);
    std::transform(tail.begin(), tail.end(), tail.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return tail == ".cs16";
}

static void print_help(const char* prog) {
    std::cout << "Hydrogen Line (21 cm) SDR & Astrophysics Suite CLI\n\n"
              << "Usage:\n"
              << "  " << prog << " record  [options]   Record raw CS16 IQ samples from SDR\n"
              << "  " << prog << " analyze [options]   Analyze CS16 file and export calibrated outputs\n\n"
              << "Record Options:\n"
              << "  --fc <freq>               Center LO frequency (e.g. 1420.5M, 1420500000)\n"
              << "  --fs <rate>               Sample rate (e.g. 2.4M, 2.4, 2400000)\n"
              << "  -d, -t, --duration <sec>  Capture duration in seconds (default: 60)\n"
              << "  -g, --gain <dB>           Manual tuner gain in dB (default: 25.4)\n"
              << "  --auto-gain               Enable hardware automatic gain control (AGC)\n"
              << "  --ppm <int>               Oscillator frequency correction in PPM (default: 0)\n"
              << "  --bias-tee                Enable 4.5V LNA bias-tee\n"
              << "  -o, --output <path>       Output .cs16 file path\n"
              << "  --virtual                 Force Virtual SDR simulation mode\n"
              << "  -q, --quiet               Suppress periodic progress output\n\n"
              << "Analyze Options:\n"
              << "  FILE, -f, --file PATH     Input CS16 file (auto-discovered if omitted)\n"
              << "  --fc, --center-freq <f>   Center LO frequency override\n"
              << "  --fs, --sample-rate <r>   Sample rate override\n"
              << "  -n, --fft-size <bins>     FFT channels (even, >= 64; default: 8192)\n"
              << "  -d, -t, --duration <sec>  Max duration in seconds to process (default: entire file)\n"
              << "  --window <type>           Spectral window: blackman, hann, blackman_harris\n"
              << "  --baseline <method>       Baseline model: ripple_filter, als, poly, savitzky_golay, standing_wave\n"
              << "  --line-method <method>    Line model: gaussian_ls, double_gaussian, triple_gaussian, voigt, gauss_hermite, moments, busy_function\n"
              << "  --tsys <Kelvin>           Assumed system noise temperature (default: 100.0 K)\n"
              << "  --spin-temp <Kelvin>      Assumed H I spin temperature for optical depth (default: 150.0 K)\n"
              << "  --hi-window-khz <kHz>     H I protection half-width in kHz (default: 190.0)\n"
              << "  --edge-fraction <0..0.45> Minimum outer passband skirt fraction (default: 0.08)\n"
              << "  --no-auto-skirt           Disable automatic IF skirt detection\n"
              << "  --line-free-vel <km/s>    Inner line-free velocity bound |v| >= val (default: 40.0)\n"
              << "  --filter-sigma-khz <kHz>  Gaussian/SavGol filter sigma in kHz (default: 35.0)\n"
              << "  --als-lambda <val>        Whittaker/ALS smoothness scale (default: 200.0)\n"
              << "  --als-p <0..1>            ALS asymmetry parameter (default: 0.50)\n"
              << "  --poly-degree <int>       Chebyshev polynomial degree 2..20 (default: 12)\n"
              << "  --local-order <-1..2>     Local baseline order (-1=off, 0=const, 1=lin, 2=quad)\n"
              << "  --dc-clear-hz <Hz>        DC notch clear bandwidth around LO (default: 5000.0)\n"
              << "  --rfi-mad <sigma>         RFI outlier MAD threshold (default: 4.5)\n"
              << "  -o, --output, --output-dir <dir>  Output directory (default: output)\n"
              << "  -q, --quiet               Suppress progress bar and verbose status output\n\n"
              << "Examples:\n"
              << "  " << prog << " record --fc 1420.5M --fs 2.4M --duration 60 -o capture.cs16\n"
              << "  " << prog << " analyze capture.cs16 -o output\n";
}

int run_record(int argc, char* argv[], int start_idx) {
    double duration_s = 60.0;
    uint64_t center_freq_hz = SdrConstants::DEFAULT_CENTER_FREQ_HZ;
    uint32_t sample_rate_hz = SdrConstants::DEFAULT_SAMPLE_RATE_HZ;
    float gain_db = SdrConstants::DEFAULT_GAIN_DB;
    bool auto_gain = false;
    int ppm = SdrConstants::DEFAULT_PPM;
    bool bias_tee = false;
    std::string output_path;
    bool force_virtual = false;
    bool quiet = false;

    for (int i = start_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--duration" || arg == "-d" || arg == "-t") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            if (!parse_strict_double(argv[++i], duration_s) || duration_s < 0.0 || duration_s > 3.1536e7) {
                std::cerr << "Error: Invalid duration value (must be finite in [0, 31536000] s).\n";
                return 1;
            }
        } else if (arg == "--fc" || arg == "--center-freq") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double parsed = 0.0;
            if (!astro::try_parse_frequency_or_rate(argv[++i], parsed, true) || parsed < 1.0 || parsed > 6e9) {
                std::cerr << "Error: Invalid center frequency (expected in [1 Hz, 6 GHz]).\n";
                return 1;
            }
            center_freq_hz = static_cast<uint64_t>(std::llround(parsed));
        } else if (arg == "--fs" || arg == "--sample-rate") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double parsed = 0.0;
            if (!astro::try_parse_sample_rate(argv[++i], parsed) || parsed < 1.0 || parsed > 20e6) {
                std::cerr << "Error: Invalid sample rate (expected in [1 Hz, 20 MSPS]).\n";
                return 1;
            }
            sample_rate_hz = static_cast<uint32_t>(std::llround(parsed));
        } else if (arg == "--gain" || arg == "-g") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double gval = 0.0;
            if (!parse_strict_double(argv[++i], gval) || gval < -20.0 || gval > 100.0) {
                std::cerr << "Error: Invalid gain value (expected finite dB in [-20, 100]).\n";
                return 1;
            }
            gain_db = static_cast<float>(gval);
        } else if (arg == "--auto-gain") {
            auto_gain = true;
        } else if (arg == "--ppm") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            if (!parse_strict_int(argv[++i], ppm) || ppm < -500 || ppm > 500) {
                std::cerr << "Error: Invalid --ppm value (expected integer in [-500, 500]).\n";
                return 1;
            }
        } else if (arg == "--bias-tee") {
            bias_tee = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            output_path = argv[++i];
            if (output_path.empty()) {
                std::cerr << "Error: Output path for " << arg << " cannot be empty.\n";
                return 1;
            }
        } else if (arg == "--virtual") {
            force_virtual = true;
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: Unrecognized option for record: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        } else {
            std::cerr << "Error: Unexpected positional argument for record: " << arg << "\n";
            return 1;
        }
    }

    // Declare writer BEFORE sdr so sdr is always shut down and destroyed before writer
    Cs16Writer writer("recordings");
    SdrController sdr;

    sdr.add_sample_callback([&writer](const uint8_t* data, size_t len) {
        writer.feed_samples(data, len);
    });

    if (!sdr.initialize(force_virtual, 0)) {
        std::cerr << "Error: Failed to initialize SDR device.\n";
        return 1;
    }

    sdr.request_frequency(center_freq_hz).wait();
    sdr.request_sample_rate(sample_rate_hz).wait();
    sdr.request_gain(gain_db, auto_gain).wait();
    if (ppm != 0) sdr.request_ppm(ppm).wait();
    if (bias_tee) sdr.request_bias_tee(true).wait();

    const uint64_t actual_fc = sdr.get_current_frequency();
    const uint32_t actual_fs = sdr.get_current_sample_rate();
    const float actual_gain = sdr.get_current_gain();
    const bool actual_auto_gain = sdr.is_auto_gain();
    const int actual_ppm = sdr.get_current_ppm();
    const bool actual_bias_tee = sdr.is_bias_tee_enabled();

    if (!quiet) {
        std::cout << "\nHydrogen Line SDR Recorder\n"
                  << std::fixed << std::setprecision(3)
                  << "  Device:            " << sdr.get_device_name() << "\n"
                  << "  Center Frequency:  " << (static_cast<double>(actual_fc) / 1e6) << " MHz\n"
                  << std::setprecision(2)
                  << "  Sample Rate:       " << (static_cast<double>(actual_fs) / 1e6) << " MSPS\n"
                  << std::setprecision(1)
                  << "  Duration:          " << duration_s << " s\n"
                  << "  Gain:              " << (actual_auto_gain ? "AUTO (AGC)" : (std::to_string(actual_gain) + " dB")) << "\n"
                  << "  PPM Correction:    " << actual_ppm << "\n"
                  << "  Bias-Tee:          " << (actual_bias_tee ? "Enabled" : "Disabled") << "\n";
    }

    std::string actual_file = writer.start_recording(
        actual_fs, actual_fc, sdr.get_device_name(), actual_gain, actual_bias_tee, output_path, duration_s
    );
    if (actual_file.empty()) {
        std::cerr << "Error: Failed to open output CS16 file for recording.\n";
        sdr.clear_sample_callbacks();
        sdr.shutdown();
        return 1;
    }

    if (!sdr.start_streaming()) {
        std::cerr << "Error: Failed to start SDR streaming.\n";
        sdr.clear_sample_callbacks();
        writer.stop_recording();
        sdr.shutdown();
        return 1;
    }

    if (!quiet) {
        std::cout << "Recording started: " << actual_file << "\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    while (!g_interrupted.load() && writer.is_recording()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        auto stats = writer.get_stats();

        if (!quiet) {
            std::cout << "\r[REC] " << std::fixed << std::setprecision(1) << el << "/" << duration_s << "s"
                      << " | Size: " << stats.size_str
                      << " | Samples: " << (static_cast<double>(stats.samples_written) / 1000000.0) << "M"
                      << " | Drops: " << stats.dropped_blocks
                      << " | Ring: " << std::setprecision(1) << sdr.get_ring_buffer_fill_pct() << "%"
                      << std::flush;
        }

        if (duration_s > 0.0 && el >= duration_s) break;
    }
    if (!quiet) std::cout << "\n";

    sdr.stop_streaming();
    sdr.clear_sample_callbacks();
    writer.stop_recording();
    sdr.shutdown();

    if (writer.has_io_error()) {
        std::cerr << "Error: Disk I/O write error occurred while recording to: " << actual_file << "\n";
        return 1;
    }

    if (g_interrupted.load()) {
        std::cerr << "Recording interrupted by user (partial capture saved: " << actual_file << ").\n";
        return 130;
    }

    if (!quiet) {
        std::cout << "\nRecording completed successfully!\n"
                  << "File saved: " << actual_file << "\n"
                  << "Sidecar metadata (.json, .txt) written alongside binary.\n";
    }
    return 0;
}

int run_analyze(int argc, char* argv[], int start_idx) {
    using namespace astro;

    AstroConfig config;
    std::string filepath;
    std::string fc_str;
    std::string fs_str;
    double parsed_fc_override = 0.0;
    double parsed_fs_override = 0.0;
    bool has_fc_override = false;
    bool has_fs_override = false;
    bool quiet = false;

    for (int i = start_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            filepath = argv[++i];
            if (filepath.empty()) {
                std::cerr << "Error: File path for " << arg << " cannot be empty.\n";
                return 1;
            }
        } else if (arg == "--fc" || arg == "--center-freq") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            fc_str = argv[++i];
            if (!try_parse_frequency_or_rate(fc_str, parsed_fc_override, true) ||
                parsed_fc_override <= 0.0 || parsed_fc_override > 1e12) {
                std::cerr << "Error: Invalid center frequency for " << arg << ": '" << fc_str << "'\n";
                return 1;
            }
            has_fc_override = true;
        } else if (arg == "--fs" || arg == "--sample-rate") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            fs_str = argv[++i];
            if (!try_parse_sample_rate(fs_str, parsed_fs_override) ||
                parsed_fs_override <= 0.0 || parsed_fs_override > 1e10) {
                std::cerr << "Error: Invalid sample rate for " << arg << ": '" << fs_str << "'\n";
                return 1;
            }
            has_fs_override = true;
        } else if (arg == "-n" || arg == "--fft-size") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            int v = 0;
            if (!parse_strict_int(argv[++i], v) || v < 64 || (v % 2) != 0 || v > 1048576) {
                std::cerr << "Error: --fft-size must be an even integer in [64, 1048576].\n";
                return 1;
            }
            config.fft_size = v;
        } else if (arg == "-d" || arg == "-t" || arg == "--duration") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v < 0.0 || v > 3.1536e7) {
                std::cerr << "Error: Invalid duration for " << arg << " (must be finite in [0, 31536000] s).\n";
                return 1;
            }
            config.max_duration_s = v;
        } else if (arg == "--window") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            std::string w_str = argv[++i];
            WindowType wt = WindowType::Blackman;
            if (!try_parse_window_type(w_str, wt)) {
                std::cerr << "Error: Unrecognized --window '" << w_str
                          << "' (expected: blackman, hann, blackman_harris).\n";
                return 1;
            }
            config.window = wt;
        } else if (arg == "--baseline") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            std::string b_str = argv[++i];
            BaselineMethod bm = BaselineMethod::RippleFilter;
            if (!try_parse_baseline_method(b_str, bm)) {
                std::cerr << "Error: Unrecognized --baseline method '" << b_str
                          << "' (expected: ripple_filter, als, poly, savitzky_golay, standing_wave).\n";
                return 1;
            }
            config.baseline_method = bm;
        } else if (arg == "--line-method" || arg == "--fit") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            std::string lm_str = argv[++i];
            LineAnalysisMethod lm = LineAnalysisMethod::GaussianLS;
            if (!try_parse_line_analysis_method(lm_str, lm)) {
                std::cerr << "Error: Unrecognized --line-method '" << lm_str
                          << "' (expected: gaussian_ls, double_gaussian, triple_gaussian, voigt, gauss_hermite, busy_function, moments).\n";
                return 1;
            }
            config.line_method = lm;
        } else if (arg == "--tsys") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 1e6) {
                std::cerr << "Error: --tsys must be a finite number in (0, 1e6] K.\n";
                return 1;
            }
            config.system_temp_k = v;
        } else if (arg == "--spin-temp" || arg == "--tspin") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 1e6) {
                std::cerr << "Error: --spin-temp must be a finite number in (0, 1e6] K.\n";
                return 1;
            }
            config.spin_temp_k = v;
        } else if (arg == "--hi-window-khz") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 5000.0) {
                std::cerr << "Error: --hi-window-khz must be finite in (0, 5000].\n";
                return 1;
            }
            config.hi_protection_half_width_khz = v;
        } else if (arg == "--edge-fraction") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v < 0.0 || v > 0.45) {
                std::cerr << "Error: --edge-fraction must be finite in [0.0, 0.45].\n";
                return 1;
            }
            config.passband_edge_fraction = v;
        } else if (arg == "--no-auto-skirt") {
            config.auto_skirt_trim = false;
        } else if (arg == "--line-free-vel") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v < 0.0 || v > 1000.0) {
                std::cerr << "Error: --line-free-vel must be finite in [0, 1000] km/s.\n";
                return 1;
            }
            config.line_free_velocity_kms = v;
        } else if (arg == "--filter-sigma-khz") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 5000.0) {
                std::cerr << "Error: --filter-sigma-khz must be finite in (0, 5000] kHz.\n";
                return 1;
            }
            config.filter_smoothing_sigma_khz = v;
        } else if (arg == "--als-lambda") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 1e12) {
                std::cerr << "Error: --als-lambda must be finite in (0, 1e12].\n";
                return 1;
            }
            config.als_lambda = v;
        } else if (arg == "--als-p") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v >= 1.0) {
                std::cerr << "Error: --als-p must be finite in (0.0, 1.0).\n";
                return 1;
            }
            config.als_p = v;
        } else if (arg == "--poly-degree") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            int v = 0;
            if (!parse_strict_int(argv[++i], v) || v < 2 || v > 20) {
                std::cerr << "Error: --poly-degree must be an integer in [2, 20].\n";
                return 1;
            }
            config.poly_degree = v;
        } else if (arg == "--local-order") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            int v = 0;
            if (!parse_strict_int(argv[++i], v) || v < -1 || v > 2) {
                std::cerr << "Error: --local-order must be an integer in [-1, 2].\n";
                return 1;
            }
            config.local_baseline_order = v;
        } else if (arg == "--dc-clear-hz") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v < 0.0 || v > 1e8) {
                std::cerr << "Error: --dc-clear-hz must be finite in [0, 1e8] Hz.\n";
                return 1;
            }
            config.dc_clear_hz = v;
        } else if (arg == "--rfi-mad") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            double v = 0.0;
            if (!parse_strict_double(argv[++i], v) || v <= 0.0 || v > 1e4) {
                std::cerr << "Error: --rfi-mad must be finite in (0, 10000].\n";
                return 1;
            }
            config.rfi_threshold_mad = v;
        } else if (arg == "-o" || arg == "--output" || arg == "--output-dir") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << "\n";
                return 1;
            }
            config.output_dir = argv[++i];
            if (config.output_dir.empty()) {
                std::cerr << "Error: Output directory for " << arg << " cannot be empty.\n";
                return 1;
            }
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
            config.verbose = false;
        } else if (!arg.empty() && arg[0] != '-') {
            if (filepath.empty()) {
                filepath = arg;
            } else {
                std::cerr << "Warning: Extra positional argument ignored: " << arg << "\n";
            }
        } else {
            std::cerr << "Error: Unrecognized option for analyze: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    // Auto-discover if no file provided
    if (filepath.empty()) {
        std::vector<std::string> search_dirs = {
            ".",
            "recordings",
            "bin/recordings",
            "../recordings",
            "../bin/recordings"
        };
        auto discovered = SidecarReader::scan_cs16_files(search_dirs);
        if (!discovered.empty()) {
            if (!quiet) {
                std::cout << "\nDiscovered CS16 files in workspace:\n";
                for (size_t i = 0; i < std::min<size_t>(discovered.size(), 5); ++i) {
                    std::error_code ec;
                    uint64_t sz = std::filesystem::file_size(astro_fs::utf8_path(discovered[i]), ec);
                    std::cout << "  [" << (i + 1) << "] " << discovered[i]
                              << " (" << std::fixed << std::setprecision(1) << static_cast<double>(sz) / (1024.0 * 1024.0) << " MB)\n";
                }
                std::cout << "Selecting newest capture: " << discovered[0] << "\n\n";
            }
            filepath = discovered[0];
        } else {
            std::cerr << "Error: No CS16 files found. Please provide a path using: hydrogen_cli analyze <file.cs16>\n";
            return 1;
        }
    }

    {
        std::error_code ec;
        const std::filesystem::path fpath = astro_fs::utf8_path(filepath);
        if (!std::filesystem::exists(fpath, ec)) {
            std::cerr << "Error: Specified file does not exist: " << filepath << "\n";
            return 1;
        }
        if (!std::filesystem::is_regular_file(fpath, ec)) {
            std::cerr << "Error: Specified path is not a regular file: " << filepath << "\n";
            return 1;
        }
    }

    // Auto-detect and parse sidecar
    ObservationMetadata sidecar = SidecarReader::parse(filepath);
    if (sidecar.found_sidecar) {
        SidecarReader::apply_to_config(sidecar, config);
        if (!quiet) {
            std::cout << "  [Metadata] Sidecar auto-detected: " << sidecar.sidecar_source << "\n"
                      << "             Center Frequency: " << (config.center_freq_hz / 1e6) << " MHz\n"
                      << "             Sample Rate:      " << (config.sample_rate_hz / 1e6) << " MSPS\n";
        }
    }

    if (has_fc_override) config.center_freq_hz = parsed_fc_override;
    if (has_fs_override) config.sample_rate_hz = parsed_fs_override;
    if (config.center_freq_hz <= 0.0 || config.sample_rate_hz <= 0.0) {
        std::cerr << "Error: Center frequency and sample rate must be positive.\n";
        return 1;
    }

    if (!quiet) {
        std::cout << "\nAstrophysical DSP Analysis Pipeline\n"
                  << "  Target CS16:       " << filepath << "\n"
                  << "  Center LO:         " << (config.center_freq_hz / 1e6) << " MHz\n"
                  << "  Sample Rate:       " << (config.sample_rate_hz / 1e6) << " MSPS\n"
                  << "  FFT Channels:      " << config.fft_size << " (" << window_type_to_string(config.window) << ")\n"
                  << "  Baseline Method:   " << baseline_method_to_string(config.baseline_method) << "\n"
                  << "  Line Model:        " << line_analysis_method_to_string(config.line_method) << "\n"
                  << "  System Temp Tsys:  " << config.system_temp_k << " K\n"
                  << "  Spin Temp Tspin:   " << config.spin_temp_k << " K\n"
                  << "  Output Directory:  " << config.output_dir << "\n"
                  << "----------------------------------------------------------------------\n";
    }

    // 1. Streaming accumulation (wired to Ctrl-C cancel token)
    if (!quiet) std::cout << "--> Accumulating FFT power spectra...\n";
    StreamingProcessor processor(filepath, config);
    processor.set_cancel_token(&g_interrupted);
    StreamingResults stream = processor.process(
        quiet ? ProgressCallback(nullptr) : ProgressCallback([](double pct, double speed, double eta) {
            std::cout << "\r[Progress] " << std::fixed << std::setprecision(1) << pct << "%"
                      << " | Throughput: " << speed << " MB/s"
                      << " | ETA: " << eta << "s   " << std::flush;
        })
    );

    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }
    if (stream.averaged_power.empty() || stream.freq_mhz.empty()) {
        std::cerr << "Error: Failed to process any valid FFT blocks from input file: " << filepath << "\n";
        return 1;
    }

    if (!quiet) {
        std::cout << "\nProcessed " << stream.total_samples << " complex samples in "
                  << std::fixed << std::setprecision(2) << stream.elapsed_processing_time_s << "s ("
                  << stream.throughput_mb_s << " MB/s)\n";
    }

    // 2. DC Notch Excision
    if (!quiet) std::cout << "--> Excising center DC local oscillator spike...\n";
    DCExcisionInfo dc_info;
    std::vector<double> dc_clean = DCExcision::excise(stream.averaged_power, config, dc_info);
    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }

    // 3. RFI Mitigation
    if (!quiet) std::cout << "--> Mitigating RFI spikes...\n";
    int num_rfi = 0;
    std::vector<double> rfi_clean = RFIFilter::mitigate(dc_clean, stream.freq_mhz, config, num_rfi);
    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }

    // 4. Baseline Modeling
    if (!quiet) std::cout << "--> Fitting baseline passband (" << baseline_method_to_string(config.baseline_method) << ")...\n";
    BaselineResults baseline = BaselineFitter::fit(rfi_clean, stream.freq_mhz, config);
    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }

    // 5. Scientific Calibration
    if (!quiet) std::cout << "--> Calibrating to Kelvin and computing Doppler velocities...\n";
    size_t nfft = stream.freq_mhz.size();
    std::vector<double> cal_k(nfft);
    std::vector<double> cal_db(nfft);
    std::vector<double> v_kms = Astrophysics::compute_doppler_velocity(stream.freq_hz);

    for (size_t i = 0; i < nfft; ++i) {
        double base_safe = std::max(1e-12, baseline.baseline_power[i]);
        cal_k[i] = config.system_temp_k * (rfi_clean[i] - baseline.baseline_power[i]) / base_safe;
        cal_db[i] = 10.0 * std::log10(std::max(1e-6, rfi_clean[i] / base_safe));
    }

    double slope = 0.0, intercept = 0.0;
    const std::vector<uint8_t>* pb_mask_ptr = baseline.effective_passband_mask.empty()
        ? nullptr : &baseline.effective_passband_mask;
    BaselineFitter::refine_local(v_kms, cal_k, cal_db, config, slope, intercept, pb_mask_ptr);
    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }

    // 6. Astrophysics Metrics & Line Curve Fit
    if (!quiet) std::cout << "--> Computing astrophysical metrics and line model (" << line_analysis_method_to_string(config.line_method) << ")...\n";
    AstrophysicalMetrics metrics = Astrophysics::calculate_metrics(
        stream.freq_mhz, v_kms, cal_k, cal_db, config, pb_mask_ptr
    );
    if (g_interrupted.load()) {
        std::cerr << "\nAnalysis interrupted by user.\n";
        return 130;
    }

    // 7. Structured Export
    if (!quiet) std::cout << "--> Exporting analysis results...\n";
    FileMetadata meta;
    meta.filepath = filepath;
    meta.filesize = stream.file_size_bytes;
    meta.total_samples = stream.total_samples;
    meta.duration_s = stream.total_duration_s;
    meta.processing_time_s = stream.elapsed_processing_time_s;
    meta.throughput_mb_s = stream.throughput_mb_s;

    bool export_ok = Exporter::export_all_artifacts(
        config.output_dir, filepath, config, meta, stream,
        rfi_clean, baseline.baseline_power, cal_k, cal_db, metrics
    );
    if (!export_ok) {
        std::cerr << "Error: Failed to export one or more analysis artifacts to: " << config.output_dir << "\n";
        return 1;
    }

    // 8. Print Executive Summary
    if (!quiet) {
        std::cout << "\nAstrophysical Results Summary\n"
                  << "  Detection Status:       " << (metrics.is_detected ? "CONFIRMED" : "NOT DETECTED")
                  << " (Matched SNR: " << std::fixed << std::setprecision(2) << metrics.snr_matched << " sigma)\n"
                  << std::fixed << std::setprecision(4)
                  << "  Peak Frequency:         " << metrics.peak_freq_mhz << " MHz\n"
                  << std::setprecision(2)
                  << "  Radial Velocity:        " << metrics.peak_vel_kms << " km/s\n"
                  << "  Peak Brightness Temp:   " << metrics.peak_temp_k << " K (" << metrics.peak_power_db << " dB)\n"
                  << "  Noise Floor RMS:        " << metrics.noise_std_k << " K (Baseline Quality: "
                  << metrics.baseline_quality_ratio << "x thermal)\n"
                  << "  Detection SNR:          " << metrics.snr << " sigma (1-bin: " << metrics.snr_single_bin
                  << ", integ: " << metrics.snr_integrated << ")\n"
                  << "  Line FWHM Width:        " << metrics.fwhm_kms << " km/s\n"
                  << "  Profile Widths W50/W20: " << metrics.w50_kms << " / " << metrics.w20_kms << " km/s\n"
                  << "  Equivalent Width W_eq:  " << metrics.equivalent_width_kms << " km/s\n"
                  << "  Max Kinetic Temp T_k:   " << std::setprecision(0) << metrics.doppler_temp_max_k << " K\n"
                  << "  Peak Optical Depth tau: " << std::setprecision(4) << metrics.peak_optical_depth << "\n"
                  << std::setprecision(2)
                  << "  Integrated Flux Area:   " << metrics.integrated_flux_k_kms << " +/- "
                  << metrics.integrated_flux_err_k_kms << " K*km/s\n"
                  << "  H I Column Density:     " << std::scientific << metrics.column_density_cm2
                  << " +/- " << metrics.column_density_err_cm2 << " atoms/cm^2 (optically thin)\n"
                  << "  H I Column (Thick):     " << std::scientific << metrics.column_density_thick_cm2 << " atoms/cm^2\n"
                  << std::fixed
                  << "  Output written to:      " << config.output_dir << "/\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
#else
    std::signal(SIGINT, PosixSignalHandler);
    std::signal(SIGTERM, PosixSignalHandler);
#endif

    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    try {
        std::string cmd = argv[1];
        std::transform(cmd.begin(), cmd.end(), cmd.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            print_help(argv[0]);
            return 0;
        }

        if (cmd == "record") {
            return run_record(argc, argv, 2);
        } else if (cmd == "analyze") {
            return run_analyze(argc, argv, 2);
        } else {
            // Auto-detect mode based on flags or input file
            bool has_record_flag = false;
            bool has_analyze_flag = false;
            bool is_cs16_file = false;

            for (int i = 1; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-o" || arg == "--output" || arg == "--output-dir") {
                    ++i;
                    continue;
                }
                if (arg == "-q" || arg == "--quiet") {
                    continue;
                }
                if (arg == "-f" || arg == "--file") {
                    has_analyze_flag = true;
                    ++i;
                    continue;
                }
                if (arg == "--baseline" || arg == "--tsys" || arg == "--spin-temp" || arg == "--tspin" ||
                    arg == "-n" || arg == "--fft-size" || arg == "--line-method" || arg == "--fit" ||
                    arg == "--hi-window-khz" || arg == "--edge-fraction" || arg == "--line-free-vel" ||
                    arg == "--filter-sigma-khz" || arg == "--als-lambda" || arg == "--als-p" ||
                    arg == "--poly-degree" || arg == "--local-order" || arg == "--window" ||
                    arg == "--dc-clear-hz" || arg == "--rfi-mad") {
                    has_analyze_flag = true;
                    ++i;
                    continue;
                }
                if (arg == "--no-auto-skirt") {
                    has_analyze_flag = true;
                    continue;
                }
                if (arg == "--gain" || arg == "-g" || arg == "--ppm") {
                    has_record_flag = true;
                    ++i;
                    continue;
                }
                if (arg == "--bias-tee" || arg == "--virtual" || arg == "--auto-gain") {
                    has_record_flag = true;
                    continue;
                }
                if (arg == "--fc" || arg == "--center-freq" || arg == "--fs" || arg == "--sample-rate" ||
                    arg == "-d" || arg == "-t" || arg == "--duration") {
                    ++i;
                    continue;
                }
                if (has_cs16_extension(arg)) {
                    is_cs16_file = true;
                }
            }

            std::error_code cli_ec;
            if (has_record_flag && !is_cs16_file && !has_analyze_flag) {
                return run_record(argc, argv, 1);
            } else if (is_cs16_file || has_analyze_flag || std::filesystem::exists(astro_fs::utf8_path(argv[1]), cli_ec)) {
                return run_analyze(argc, argv, 1);
            } else if (cmd == "--fc" || cmd == "--center-freq" || cmd == "--fs" || cmd == "--sample-rate" ||
                       cmd == "-d" || cmd == "-t" || cmd == "--duration" ||
                       cmd == "-o" || cmd == "--output" || cmd == "--output-dir") {
                return run_record(argc, argv, 1);
            } else {
                std::cerr << "Error: Unknown subcommand or file '" << argv[1] << "'.\n";
                print_help(argv[0]);
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Error: An unexpected fatal error occurred.\n";
        return 1;
    }
}
