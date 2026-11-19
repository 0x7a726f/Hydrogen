#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <cmath>
#include "constants.h"

namespace astro {

enum class WindowType {
    Blackman,
    Hann,
    BlackmanHarris
};

enum class BaselineMethod {
    ALS,            // Asymmetric Least Squares / Whittaker smoothing
    Poly,           // Iterative sigma-clipped high-order polynomial
    RippleFilter,   // Ripple-tuned Gaussian smoothed background
    SavitzkyGolay,  // Moment-preserving Savitzky-Golay passband filter
    StandingWave    // Coaxial standing-wave reflection sinusoid + continuum solver
};

enum class LineAnalysisMethod {
    GaussianLS,      // 1-Component Levenberg-Marquardt Gaussian fit
    DoubleGaussian,  // 2-Component Levenberg-Marquardt Gaussian decomposition (multi-arm kinematics)
    TripleGaussian,  // 3-Component Levenberg-Marquardt Gaussian decomposition (3-arm / CNM+WNM)
    VoigtProfile,    // Pseudo-Voigt Levenberg-Marquardt profile (Gaussian thermal/turbulent core + Lorentzian damping wings)
    GaussHermite,    // Gauss-Hermite h3 (skewness) & h4 (kurtosis) asymmetric line profile
    BusyFunction,    // Westmeier (2014) generalized H I Busy Function (double-horn / flat-top / asymmetric)
    SpectralMoments  // Non-parametric Moment 0 / Moment 1 / Moment 2 + smoothed envelope
};

struct AstroConfig {
    // SDR & Hardware Settings
    double center_freq_hz = 1420.5e6;      // SDR LO Tuned Frequency (e.g. 1420.5 MHz)
    double sample_rate_hz = 2.4e6;         // SDR Sampling Rate (e.g. 2.4 MSPS)
    double system_temp_k = 100.0;          // Assumed system noise temperature (Kelvin)
    double spin_temp_k = DEFAULT_HI_SPIN_TEMP_K; // Assumed H I harmonic-mean spin temperature T_s (Kelvin)

    // DSP Settings
    int fft_size = 8192;                   // Number of FFT bins
    WindowType window = WindowType::Blackman;
    int chunk_blocks = 512;                // FFT blocks per streaming batch
    double waterfall_dt_s = 1.0;           // Integration time per waterfall row
    double max_duration_s = 0.0;           // 0.0 or negative = full observation

    // DC Excision & RFI Settings
    double dc_clear_hz = 5000.0;           // Bandwidth around 0 Hz IF to clear (+/- 2.5 kHz)
    double rfi_threshold_mad = 4.5;        // Outlier detection threshold (MAD units)

    // Baseline Estimation Settings
    BaselineMethod baseline_method = BaselineMethod::RippleFilter;
    LineAnalysisMethod line_method = LineAnalysisMethod::GaussianLS;
    double passband_edge_fraction = 0.08;  // Minimum outer skirts fraction (8%)
    bool auto_skirt_trim = true;           // Auto-detect steep IF filter roll-off skirts up to 22%
    double hi_protection_half_width_khz = 190.0; // Protection zone (+/- 190 kHz, ~40 km/s)
    double filter_smoothing_sigma_khz = 35.0;   // Gaussian/SG filter sigma in kHz
    double als_lambda = 200.0;             // Whittaker/ALS smoothness scale
    double als_p = 0.50;                   // ALS asymmetry (0.50 = symmetric Huber IRLS)
    int als_diff_order = 3;                // Whittaker penalty order (2 = cubic, 3 = quintic Hermite)
    double als_cutoff_khz = 250.0;         // Physical cutoff period for 3rd-order Whittaker [kHz]
    int poly_degree = 12;                  // Chebyshev polynomial degree
    int local_baseline_order = 1;          // Local baseline order (-1 = off, 0 = offset, 1 = linear, 2 = quad)
    double line_free_velocity_kms = 40.0;  // Inner line-free velocity threshold (|v| >= 40 km/s)
    double local_baseline_outer_kms = 160.0; // Outer velocity bound for local refinement / noise floor
    int standing_wave_max_components = 2;  // Max sinusoidal standing-wave modes (1..2)
    double standing_wave_min_period_mhz = 0.30; // Minimum standing-wave period [MHz]
    double detection_snr_threshold = 4.5;  // Matched-filter SNR threshold for confirmed H I detection
    bool enable_welch_overlap = true;      // 50% Welch overlap in StreamingProcessor (~2x sensitivity)

    // Output & Logging
    std::string output_dir = "output";
    bool verbose = true;
};

// Strict & lenient WindowType helpers
inline bool try_parse_window_type(const std::string& str, WindowType& out) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "hann" || s == "hanning") { out = WindowType::Hann; return true; }
    if (s == "blackmanharris" || s == "blackman_harris" || s == "blackman-harris" || s == "bh") { out = WindowType::BlackmanHarris; return true; }
    if (s == "blackman") { out = WindowType::Blackman; return true; }
    return false;
}

inline WindowType parse_window_type(const std::string& str) {
    WindowType w = WindowType::Blackman;
    if (try_parse_window_type(str, w)) return w;
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.find("hanning") != std::string::npos) return WindowType::Hann;
    return WindowType::Blackman;
}

inline std::string window_type_to_string(WindowType w) {
    switch (w) {
        case WindowType::Hann: return "Hann";
        case WindowType::BlackmanHarris: return "Blackman-Harris";
        default: return "Blackman";
    }
}

// Strict & lenient BaselineMethod helpers
inline bool try_parse_baseline_method(const std::string& str, BaselineMethod& out) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "als" || s == "whittaker") { out = BaselineMethod::ALS; return true; }
    if (s == "poly" || s == "polynomial" || s == "chebyshev") { out = BaselineMethod::Poly; return true; }
    if (s == "savgol" || s == "savitzky_golay" || s == "savitzky-golay" || s == "sg") { out = BaselineMethod::SavitzkyGolay; return true; }
    if (s == "standing_wave" || s == "standing-wave" || s == "sinusoid" || s == "sinusoidal" || s == "swr") { out = BaselineMethod::StandingWave; return true; }
    if (s == "ripple_filter" || s == "ripple-filter" || s == "ripple" || s == "filter") { out = BaselineMethod::RippleFilter; return true; }
    return false;
}

inline BaselineMethod parse_baseline_method(const std::string& str) {
    BaselineMethod m = BaselineMethod::RippleFilter;
    try_parse_baseline_method(str, m);
    return m;
}

inline std::string baseline_method_to_string(BaselineMethod m) {
    switch (m) {
        case BaselineMethod::ALS:           return "als";
        case BaselineMethod::Poly:          return "poly";
        case BaselineMethod::SavitzkyGolay: return "savitzky_golay";
        case BaselineMethod::StandingWave:  return "standing_wave";
        default:                            return "ripple_filter";
    }
}

inline std::string baseline_method_display_name(BaselineMethod m) {
    switch (m) {
        case BaselineMethod::ALS:           return "Whittaker / ALS (Quintic)";
        case BaselineMethod::Poly:          return "Chebyshev Polynomial (QR)";
        case BaselineMethod::SavitzkyGolay: return "Savitzky-Golay (Degree-4)";
        case BaselineMethod::StandingWave:  return "Standing-Wave Reflection Solver";
        default:                            return "Ripple Filter (Inpainted)";
    }
}

// Strict & lenient LineAnalysisMethod helpers
inline bool try_parse_line_analysis_method(const std::string& str, LineAnalysisMethod& out) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "gaussian" || s == "gaussian_ls" || s == "gaussian-ls" || s == "gauss" || s == "single") {
        out = LineAnalysisMethod::GaussianLS; return true;
    }
    if (s == "double_gaussian" || s == "double" || s == "2gauss" || s == "two_gaussian") {
        out = LineAnalysisMethod::DoubleGaussian; return true;
    }
    if (s == "triple_gaussian" || s == "triple" || s == "3gauss" || s == "three_gaussian" || s == "cnm_wnm") {
        out = LineAnalysisMethod::TripleGaussian; return true;
    }
    if (s == "voigt" || s == "voigt_profile" || s == "pseudovoigt") {
        out = LineAnalysisMethod::VoigtProfile; return true;
    }
    if (s == "gauss_hermite" || s == "hermite" || s == "gh" || s == "h3h4") {
        out = LineAnalysisMethod::GaussHermite; return true;
    }
    if (s == "busy" || s == "busy_function" || s == "busy-function" || s == "westmeier") {
        out = LineAnalysisMethod::BusyFunction; return true;
    }
    if (s == "moments" || s == "spectral_moments" || s == "nonparametric") {
        out = LineAnalysisMethod::SpectralMoments; return true;
    }
    return false;
}

inline LineAnalysisMethod parse_line_analysis_method(const std::string& str) {
    LineAnalysisMethod m = LineAnalysisMethod::GaussianLS;
    try_parse_line_analysis_method(str, m);
    return m;
}

inline std::string line_analysis_method_to_string(LineAnalysisMethod m) {
    switch (m) {
        case LineAnalysisMethod::DoubleGaussian:  return "double_gaussian";
        case LineAnalysisMethod::TripleGaussian:  return "triple_gaussian";
        case LineAnalysisMethod::VoigtProfile:    return "voigt_profile";
        case LineAnalysisMethod::GaussHermite:    return "gauss_hermite";
        case LineAnalysisMethod::BusyFunction:    return "busy_function";
        case LineAnalysisMethod::SpectralMoments: return "spectral_moments";
        default:                                  return "gaussian_ls";
    }
}

inline std::string line_analysis_method_display_name(LineAnalysisMethod m) {
    switch (m) {
        case LineAnalysisMethod::DoubleGaussian:  return "Double Gaussian (2-Component)";
        case LineAnalysisMethod::TripleGaussian:  return "Triple Gaussian (3-Arm / CNM+WNM)";
        case LineAnalysisMethod::VoigtProfile:    return "Pseudo-Voigt (Core + Wings)";
        case LineAnalysisMethod::GaussHermite:    return "Gauss-Hermite (h3/h4 Asym)";
        case LineAnalysisMethod::BusyFunction:    return "Busy Function (Double-Horn / Flat)";
        case LineAnalysisMethod::SpectralMoments: return "Spectral Moments (M0/M1/M2)";
        default:                                  return "Single Gaussian (L-M)";
    }
}

inline bool str_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

// Strict frequency/rate parser: returns false on unparseable, non-finite, or non-positive values.
inline bool try_parse_frequency_or_rate(const std::string& val_str, double& out_val, bool default_is_mhz = true, bool* out_had_unit = nullptr) {
    if (out_had_unit) *out_had_unit = false;
    std::string cleaned = val_str;
    // Strip trailing parenthetical annotations such as " (2400000 Hz)"
    size_t paren = cleaned.find('(');
    if (paren != std::string::npos) {
        cleaned = cleaned.substr(0, paren);
    }

    cleaned.erase(cleaned.begin(), std::find_if(cleaned.begin(), cleaned.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    cleaned.erase(std::find_if(cleaned.rbegin(), cleaned.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), cleaned.end());

    // Remove thousands separators
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ','), cleaned.end());
    if (cleaned.empty()) return false;

    std::string lower = cleaned;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    double mult = 1.0;
    bool has_unit = false;
    if (str_ends_with(lower, "ghz") || str_ends_with(lower, "gsps")) {
        mult = 1e9;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - (str_ends_with(lower, "ghz") ? 3 : 4));
    } else if (str_ends_with(lower, "mhz") || str_ends_with(lower, "msps")) {
        mult = 1e6;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - (str_ends_with(lower, "mhz") ? 3 : 4));
    } else if (str_ends_with(lower, "khz") || str_ends_with(lower, "ksps")) {
        mult = 1e3;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - (str_ends_with(lower, "khz") ? 3 : 4));
    } else if (str_ends_with(lower, "hz") || str_ends_with(lower, "sps")) {
        mult = 1.0;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - (str_ends_with(lower, "hz") ? 2 : 3));
    } else if (str_ends_with(lower, "g")) {
        mult = 1e9;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - 1);
    } else if (str_ends_with(lower, "m")) {
        mult = 1e6;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - 1);
    } else if (str_ends_with(lower, "k")) {
        mult = 1e3;
        has_unit = true;
        cleaned = cleaned.substr(0, cleaned.size() - 1);
    }

    // Trim trailing spaces before unit suffix
    cleaned.erase(std::find_if(cleaned.rbegin(), cleaned.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), cleaned.end());
    if (cleaned.empty()) return false;

    try {
        size_t pos = 0;
        double val = std::stod(cleaned, &pos);
        if (pos != cleaned.size() || !std::isfinite(val) || val <= 0.0) {
            return false;
        }
        if (out_had_unit) *out_had_unit = has_unit;
        if (has_unit) {
            out_val = val * mult;
            return std::isfinite(out_val) && out_val > 0.0;
        }
        out_val = (default_is_mhz && val < 10000.0) ? (val * 1e6) : val;
        return std::isfinite(out_val) && out_val > 0.0;
    } catch (...) {
        return false;
    }
}

inline bool try_parse_sample_rate(const std::string& val_str, double& out_val) {
    bool had_unit = false;
    double v = 0.0;
    if (!try_parse_frequency_or_rate(val_str, v, false, &had_unit)) {
        return false;
    }
    if (!had_unit && v > 0.0 && v < 100.0) {
        v *= 1e6;
    }
    out_val = v;
    return std::isfinite(out_val) && out_val > 0.0;
}

// Flexible frequency/rate parser with safe fallback
inline double parse_frequency_or_rate(const std::string& val_str, bool default_is_mhz = true) {
    double v = 0.0;
    if (try_parse_frequency_or_rate(val_str, v, default_is_mhz)) {
        return v;
    }
    return default_is_mhz ? 1420.5e6 : 2.4e6;
}

// Parse sample rate where suffixless values < 100.0 (e.g. "2.4") are interpreted as MSPS
inline double parse_sample_rate(const std::string& val_str) {
    double v = 0.0;
    if (try_parse_sample_rate(val_str, v)) {
        return v;
    }
    return 2.4e6;
}

} // namespace astro
