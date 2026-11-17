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
    double passband_edge_fraction = 0.065; // Outer skirts fraction (6.5%)
    double hi_protection_half_width_khz = 190.0; // Protection zone (+/- 190 kHz, ~40 km/s)
    double filter_smoothing_sigma_khz = 35.0;   // Gaussian filter sigma in kHz
    double als_lambda = 2000.0;            // ALS smoothness parameter
    double als_p = 0.02;                   // ALS asymmetry parameter
    int poly_degree = 10;                  // Polynomial degree
    int local_baseline_order = 1;          // Local baseline order (1 = linear)
    double line_free_velocity_kms = 40.0;  // Radial velocity threshold (|v| > 40 km/s)

    // Output & Logging
    std::string output_dir = "output";
    bool verbose = true;
};

// String to WindowType helper
inline WindowType parse_window_type(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "hann" || s.find("hanning") != std::string::npos) return WindowType::Hann;
    if (s == "blackmanharris" || s == "blackman_harris" || s == "bh") return WindowType::BlackmanHarris;
    return WindowType::Blackman;
}

inline std::string window_type_to_string(WindowType w) {
    switch (w) {
        case WindowType::Hann: return "Hann";
        case WindowType::BlackmanHarris: return "Blackman-Harris";
        default: return "Blackman";
    }
}

// String to BaselineMethod helper
inline BaselineMethod parse_baseline_method(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "als" || s == "whittaker") return BaselineMethod::ALS;
    if (s == "poly" || s == "polynomial") return BaselineMethod::Poly;
    if (s == "savgol" || s == "savitzky_golay" || s == "savitzky-golay" || s == "sg")
        return BaselineMethod::SavitzkyGolay;
    if (s == "standing_wave" || s == "standing-wave" || s == "sinusoid" || s == "sinusoidal" || s == "swr")
        return BaselineMethod::StandingWave;
    return BaselineMethod::RippleFilter;
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
        case BaselineMethod::ALS:           return "Asymmetric Least Squares (ALS)";
        case BaselineMethod::Poly:          return "Sigma-Clipped Polynomial";
        case BaselineMethod::SavitzkyGolay: return "Savitzky-Golay (Moment-Preserving)";
        case BaselineMethod::StandingWave:  return "Standing-Wave Reflection Solver";
        default:                            return "Ripple Filter (Gaussian)";
    }
}

// String to LineAnalysisMethod helper
inline LineAnalysisMethod parse_line_analysis_method(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "double_gaussian" || s == "double" || s == "2gauss" || s == "two_gaussian")
        return LineAnalysisMethod::DoubleGaussian;
    if (s == "triple_gaussian" || s == "triple" || s == "3gauss" || s == "three_gaussian" || s == "cnm_wnm")
        return LineAnalysisMethod::TripleGaussian;
    if (s == "voigt" || s == "voigt_profile" || s == "pseudovoigt")
        return LineAnalysisMethod::VoigtProfile;
    if (s == "gauss_hermite" || s == "hermite" || s == "gh" || s == "h3h4")
        return LineAnalysisMethod::GaussHermite;
    if (s == "busy" || s == "busy_function" || s == "busy-function" || s == "westmeier")
        return LineAnalysisMethod::BusyFunction;
    if (s == "moments" || s == "spectral_moments" || s == "nonparametric")
        return LineAnalysisMethod::SpectralMoments;
    return LineAnalysisMethod::GaussianLS;
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

// Flexible frequency/rate parser: supports "1420.5", "1420.5M", "1420.5 MHz (1420500000 Hz)", "2.4MSPS", "2400000"
inline double parse_frequency_or_rate(const std::string& val_str, bool default_is_mhz = true) {
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

    std::string lower = cleaned;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

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

    try {
        double val = std::stod(cleaned);
        if (has_unit) {
            return val * mult;
        }
        if (default_is_mhz && val < 10000.0) {
            return val * 1e6;
        }
        return val;
    } catch (...) {
        return default_is_mhz ? 1420.5e6 : 2.4e6;
    }
}

// Parse sample rate where suffixless values < 100.0 (e.g. "2.4") are interpreted as MSPS
inline double parse_sample_rate(const std::string& val_str) {
    double v = parse_frequency_or_rate(val_str, false);
    if (v > 0.0 && v < 100.0) {
        return v * 1e6;
    }
    return v;
}

} // namespace astro
