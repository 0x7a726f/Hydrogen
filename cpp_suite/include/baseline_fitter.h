#pragma once

#include <vector>
#include <cstdint>
#include "config.h"

namespace astro {

struct BaselineResults {
    std::vector<double> baseline_power;          // Linear power baseline matching raw input
    std::vector<double> baseline_db;             // In dB scale

    // Effective passband mask after automatic/configured IF skirt trimming
    std::vector<uint8_t> effective_passband_mask;
    int effective_edge_left_bins = 0;
    int effective_edge_right_bins = 0;

    // Off-line continuum flatness & quality diagnostics
    double off_line_rms_db = 0.0;
    double thermal_rms_db = 0.0;
    double baseline_quality_ratio = 1.0;         // off_line_rms_db / thermal_rms_db (~1.0 = thermal-limited)
    double residual_lag1_autocorr = 0.0;         // Lag-1 autocorrelation of off-line residuals

    // Standing-wave solver telemetry (populated when StandingWave is used)
    int standing_wave_components_used = 0;
    double standing_wave_period1_mhz = 0.0;
    double standing_wave_period2_mhz = 0.0;
};

class BaselineFitter {
public:
    // Stage 1: Fit instrumental baseline across the SDR passband
    static BaselineResults fit(
        const std::vector<double>& power,
        const std::vector<double>& freqs_mhz,
        const AstroConfig& config
    );

    // Stage 2: Refine local baseline in line-free velocity space using robust multiplicative IRLS
    static void refine_local(
        const std::vector<double>& vel_kms,
        std::vector<double>& calibrated_k,
        std::vector<double>& calibrated_db,
        const AstroConfig& config,
        double& out_slope,
        double& out_intercept,
        const std::vector<uint8_t>* effective_passband_mask = nullptr
    );

private:
    static std::vector<double> fit_als(
        const std::vector<double>& p_db,
        const std::vector<double>& freqs_mhz,
        const std::vector<uint8_t>& passband_mask,
        const std::vector<uint8_t>& hi_window,
        const AstroConfig& config
    );

    static std::vector<double> fit_poly(
        const std::vector<double>& p_db,
        const std::vector<double>& freqs_mhz,
        const std::vector<uint8_t>& passband_mask,
        const std::vector<uint8_t>& hi_window,
        const AstroConfig& config
    );

    static std::vector<double> fit_filter(
        const std::vector<double>& p_db,
        const std::vector<double>& freqs_mhz,
        const std::vector<uint8_t>& passband_mask,
        const std::vector<uint8_t>& hi_window,
        const AstroConfig& config
    );

    static std::vector<double> fit_savgol(
        const std::vector<double>& p_db,
        const std::vector<double>& freqs_mhz,
        const std::vector<uint8_t>& passband_mask,
        const std::vector<uint8_t>& hi_window,
        const AstroConfig& config
    );

    static std::vector<double> fit_standing_wave(
        const std::vector<double>& p_db,
        const std::vector<double>& freqs_mhz,
        const std::vector<uint8_t>& passband_mask,
        const std::vector<uint8_t>& hi_window,
        const AstroConfig& config,
        int* out_components_used = nullptr,
        double* out_period1_mhz = nullptr,
        double* out_period2_mhz = nullptr
    );
};

// Fast Pentadiagonal Linear System Solver for 2nd-order Whittaker / ALS (D2^T D2)
void solve_pentadiagonal_symmetric(
    int n,
    const double* d,    // main diagonal [0..n-1]
    const double* e,    // sub/super diagonal 1 [0..n-2]
    const double* f,    // sub/super diagonal 2 [0..n-3]
    const double* rhs,  // right-hand side [0..n-1]
    double* x           // solution output [0..n-1]
);

// Fast Heptadiagonal Linear System Solver for 3rd-order Whittaker (D3^T D3, quintic Hermite bridge)
void solve_heptadiagonal_symmetric(
    int n,
    const double* d,    // main diagonal [0..n-1]
    const double* e1,   // sub/super diagonal 1 [0..n-2]
    const double* e2,   // sub/super diagonal 2 [0..n-3]
    const double* e3,   // sub/super diagonal 3 [0..n-4]
    const double* rhs,  // right-hand side [0..n-1]
    double* x           // solution output [0..n-1]
);

} // namespace astro
