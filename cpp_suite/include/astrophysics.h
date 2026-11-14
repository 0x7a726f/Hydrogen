#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "config.h"

namespace astro {

struct GaussianFit {
    bool success = false;
    LineAnalysisMethod method = LineAnalysisMethod::GaussianLS;

    // Primary component / overall profile metrics
    double amplitude_k = 0.0;
    double v_center_kms = 0.0;
    double sigma_kms = 0.0;
    double fwhm_kms = 0.0;
    double rmse_k = 0.0;              // Residual RMS inside H I window [K]

    // Secondary component (populated for DoubleGaussian)
    bool has_secondary = false;
    double amp2_k = 0.0;
    double v2_center_kms = 0.0;
    double sigma2_kms = 0.0;
    double fwhm2_kms = 0.0;

    // Pseudo-Voigt parameters (populated for VoigtProfile)
    double voigt_eta = 0.0;           // Lorentzian mixing fraction in [0, 1]
    double lorentz_gamma_kms = 0.0;   // Lorentzian HWHM gamma [km/s]

    // Gauss-Hermite higher-order moments (populated for GaussHermite)
    double gh_h3 = 0.0;               // Skewness coefficient h3
    double gh_h4 = 0.0;               // Kurtosis coefficient h4

    // Non-parametric spectral moments (always computed for reference & SpectralMoments)
    double moment0_k_kms = 0.0;       // Integrated intensity M0 [K * km/s]
    double moment1_kms = 0.0;         // Intensity-weighted centroid velocity M1 [km/s]
    double moment2_sigma_kms = 0.0;   // Intensity-weighted dispersion sqrt(M2) [km/s]

    // Curves evaluated over the entire vel_kms axis
    std::vector<double> fitted_curve;     // Total composite model or smoothed profile
    std::vector<double> component1_curve; // Component 1 (for DoubleGaussian)
    std::vector<double> component2_curve; // Component 2 (for DoubleGaussian)
};

struct AstrophysicalMetrics {
    double peak_freq_mhz = 0.0;
    double peak_vel_kms = 0.0;
    double peak_temp_k = 0.0;
    double peak_power_db = 0.0;
    double noise_std_k = 0.0;
    double noise_std_db = 0.0;
    double snr = 0.0;
    double integrated_flux_k_kms = 0.0;
    double column_density_cm2 = 0.0;
    double fwhm_kms = 0.0;
    double sigma_kms = 0.0;
    GaussianFit gauss_fit;
    double lo_offset_khz = 0.0;
    int bin_separation = 0;
    int dc_bin = 0;
    int hi_bin = 0;
};

class Astrophysics {
public:
    // Convert frequency array (Hz) to Topocentric Doppler Radial Velocity in km/s:
    // v = c * (f_rest - f) / f_rest
    static std::vector<double> compute_doppler_velocity(const std::vector<double>& freqs_hz);

    // Fit 1D Gaussian profile to HI emission line using Levenberg-Marquardt
    static GaussianFit fit_gaussian(
        const std::vector<double>& vel_kms,
        const std::vector<double>& temp_k,
        const std::vector<bool>& hi_window
    );

    static GaussianFit fit_gaussian(
        const std::vector<double>& vel_kms,
        const std::vector<double>& temp_k,
        const std::vector<uint8_t>& hi_window
    );

    // Multi-method H I 21-cm line profile analysis dispatcher
    static GaussianFit fit_line_profile(
        const std::vector<double>& vel_kms,
        const std::vector<double>& temp_k,
        const std::vector<uint8_t>& hi_window,
        LineAnalysisMethod method
    );

    // Compute comprehensive astrophysical metrics
    static AstrophysicalMetrics calculate_metrics(
        const std::vector<double>& freqs_mhz,
        const std::vector<double>& vel_kms,
        const std::vector<double>& calibrated_k,
        const std::vector<double>& calibrated_db,
        const AstroConfig& config
    );
};

} // namespace astro
