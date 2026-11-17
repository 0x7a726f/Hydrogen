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
    double fwhm_left_kms = 0.0;
    double fwhm_right_kms = 0.0;
    double fwhm_half_temp_k = 0.0;
    double rmse_k = 0.0;              // Residual RMS inside H I window [K]

    // Secondary component (populated for DoubleGaussian & TripleGaussian)
    bool has_secondary = false;
    double amp2_k = 0.0;
    double v2_center_kms = 0.0;
    double sigma2_kms = 0.0;
    double fwhm2_kms = 0.0;

    // Tertiary component (populated for TripleGaussian)
    bool has_tertiary = false;
    double amp3_k = 0.0;
    double v3_center_kms = 0.0;
    double sigma3_kms = 0.0;
    double fwhm3_kms = 0.0;

    // Pseudo-Voigt parameters (populated for VoigtProfile)
    double voigt_eta = 0.0;           // Lorentzian mixing fraction in [0, 1]
    double lorentz_gamma_kms = 0.0;   // Lorentzian HWHM gamma [km/s]

    // Gauss-Hermite higher-order moments (populated for GaussHermite)
    double gh_h3 = 0.0;               // Skewness coefficient h3
    double gh_h4 = 0.0;               // Kurtosis coefficient h4

    // Generalized Busy Function parameters (populated for BusyFunction)
    double busy_half_width_kms = 0.0; // Error-function flank half-width w [km/s]
    double busy_flank_b1 = 0.0;       // Blueward flank steepness b1 [(km/s)^-1]
    double busy_flank_b2 = 0.0;       // Redward flank steepness b2 [(km/s)^-1]
    double busy_trough_c = 0.0;       // Central trough polynomial curvature c [(km/s)^-2]

    // Non-parametric spectral moments (always computed for reference & SpectralMoments)
    double moment0_k_kms = 0.0;       // Integrated intensity M0 [K * km/s]
    double moment1_kms = 0.0;         // Intensity-weighted centroid velocity M1 [km/s]
    double moment2_sigma_kms = 0.0;   // Intensity-weighted dispersion sqrt(M2) [km/s]

    // Curves evaluated over the entire vel_kms axis
    std::vector<double> fitted_curve;     // Total composite model or smoothed profile
    std::vector<double> component1_curve; // Component 1 (for DoubleGaussian / TripleGaussian)
    std::vector<double> component2_curve; // Component 2 (for DoubleGaussian / TripleGaussian)
    std::vector<double> component3_curve; // Component 3 (for TripleGaussian)
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

    // Derived 21-cm physical & kinematic metrics
    double doppler_temp_max_k = 0.0;       // Upper-bound kinetic temperature T_k,max = 21.855 * FWHM^2 [K]
    double peak_optical_depth = 0.0;       // Peak optical depth tau_max = -ln(1 - T_peak / T_spin)
    double column_density_thick_cm2 = 0.0; // Optically-thick corrected column density N_HI,thick [atoms/cm^2]
    double equivalent_width_kms = 0.0;     // Equivalent width W_eq = int(T_B dv) / T_peak [km/s]
    double w50_kms = 0.0;                  // Full velocity width at 50% peak intensity [km/s]
    double w20_kms = 0.0;                  // Full velocity width at 20% peak intensity [km/s]

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
