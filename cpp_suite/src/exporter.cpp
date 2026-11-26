#include "exporter.h"
#include "plot_renderer.h"
#include "simd_dsp.h"
#include "utf8_path.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <locale>
#include <sstream>

namespace astro {

bool Exporter::export_csv(
    const std::vector<double>& freqs_hz,
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& vel_kms,
    const std::vector<double>& raw_power,
    const std::vector<double>& baseline_power,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const std::string& output_path
) {
    if (output_path.empty()) return false;
    std::filesystem::path p = astro_fs::utf8_path(output_path);
    if (!p.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(p, std::ios::binary);
    if (!f.is_open()) return false;

    const size_t n = std::min({freqs_hz.size(), freqs_mhz.size(), vel_kms.size(),
                               raw_power.size(), baseline_power.size(),
                               calibrated_k.size(), calibrated_db.size()});
    std::vector<double> raw_db(n), base_db(n);
    SimdDsp::power_to_db_f64(raw_power.data(), raw_db.data(), n, 1e-30);
    SimdDsp::power_to_db_f64(baseline_power.data(), base_db.data(), n, 1e-30);

    std::string out_buf;
    out_buf.reserve(65536);
    out_buf = "Frequency_Hz,Frequency_MHz,Doppler_Velocity_kms,Raw_Power_dB,Baseline_Power_dB,Calibrated_Power_dB,Antenna_Temperature_K\n";

    char line_buf[512];
    // Reserve space for delim, guard ptr < end, and use scientific notation for huge magnitudes
    auto append_fixed = [](char*& ptr, char* end, double val, int prec, char delim) {
        if (ptr >= end) return;
        char* num_end = end - 1; // Reserve 1 byte for delim
        if (!std::isfinite(val)) val = 0.0;
        if (ptr < num_end) {
            std::to_chars_result r{};
            if (std::abs(val) > 1e18) {
                r = std::to_chars(ptr, num_end, val, std::chars_format::scientific, prec);
            } else {
                r = std::to_chars(ptr, num_end, val, std::chars_format::fixed, prec);
            }
            if (r.ec == std::errc()) {
                ptr = r.ptr;
            } else {
                *ptr++ = '0';
            }
        }
        if (ptr < end) {
            *ptr++ = delim;
        }
    };
    for (size_t i = 0; i < n; ++i) {
        char* ptr = line_buf;
        char* end = line_buf + sizeof(line_buf);
        append_fixed(ptr, end, freqs_hz[i], 3, ',');
        append_fixed(ptr, end, freqs_mhz[i], 6, ',');
        append_fixed(ptr, end, vel_kms[i], 3, ',');
        append_fixed(ptr, end, raw_db[i], 4, ',');
        append_fixed(ptr, end, base_db[i], 4, ',');
        append_fixed(ptr, end, calibrated_db[i], 4, ',');
        append_fixed(ptr, end, calibrated_k[i], 4, '\n');
        out_buf.append(line_buf, static_cast<size_t>(ptr - line_buf));
        if (out_buf.size() >= 60000) {
            f.write(out_buf.data(), static_cast<std::streamsize>(out_buf.size()));
            out_buf.clear();
        }
    }
    if (!out_buf.empty()) {
        f.write(out_buf.data(), static_cast<std::streamsize>(out_buf.size()));
    }

    f.flush();
    const bool ok = f.good();
    f.close();
    return ok && !f.fail();
}

static std::string escape_json_str(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (unsigned char uc : input) {
        switch (uc) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20u) {
                    char hex_buf[8];
                    std::snprintf(hex_buf, sizeof(hex_buf), "\\u%04x", static_cast<unsigned int>(uc));
                    out += hex_buf;
                } else {
                    out.push_back(static_cast<char>(uc));
                }
                break;
        }
    }
    return out;
}

static inline double safe_num(double v) {
    return std::isfinite(v) ? v : 0.0;
}

bool Exporter::export_json(
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const FileMetadata& meta,
    const std::string& output_path
) {
    if (output_path.empty()) return false;
    std::filesystem::path p = astro_fs::utf8_path(output_path);
    if (!p.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(p);
    if (!f.is_open()) return false;
    f.imbue(std::locale::classic());

    // Normalize path separators to forward slash in JSON
    std::string clean_source = meta.filepath;
    std::replace(clean_source.begin(), clean_source.end(), '\\', '/');

    const double bin_res_hz = (config.fft_size > 0) ? (config.sample_rate_hz / static_cast<double>(config.fft_size)) : 0.0;

    f << "{\n"
      << "  \"metadata\": {\n"
      << "    \"source_file\": \"" << escape_json_str(clean_source) << "\",\n"
      << "    \"filesize_bytes\": " << meta.filesize << ",\n"
      << "    \"total_samples\": " << meta.total_samples << ",\n"
      << "    \"duration_seconds\": " << std::fixed << std::setprecision(6) << safe_num(meta.duration_s) << ",\n"
      << "    \"processing_time_seconds\": " << std::fixed << std::setprecision(6) << safe_num(meta.processing_time_s) << ",\n"
      << "    \"throughput_mb_s\": " << std::fixed << std::setprecision(2) << safe_num(meta.throughput_mb_s) << "\n"
      << "  },\n"
      << "  \"sdr_settings\": {\n"
      << "    \"center_frequency_hz\": " << std::fixed << std::setprecision(1) << safe_num(config.center_freq_hz) << ",\n"
      << "    \"center_frequency_mhz\": " << std::setprecision(4) << safe_num(config.center_freq_hz / 1e6) << ",\n"
      << "    \"sample_rate_hz\": " << std::setprecision(1) << safe_num(config.sample_rate_hz) << ",\n"
      << "    \"sample_rate_msps\": " << std::setprecision(3) << safe_num(config.sample_rate_hz / 1e6) << ",\n"
      << "    \"system_temp_k\": " << std::setprecision(1) << safe_num(config.system_temp_k) << ",\n"
      << "    \"spin_temp_k\": " << std::setprecision(1) << safe_num(config.spin_temp_k) << ",\n"
      << "    \"fft_size\": " << config.fft_size << ",\n"
      << "    \"bin_resolution_hz\": " << std::setprecision(3) << safe_num(bin_res_hz) << ",\n"
      << "    \"window\": \"" << window_type_to_string(config.window) << "\",\n"
      << "    \"enable_welch_overlap\": " << (config.enable_welch_overlap ? "true" : "false") << ",\n"
      << "    \"waterfall_dt_s\": " << std::setprecision(3) << safe_num(config.waterfall_dt_s) << ",\n"
      << "    \"max_duration_s\": " << std::setprecision(3) << safe_num(config.max_duration_s) << ",\n"
      << "    \"baseline_method\": \"" << baseline_method_to_string(config.baseline_method) << "\",\n"
      << "    \"passband_edge_fraction\": " << std::setprecision(4) << safe_num(config.passband_edge_fraction) << ",\n"
      << "    \"auto_skirt_trim\": " << (config.auto_skirt_trim ? "true" : "false") << ",\n"
      << "    \"hi_protection_half_width_khz\": " << std::setprecision(2) << safe_num(config.hi_protection_half_width_khz) << ",\n"
      << "    \"filter_smoothing_sigma_khz\": " << std::setprecision(2) << safe_num(config.filter_smoothing_sigma_khz) << ",\n"
      << "    \"als_lambda\": " << std::setprecision(3) << safe_num(config.als_lambda) << ",\n"
      << "    \"als_p\": " << std::setprecision(4) << safe_num(config.als_p) << ",\n"
      << "    \"als_diff_order\": " << config.als_diff_order << ",\n"
      << "    \"als_cutoff_khz\": " << std::setprecision(2) << safe_num(config.als_cutoff_khz) << ",\n"
      << "    \"poly_degree\": " << config.poly_degree << ",\n"
      << "    \"local_baseline_order\": " << config.local_baseline_order << ",\n"
      << "    \"line_free_velocity_kms\": " << std::setprecision(2) << safe_num(config.line_free_velocity_kms) << ",\n"
      << "    \"local_baseline_outer_kms\": " << std::setprecision(2) << safe_num(config.local_baseline_outer_kms) << ",\n"
      << "    \"standing_wave_max_components\": " << config.standing_wave_max_components << ",\n"
      << "    \"standing_wave_min_period_mhz\": " << std::setprecision(3) << safe_num(config.standing_wave_min_period_mhz) << ",\n"
      << "    \"detection_snr_threshold\": " << std::setprecision(2) << safe_num(config.detection_snr_threshold) << ",\n"
      << "    \"rfi_threshold_mad\": " << std::setprecision(2) << safe_num(config.rfi_threshold_mad) << ",\n"
      << "    \"dc_spike_cleared_hz\": " << std::setprecision(1) << safe_num(config.dc_clear_hz) << "\n"
      << "  },\n"
      << "  \"hydrogen_line_detection\": {\n"
      << "    \"target_line\": \"Neutral Hydrogen (H I 21cm, 1S F=1 -> F=0)\",\n"
      << "    \"line_analysis_method\": \"" << line_analysis_method_to_string(config.line_method) << "\",\n"
      << "    \"is_detected\": " << (metrics.is_detected ? "true" : "false") << ",\n"
      << "    \"fit_success\": " << (metrics.gauss_fit.success ? "true" : "false") << ",\n"
      << "    \"rest_frequency_mhz\": " << std::setprecision(6) << HYDROGEN_REST_FREQ_MHZ << ",\n"
      << "    \"observed_peak_frequency_mhz\": " << std::setprecision(6) << safe_num(metrics.peak_freq_mhz) << ",\n"
      << "    \"doppler_recession_velocity_kms\": " << std::setprecision(3) << safe_num(metrics.peak_vel_kms) << ",\n"
      << "    \"peak_antenna_temperature_k\": " << std::setprecision(3) << safe_num(metrics.peak_temp_k) << ",\n"
      << "    \"peak_temp_smoothed_k\": " << std::setprecision(3) << safe_num(metrics.peak_temp_smoothed_k) << ",\n"
      << "    \"peak_vel_smoothed_kms\": " << std::setprecision(3) << safe_num(metrics.peak_vel_smoothed_kms) << ",\n"
      << "    \"peak_power_above_baseline_db\": " << std::setprecision(4) << safe_num(metrics.peak_power_db) << ",\n"
      << "    \"noise_floor_rms_k\": " << std::setprecision(3) << safe_num(metrics.noise_std_k) << ",\n"
      << "    \"noise_floor_rms_db\": " << std::setprecision(4) << safe_num(metrics.noise_std_db) << ",\n"
      << "    \"thermal_noise_std_k\": " << std::setprecision(4) << safe_num(metrics.thermal_noise_std_k) << ",\n"
      << "    \"off_line_rms_k\": " << std::setprecision(4) << safe_num(metrics.off_line_rms_k) << ",\n"
      << "    \"baseline_sys_error_k\": " << std::setprecision(4) << safe_num(metrics.baseline_sys_error_k) << ",\n"
      << "    \"baseline_quality_ratio\": " << std::setprecision(3) << safe_num(metrics.baseline_quality_ratio) << ",\n"
      << "    \"window_corr_factor\": " << std::setprecision(3) << safe_num(metrics.window_corr_factor) << ",\n"
      << "    \"signal_to_noise_ratio_snr\": " << std::setprecision(2) << safe_num(metrics.snr) << ",\n"
      << "    \"snr_matched\": " << std::setprecision(2) << safe_num(metrics.snr_matched) << ",\n"
      << "    \"snr_single_bin\": " << std::setprecision(2) << safe_num(metrics.snr_single_bin) << ",\n"
      << "    \"snr_integrated\": " << std::setprecision(2) << safe_num(metrics.snr_integrated) << ",\n"
      << "    \"delta_bic\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.delta_bic) << ",\n"
      << "    \"reduced_chi2\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.reduced_chi2) << ",\n"
      << "    \"line_fwhm_kms\": " << std::setprecision(2) << safe_num(metrics.fwhm_kms) << ",\n"
      << "    \"velocity_dispersion_sigma_kms\": " << std::setprecision(2) << safe_num(metrics.sigma_kms) << ",\n"
      << "    \"w50_kms\": " << std::setprecision(2) << safe_num(metrics.w50_kms) << ",\n"
      << "    \"w20_kms\": " << std::setprecision(2) << safe_num(metrics.w20_kms) << ",\n"
      << "    \"equivalent_width_kms\": " << std::setprecision(2) << safe_num(metrics.equivalent_width_kms) << ",\n"
      << "    \"doppler_temp_max_k\": " << std::setprecision(1) << safe_num(metrics.doppler_temp_max_k) << ",\n"
      << "    \"peak_optical_depth\": " << std::setprecision(5) << safe_num(metrics.peak_optical_depth) << ",\n"
      << "    \"fit_rmse_k\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.rmse_k) << ",\n"
      << "    \"secondary_component_detected\": " << (metrics.gauss_fit.has_secondary ? "true" : "false") << ",\n"
      << "    \"secondary_amplitude_k\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.amp2_k) << ",\n"
      << "    \"secondary_velocity_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.v2_center_kms) << ",\n"
      << "    \"secondary_fwhm_kms\": " << std::setprecision(2) << safe_num(metrics.gauss_fit.fwhm2_kms) << ",\n"
      << "    \"tertiary_component_detected\": " << (metrics.gauss_fit.has_tertiary ? "true" : "false") << ",\n"
      << "    \"tertiary_amplitude_k\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.amp3_k) << ",\n"
      << "    \"tertiary_velocity_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.v3_center_kms) << ",\n"
      << "    \"tertiary_fwhm_kms\": " << std::setprecision(2) << safe_num(metrics.gauss_fit.fwhm3_kms) << ",\n"
      << "    \"voigt_eta\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.voigt_eta) << ",\n"
      << "    \"voigt_lorentz_gamma_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.lorentz_gamma_kms) << ",\n"
      << "    \"gauss_hermite_h3\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.gh_h3) << ",\n"
      << "    \"gauss_hermite_h4\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.gh_h4) << ",\n"
      << "    \"busy_half_width_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.busy_half_width_kms) << ",\n"
      << "    \"busy_flank_b1\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.busy_flank_b1) << ",\n"
      << "    \"busy_flank_b2\": " << std::setprecision(4) << safe_num(metrics.gauss_fit.busy_flank_b2) << ",\n"
      << "    \"busy_trough_c\": " << std::setprecision(6) << safe_num(metrics.gauss_fit.busy_trough_c) << ",\n"
      << "    \"moment0_k_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.moment0_k_kms) << ",\n"
      << "    \"moment1_centroid_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.moment1_kms) << ",\n"
      << "    \"moment2_dispersion_kms\": " << std::setprecision(3) << safe_num(metrics.gauss_fit.moment2_sigma_kms) << ",\n"
      << "    \"integrated_flux_k_kms\": " << std::setprecision(2) << safe_num(metrics.integrated_flux_k_kms) << ",\n"
      << "    \"integrated_flux_err_k_kms\": " << std::setprecision(3) << safe_num(metrics.integrated_flux_err_k_kms) << ",\n"
      << "    \"neutral_hydrogen_column_density_cm2\": " << std::scientific << std::setprecision(3) << safe_num(metrics.column_density_cm2) << ",\n"
      << "    \"column_density_err_cm2\": " << std::scientific << std::setprecision(3) << safe_num(metrics.column_density_err_cm2) << ",\n"
      << "    \"neutral_hydrogen_column_density_thick_cm2\": " << std::scientific << std::setprecision(3) << safe_num(metrics.column_density_thick_cm2) << ",\n"
      << "    \"lo_dc_offset_khz\": " << std::fixed << std::setprecision(3) << safe_num(metrics.lo_offset_khz) << ",\n"
      << "    \"bins_from_dc_to_hi\": " << metrics.bin_separation << "\n"
      << "  }\n"
      << "}\n";

    f.flush();
    return f.good();
}

bool Exporter::export_summary_text(
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const FileMetadata& meta,
    const std::string& output_path
) {
    if (output_path.empty()) return false;
    std::filesystem::path p = astro_fs::utf8_path(output_path);
    if (!p.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(p);
    if (!f.is_open()) return false;
    f.imbue(std::locale::classic());

    const double dur_s = safe_num(meta.duration_s);
    const double dur_min = dur_s / 60.0;
    const double fc_mhz = safe_num(config.center_freq_hz / 1e6);
    const double fs_msps = safe_num(config.sample_rate_hz / 1e6);
    const double bin_res_hz = (config.fft_size > 0) ? safe_num(config.sample_rate_hz / static_cast<double>(config.fft_size)) : 0.0;
    const double bin_res_kms = safe_num(SPEED_OF_LIGHT_KMS * (bin_res_hz / HYDROGEN_REST_FREQ_HZ));
    const double fwhm_khz = safe_num(metrics.fwhm_kms / SPEED_OF_LIGHT_KMS * HYDROGEN_REST_FREQ_MHZ * 1e3);

    const char* snr_label = "No confirmed H I detection";
    if (metrics.is_detected) {
        snr_label = "CONFIRMED H I detection (passed matched-filter SNR, BIC & width gates)";
    } else if (safe_num(metrics.snr_matched) >= 3.0) {
        snr_label = "Marginal candidate (below confirmed detection threshold)";
    }

    std::string dc_assessment;
    if (std::abs(safe_num(metrics.lo_offset_khz)) < 50.0) {
        dc_assessment = "WARNING: LO DC notch lies within +/-50 kHz of H I rest frequency; line profile may be affected.";
    } else {
        std::ostringstream dss;
        dss.imbue(std::locale::classic());
        dss << "LO tuned " << std::fixed << std::setprecision(1) << std::abs(safe_num(metrics.lo_offset_khz))
            << " kHz from H I rest frequency (clear of DC excision notch).";
        dc_assessment = dss.str();
    }

    f << "HYDROGEN LINE (H I 21cm) ANALYSIS REPORT\n"
      << "======================================================================\n\n"
      << "1. Observation Metadata\n"
      << "----------------------------------------------------------------------\n"
      << "Source IQ File:             " << astro_fs::path_to_utf8(astro_fs::utf8_path(meta.filepath).filename()) << "\n"
      << "File Location:              " << meta.filepath << "\n"
      << "File Size:                  " << std::fixed << std::setprecision(2)
      << safe_num(static_cast<double>(meta.filesize) / (1024.0 * 1024.0 * 1024.0))
      << " GB (" << std::setprecision(1)
      << safe_num(static_cast<double>(meta.filesize) / (1024.0 * 1024.0)) << " MB)\n"
      << "Data Format:                CS16 (Interleaved 16-bit Signed Integer I/Q)\n"
      << "Observation Duration:       " << std::setprecision(2) << dur_s << " s (" << dur_min << " min)\n"
      << "Processing Throughput:      " << std::setprecision(1) << safe_num(meta.throughput_mb_s) << " MB/s ("
      << std::setprecision(2) << safe_num(meta.processing_time_s) << " s)\n\n"
      << "2. SDR, DSP & Baseline Configuration\n"
      << "----------------------------------------------------------------------\n"
      << "SDR Center Frequency (LO):  " << std::setprecision(6) << fc_mhz << " MHz\n"
      << "SDR Sample Rate:            " << std::setprecision(3) << fs_msps << " MSPS ("
      << static_cast<uint64_t>(std::max(0.0, safe_num(config.sample_rate_hz))) << " Hz)\n"
      << "FFT Channel Count:          " << config.fft_size << " bins (Welch 50% overlap: "
      << (config.enable_welch_overlap ? "Enabled" : "Disabled") << ")\n"
      << "Spectral Window:            " << window_type_to_string(config.window) << "\n"
      << "Channel Width (Frequency):  " << std::setprecision(2) << bin_res_hz << " Hz\n"
      << "Channel Width (Velocity):   " << std::setprecision(3) << bin_res_kms << " km/s\n"
      << "Assumed System Temp (Tsys): " << std::setprecision(1) << safe_num(config.system_temp_k) << " K\n"
      << "Assumed Spin Temp (Tspin):  " << std::setprecision(1) << safe_num(config.spin_temp_k) << " K\n"
      << "Baseline Modeling Method:   " << baseline_method_display_name(config.baseline_method)
      << " (" << baseline_method_to_string(config.baseline_method) << ")\n"
      << "Passband Skirt Trim:        " << std::setprecision(1) << safe_num(config.passband_edge_fraction * 100.0)
      << "% min (Auto-Skirt: " << (config.auto_skirt_trim ? "Enabled" : "Disabled") << ")\n"
      << "H I Protection Half-Width:  +/-" << std::setprecision(1) << safe_num(config.hi_protection_half_width_khz) << " kHz\n"
      << "Baseline Parameters:        sigma=" << std::setprecision(1) << safe_num(config.filter_smoothing_sigma_khz)
      << " kHz, als_lambda=" << std::setprecision(1) << safe_num(config.als_lambda)
      << ", als_p=" << std::setprecision(2) << safe_num(config.als_p)
      << ", poly_deg=" << config.poly_degree
      << ", local_order=" << config.local_baseline_order << "\n"
      << "Line-Free Velocity Zone:    |v| in [" << std::setprecision(1) << safe_num(config.line_free_velocity_kms)
      << ", " << safe_num(config.local_baseline_outer_kms) << "] km/s\n"
      << "Line Profile Analysis:      " << line_analysis_method_display_name(config.line_method) << "\n"
      << "RFI Outlier Threshold:      " << std::setprecision(1) << safe_num(config.rfi_threshold_mad) << " MAD\n"
      << "DC Spike Excision:          " << std::setprecision(0) << safe_num(config.dc_clear_hz) << " Hz centered at LO "
      << std::setprecision(4) << fc_mhz << " MHz\n\n"
      << "3. LO Offset & DC Excision Check\n"
      << "----------------------------------------------------------------------\n"
      << "Astronomical H I Rest Freq: " << std::setprecision(6) << HYDROGEN_REST_FREQ_MHZ << " MHz\n"
      << "LO Frequency Offset:        " << std::showpos << std::setprecision(2) << safe_num(metrics.lo_offset_khz) << " kHz\n"
      << "Bin Separation:             " << std::noshowpos << metrics.bin_separation << " FFT bins\n"
      << "Assessment:                 " << dc_assessment << "\n\n"
      << "4. Measured Line Parameters (Topocentric Frame)\n"
      << "----------------------------------------------------------------------\n"
      << "Detection Status:           " << (metrics.is_detected ? "CONFIRMED" : "NOT DETECTED")
      << " (" << snr_label << ")\n"
      << "Observed Peak Frequency:    " << std::setprecision(6) << safe_num(metrics.peak_freq_mhz) << " MHz\n"
      << "Doppler Radial Velocity:    " << std::showpos << std::setprecision(2) << safe_num(metrics.peak_vel_kms) << " km/s (Topocentric)\n"
      << "Velocity Sign:              " << (safe_num(metrics.peak_vel_kms) >= 0.0 ? "Receding (Redshifted)" : "Approaching (Blueshifted)")
      << " relative to the receiver (topocentric)\n"
      << "Peak Antenna Temperature:   " << std::noshowpos << std::setprecision(2) << safe_num(metrics.peak_temp_k) << " K\n"
      << "Peak Relative Power:        " << std::showpos << std::setprecision(3) << safe_num(metrics.peak_power_db) << " dB\n"
      << "Line-Free Noise RMS:        " << std::noshowpos << std::setprecision(3) << safe_num(metrics.noise_std_k) << " K ("
      << std::setprecision(4) << safe_num(metrics.noise_std_db) << " dB)\n"
      << "Thermal Radiometer RMS:     " << std::setprecision(3) << safe_num(metrics.thermal_noise_std_k) << " K\n"
      << "Off-Line Residual RMS:      " << std::setprecision(3) << safe_num(metrics.off_line_rms_k)
      << " K (Baseline Sys Error: " << safe_num(metrics.baseline_sys_error_k)
      << " K, Quality: " << std::setprecision(2) << safe_num(metrics.baseline_quality_ratio) << "x)\n"
      << "Matched-Filter SNR:         " << std::setprecision(2) << safe_num(metrics.snr_matched) << " sigma (threshold: "
      << std::setprecision(1) << safe_num(config.detection_snr_threshold) << " sigma)\n"
      << "Single-Bin / Integ. SNR:    " << std::setprecision(2) << safe_num(metrics.snr_single_bin)
      << " / " << safe_num(metrics.snr_integrated) << " sigma\n"
      << "Model Fit Diagnostics:      " << (metrics.gauss_fit.success ? "Converged" : "Did not converge")
      << " (Delta-BIC: " << std::setprecision(2) << safe_num(metrics.gauss_fit.delta_bic)
      << ", Reduced Chi2: " << std::setprecision(3) << safe_num(metrics.gauss_fit.reduced_chi2) << ")\n"
      << "Line FWHM:                  " << std::setprecision(2) << safe_num(metrics.fwhm_kms) << " km/s ("
      << std::setprecision(2) << fwhm_khz << " kHz)\n"
      << "Profile Widths W50 / W20:   " << std::setprecision(2) << safe_num(metrics.w50_kms) << " / "
      << safe_num(metrics.w20_kms) << " km/s\n"
      << "Equivalent Width (W_eq):    " << std::setprecision(2) << safe_num(metrics.equivalent_width_kms) << " km/s\n"
      << "Velocity Dispersion (sigma): " << std::setprecision(2) << safe_num(metrics.sigma_kms) << " km/s\n"
      << "Max Kinetic Temp (T_k,max): " << std::setprecision(0) << safe_num(metrics.doppler_temp_max_k) << " K\n"
      << "Peak Optical Depth (tau):   " << std::setprecision(4) << safe_num(metrics.peak_optical_depth)
      << " (at T_spin = " << std::setprecision(0) << safe_num(config.spin_temp_k) << " K)\n"
      << "Integrated Flux (W):        " << std::setprecision(2) << safe_num(metrics.integrated_flux_k_kms)
      << " +/- " << std::setprecision(2) << safe_num(metrics.integrated_flux_err_k_kms) << " K * km/s\n"
      << "H I Column Density (thin):  " << std::scientific << std::setprecision(3) << safe_num(metrics.column_density_cm2)
      << " +/- " << safe_num(metrics.column_density_err_cm2) << " atoms/cm^2\n"
      << "H I Column Density (thick): " << std::scientific << std::setprecision(3) << safe_num(metrics.column_density_thick_cm2)
      << " atoms/cm^2\n\n"
      << "Notes\n"
      << "----------------------------------------------------------------------\n"
      << "- Velocities are topocentric (Earth-fixed); LSR correction is not applied automatically.\n"
      << "- Temperatures scale linearly with the assumed system temperature T_sys.\n";

    f.flush();
    return f.good();
}

bool Exporter::export_all_artifacts(
    const std::string& output_dir,
    const std::string& source_file,
    const AstroConfig& config,
    const FileMetadata& meta,
    const StreamingResults& stream_res,
    const std::vector<double>& raw_power,
    const std::vector<double>& baseline_power,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const AstrophysicalMetrics& metrics
) {
    if (output_dir.empty()) return false;
    std::filesystem::path out = astro_fs::utf8_path(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    if (ec || !std::filesystem::is_directory(out, ec)) {
        return false;
    }

    std::string p1 = astro_fs::path_to_utf8(out / "1_hi_spectrum_calibrated.png");
    std::string p2 = astro_fs::path_to_utf8(out / "2_hi_raw_and_baseline.png");
    std::string p3 = astro_fs::path_to_utf8(out / "3_hi_doppler_velocity.png");
    std::string p4 = astro_fs::path_to_utf8(out / "4_hi_waterfall_time_frequency.png");
    std::string p5 = astro_fs::path_to_utf8(out / "5_hi_master_dashboard.png");

    std::vector<double> vel_kms = Astrophysics::compute_doppler_velocity(stream_res.freq_hz);

    const bool has_waterfall =
        (stream_res.waterfall_rows > 0 &&
         stream_res.waterfall_cols > 0 &&
         !stream_res.waterfall_data.empty());

    // Launch heaviest master dashboard first, followed by remaining PNG plots concurrently with text/CSV reports
    auto f5 = std::async(std::launch::async, [&]() {
        return PlotRenderer::render_master_dashboard(
            stream_res.freq_mhz, vel_kms, raw_power, baseline_power, calibrated_k, calibrated_db,
            stream_res.waterfall_data, stream_res.waterfall_rows, stream_res.waterfall_cols,
            stream_res.waterfall_times, metrics, config, source_file, p5, meta.duration_s);
    });
    auto f1 = std::async(std::launch::async, [&]() {
        return PlotRenderer::render_calibrated_spectrum(
            stream_res.freq_mhz, vel_kms, calibrated_k, calibrated_db, metrics, config, p1);
    });
    auto f2 = std::async(std::launch::async, [&]() {
        return PlotRenderer::render_raw_and_baseline(
            stream_res.freq_mhz, raw_power, baseline_power, metrics, config, p2);
    });
    auto f3 = std::async(std::launch::async, [&]() {
        return PlotRenderer::render_doppler_profile(
            vel_kms, calibrated_k, metrics, config, p3);
    });
    std::future<bool> f4;
    if (has_waterfall) {
        f4 = std::async(std::launch::async, [&]() {
            return PlotRenderer::render_waterfall_plot(
                stream_res.waterfall_data, stream_res.waterfall_rows, stream_res.waterfall_cols,
                stream_res.waterfall_times, stream_res.freq_mhz, baseline_power, config, p4);
        });
    }

    std::string csv_path = astro_fs::path_to_utf8(out / "calibrated_spectrum.csv");
    std::string txt_path = astro_fs::path_to_utf8(out / "analysis_report.txt");
    std::string json_path = astro_fs::path_to_utf8(out / "analysis_report.json");

    bool ok = true;
    ok &= export_csv(stream_res.freq_hz, stream_res.freq_mhz, vel_kms, raw_power, baseline_power, calibrated_k, calibrated_db, csv_path);
    ok &= export_summary_text(metrics, config, meta, txt_path);
    ok &= export_json(metrics, config, meta, json_path);

    ok &= f1.get();
    ok &= f2.get();
    ok &= f3.get();
    if (has_waterfall) {
        ok &= f4.get();
    }
    ok &= f5.get();

    return ok;
}

} // namespace astro
