#include "dc_excision.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace astro {

static double calculate_median_inplace(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    if (n % 2 != 0) {
        return v[mid];
    }
    auto left_max = std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (*left_max + v[mid]);
}

std::vector<double> DCExcision::excise(
    const std::vector<double>& power,
    const AstroConfig& config,
    DCExcisionInfo& out_info
) {
    const int nfft = static_cast<int>(power.size());
    std::vector<double> clean_power = power;

    out_info = DCExcisionInfo{};
    out_info.dc_center_freq_mhz = config.center_freq_hz / 1e6;
    if (nfft == 0) {
        return clean_power;
    }

    const int dc_bin = nfft / 2;
    out_info.dc_bin = dc_bin;

    // Respect disabled DC excision (dc_clear_hz <= 0): do not force-clear bins
    if (!std::isfinite(config.dc_clear_hz) || config.dc_clear_hz <= 0.0 ||
        !std::isfinite(config.sample_rate_hz) || config.sample_rate_hz <= 0.0) {
        out_info.cleared_bins = 0;
        out_info.cleared_bandwidth_khz = 0.0;
        return clean_power;
    }

    const double bin_res_hz = config.sample_rate_hz / static_cast<double>(nfft);
    const double raw_half = std::ceil((config.dc_clear_hz * 0.5) / bin_res_hz);
    const int half_width_bins = std::isfinite(raw_half)
        ? static_cast<int>(std::clamp(raw_half, 1.0, static_cast<double>(nfft)))
        : nfft;
    const int guard_bins = std::min(nfft, std::max(2, half_width_bins * 2));

    const int start_dc = std::max(0, dc_bin - half_width_bins);
    const int end_dc = std::min(nfft - 1, dc_bin + half_width_bins);

    const int left_start = std::max(0, start_dc - guard_bins);
    const int right_end = std::min(nfft, end_dc + 1 + guard_bins);

    std::vector<double> left_guard(power.begin() + left_start, power.begin() + start_dc);
    std::vector<double> right_guard(power.begin() + end_dc + 1, power.begin() + right_end);

    double replacement_median = 0.0;
    if (left_guard.empty() || right_guard.empty()) {
        std::vector<double> all_guards;
        all_guards.reserve(left_guard.size() + right_guard.size());
        all_guards.insert(all_guards.end(), left_guard.begin(), left_guard.end());
        all_guards.insert(all_guards.end(), right_guard.begin(), right_guard.end());
        if (all_guards.empty()) {
            std::vector<double> tmp = power;
            replacement_median = calculate_median_inplace(tmp);
        } else {
            replacement_median = calculate_median_inplace(all_guards);
        }
    }

    const double left_val = left_guard.empty() ? replacement_median : calculate_median_inplace(left_guard);
    const double right_val = right_guard.empty() ? replacement_median : calculate_median_inplace(right_guard);

    // Interpolate linearly between exact guard band centroids so linear passband slopes
    // across the excised DC notch are reproduced without step jumps or slope distortion.
    if (!left_guard.empty() && !right_guard.empty()) {
        const double x_left = 0.5 * static_cast<double>(left_start + (start_dc - 1));
        const double x_right = 0.5 * static_cast<double>((end_dc + 1) + (right_end - 1));
        const double inv_span = 1.0 / std::max(1e-9, x_right - x_left);
        for (int idx = start_dc; idx <= end_dc; ++idx) {
            const double alpha = (static_cast<double>(idx) - x_left) * inv_span;
            clean_power[static_cast<size_t>(idx)] = left_val + alpha * (right_val - left_val);
        }
    } else {
        const double fill_val = 0.5 * (left_val + right_val);
        for (int idx = start_dc; idx <= end_dc; ++idx) {
            clean_power[static_cast<size_t>(idx)] = fill_val;
        }
    }

    const int count = end_dc - start_dc + 1;
    out_info.cleared_bins = count;
    out_info.cleared_bandwidth_khz = static_cast<double>(count) * bin_res_hz * 1e-3;

    return clean_power;
}

} // namespace astro
