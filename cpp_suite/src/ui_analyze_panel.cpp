#include "ui_analyze_panel.h"
#include "baseline_fitter.h"
#include "exporter.h"
#include "rfi_filter.h"
#include "simd_dsp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace astro {

namespace {

static std::string get_exe_parent_output_dir() {
    fs::path dir = ui_obs::win_fs::get_exe_dir() / "output";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir.string();
}

static void build_normalized_waterfall(const StreamingResults& stream,
                                       int& out_w, int& out_h,
                                       std::vector<uint8_t>& out_u8,
                                       double& out_vmin_db, double& out_vmax_db) {
    out_w = 0;
    out_h = 0;
    out_u8.clear();
    out_vmin_db = -10.0;
    out_vmax_db = 10.0;

    const int src_r = stream.waterfall_rows;
    const int src_c = stream.waterfall_cols;
    if (src_r <= 0 || src_c <= 0 || stream.waterfall_data.size() < static_cast<size_t>(src_r) * src_c) {
        return;
    }

    // Estimate median passband reference power for relative dB scale
    std::vector<double> valid_p;
    valid_p.reserve(stream.averaged_power.size());
    for (double p : stream.averaged_power) {
        if (std::isfinite(p) && p > 1e-18) valid_p.push_back(p);
    }
    double ref_power = 1.0;
    if (!valid_p.empty()) {
        size_t mid = valid_p.size() / 2;
        std::nth_element(valid_p.begin(), valid_p.begin() + mid, valid_p.end());
        ref_power = std::max(valid_p[mid], 1e-18);
    }
    const double inv_ref = 1.0 / ref_power;

    // Downsample cleanly within DX11 safe texture caps (<= 2048 x 1024)
    out_w = std::min(src_c, 2048);
    out_h = std::min(src_r, 1024);

    const size_t n = static_cast<size_t>(out_w) * static_cast<size_t>(out_h);
    std::vector<float> rel_db(n, 0.0f);
    for (int r = 0; r < out_h; ++r) {
        int r0 = static_cast<int>((static_cast<double>(r) / out_h) * src_r);
        int r1 = std::max(r0 + 1, static_cast<int>((static_cast<double>(r + 1) / out_h) * src_r));
        r1 = std::min(r1, src_r);

        float* dst_row = rel_db.data() + static_cast<size_t>(r) * out_w;
        for (int c = 0; c < out_w; ++c) {
            int c0 = static_cast<int>((static_cast<double>(c) / out_w) * src_c);
            int c1 = std::max(c0 + 1, static_cast<int>((static_cast<double>(c + 1) / out_w) * src_c));
            c1 = std::min(c1, src_c);

            // Max-pool over sub-bins so narrow H I/RFI peaks remain crisp when downsampled
            float peak_val = 0.0f;
            for (int sr = r0; sr < r1; ++sr) {
                const float* row_ptr = stream.waterfall_data.data() + static_cast<size_t>(sr) * src_c;
                for (int sc = c0; sc < c1; ++sc) {
                    const float v = row_ptr[sc];
                    if (v > peak_val) peak_val = v;
                }
            }
            const double scaled = std::max(static_cast<double>(peak_val) * inv_ref, 1e-12);
            dst_row[c] = static_cast<float>(10.0 * std::log10(scaled));
        }
    }

    // Compute 5th .. 99.5th percentile contrast window in O(S) over <= 65536 strided samples
    const size_t stride = std::max<size_t>(1, n / 65536);
    std::vector<float> sample_vals;
    sample_vals.reserve((n + stride - 1) / stride);
    for (size_t i = 0; i < n; i += stride) {
        sample_vals.push_back(rel_db[i]);
    }
    const size_t ns = sample_vals.size();
    const size_t idx_low = static_cast<size_t>(0.05 * static_cast<double>(ns - 1));
    const size_t idx_high = static_cast<size_t>(0.995 * static_cast<double>(ns - 1));
    std::nth_element(sample_vals.begin(), sample_vals.begin() + idx_low, sample_vals.end());
    out_vmin_db = static_cast<double>(sample_vals[idx_low]);
    std::nth_element(sample_vals.begin() + idx_low, sample_vals.begin() + idx_high, sample_vals.end());
    out_vmax_db = std::max(out_vmin_db + 1.0, static_cast<double>(sample_vals[idx_high]));

    const float vmin_f = static_cast<float>(out_vmin_db);
    const float inv_span = 255.0f / static_cast<float>(out_vmax_db - out_vmin_db);
    out_u8.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float scaled = (rel_db[i] - vmin_f) * inv_span;
        out_u8[i] = static_cast<uint8_t>(std::clamp(scaled, 0.0f, 255.0f));
    }
}

static void format_byte_size(uint64_t bytes, char* out, size_t out_sz) {
    if (bytes >= (1024ULL * 1024ULL * 1024ULL)) {
        std::snprintf(out, out_sz, "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (1024ULL * 1024ULL)) {
        std::snprintf(out, out_sz, "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else if (bytes >= 1024ULL) {
        std::snprintf(out, out_sz, "%.0f KB", static_cast<double>(bytes) / 1024.0);
    } else {
        std::snprintf(out, out_sz, "%llu B", static_cast<unsigned long long>(bytes));
    }
}

struct QuadSwatch {
    const char* label;
    ImVec4      color;
};

static void draw_quad_header(ImDrawList* dl, ImFont* semibold, ImFont* regular,
                             const char* title, const QuadSwatch* items, int count) {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float avail_w = ImGui::GetContentRegionAvail().x;
    float row_h = ui_obs::dp(18.0f);

    {
        ui_obs::ScopedFont tf(semibold, 12.0f);
        row_h = std::max(row_h, ImGui::GetFontSize() + ui_obs::dp(2.0f));
        dl->AddText(pos, ui_obs::palette::U32(ui_obs::palette::TextPrimary), title);
    }

    if (items && count > 0) {
        ui_obs::ScopedFont lf(regular, 11.0f);
        const float lfs = ImGui::GetFontSize();
        const float text_y = pos.y + (row_h - lfs) * 0.5f;
        const float sw_w = ui_obs::dp(10.0f);
        const float sw_h = ui_obs::dp(2.5f);
        float rx = pos.x + avail_w;

        for (int i = count - 1; i >= 0; --i) {
            const ImVec2 tsz = ImGui::CalcTextSize(items[i].label);
            rx -= tsz.x;
            dl->AddText(ImVec2(rx, text_y), ui_obs::palette::U32(ui_obs::palette::TextMuted), items[i].label);
            rx -= ui_obs::dp(4.0f) + sw_w;
            const float sy = pos.y + row_h * 0.5f - sw_h * 0.5f;
            dl->AddRectFilled(ImVec2(rx, sy), ImVec2(rx + sw_w, sy + sw_h),
                              ui_obs::palette::U32(items[i].color), ui_obs::dp(1.2f));
            rx -= ui_obs::dp(10.0f);
        }
    }

    ImGui::Dummy(ImVec2(avail_w, row_h));
}

static size_t find_nearest_index(const std::vector<double>& axis, double x) {
    if (axis.size() <= 1) return 0;
    const bool ascending = axis.back() >= axis.front();
    if (ascending) {
        auto it = std::lower_bound(axis.begin(), axis.end(), x);
        if (it == axis.begin()) return 0;
        if (it == axis.end()) return axis.size() - 1;
        size_t hi = static_cast<size_t>(it - axis.begin());
        return (std::abs(axis[hi] - x) < std::abs(axis[hi - 1] - x)) ? hi : (hi - 1);
    }
    auto it = std::lower_bound(axis.begin(), axis.end(), x, std::greater<double>());
    if (it == axis.begin()) return 0;
    if (it == axis.end()) return axis.size() - 1;
    size_t hi = static_cast<size_t>(it - axis.begin());
    return (std::abs(axis[hi] - x) < std::abs(axis[hi - 1] - x)) ? hi : (hi - 1);
}

static void draw_gradient_progress_bar(ImDrawList* dl, ImVec2 pmin, ImVec2 pmax,
                                       float frac, const char* label, ImFont* mono_font) {
    const float w = pmax.x - pmin.x;
    const float h = pmax.y - pmin.y;
    const float rounding = ui_obs::dp(5.0f);
    dl->AddRectFilled(pmin, pmax, ui_obs::palette::U32(ui_obs::palette::BgWell), rounding);
    dl->AddRect(pmin, pmax, ui_obs::palette::U32(ui_obs::palette::Border), rounding, 1.0f, ImDrawFlags_None);

    const float clamped = std::clamp(frac, 0.0f, 1.0f);
    const float fill_w = w * clamped;
    if (fill_w > 2.0f) {
        const ImVec2 fmax(pmin.x + fill_w, pmax.y);
        dl->PushClipRect(pmin, fmax, true);
        dl->AddRectFilled(pmin, fmax, ui_obs::palette::U32(ui_obs::palette::AccentBlue, 0.85f), rounding);
        dl->PopClipRect();
    }

    if (label && label[0] != '\0') {
        ui_obs::ScopedFont mf(mono_font, 11.5f);
        const ImVec2 tsz = ImGui::CalcTextSize(label);
        const ImVec2 tpos(pmin.x + (w - tsz.x) * 0.5f, pmin.y + (h - tsz.y) * 0.5f);
        dl->AddText(ImVec2(tpos.x + 1.0f, tpos.y + 1.0f), IM_COL32(0, 0, 0, 150), label);
        dl->AddText(tpos, ui_obs::palette::U32(ui_obs::palette::TextPrimary), label);
    }
}

} // namespace

UiAnalyzePanel::UiAnalyzePanel(AppState& app_state)
    : app_state_(app_state)
{
    trace_scratch_.reserve(16384);
    fill_scratch_.reserve(8192);
    active_lut256_ = ui_obs::build_colormap_lut256(ui_colormap_idx_);
    scan_files();
    if (!discovered_files_.empty()) {
        on_file_selected(discovered_files_[0]);
        selected_file_idx_ = 0;
    }
}

UiAnalyzePanel::~UiAnalyzePanel() {
    cancel_requested_ = true;
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }
    release_d3d_resources();
}

void UiAnalyzePanel::init_d3d_resources(ID3D11Device* device) {
    d3d_device_ = device;
}

void UiAnalyzePanel::release_d3d_resources() {
    if (waterfall_srv_) {
        waterfall_srv_->Release();
        waterfall_srv_ = nullptr;
    }
    if (waterfall_texture_) {
        waterfall_texture_->Release();
        waterfall_texture_ = nullptr;
    }
    tex_width_ = 0;
    tex_height_ = 0;
    last_uploaded_snap_ = nullptr;
}

std::shared_ptr<const AnalysisSnapshot> UiAnalyzePanel::get_snapshot() {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return active_snapshot_;
}

std::string UiAnalyzePanel::get_status_message() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_message_;
}

void UiAnalyzePanel::set_status_message(const std::string& msg) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_message_ = msg;
}

std::string UiAnalyzePanel::get_export_message() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return export_status_msg_;
}

void UiAnalyzePanel::set_export_message(const std::string& msg) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    export_status_msg_ = msg;
}

void UiAnalyzePanel::scan_files() {
    std::vector<std::string> search_dirs = {
        ui_obs::win_fs::get_default_recordings_dir(),
        "recordings",
        "bin/recordings",
        "../recordings",
        "."
    };

    if (custom_file_path_[0] != '\0') {
        std::error_code ec;
        fs::path p(custom_file_path_);
        if (p.has_parent_path() && fs::exists(p.parent_path(), ec)) {
            search_dirs.push_back(p.parent_path().string());
        }
    }

    discovered_files_ = SidecarReader::scan_cs16_files(search_dirs);

    // Resync selected_file_idx_ cleanly without unsafe equivalent() exceptions
    selected_file_idx_ = -1;
    if (custom_file_path_[0] != '\0') {
        std::error_code ec;
        fs::path cur_can = fs::weakly_canonical(fs::path(custom_file_path_), ec);
        for (size_t i = 0; i < discovered_files_.size(); ++i) {
            std::error_code ec2;
            fs::path cand = fs::weakly_canonical(fs::path(discovered_files_[i]), ec2);
            if ((!ec && !ec2 && cur_can == cand) || discovered_files_[i] == custom_file_path_) {
                selected_file_idx_ = static_cast<int>(i);
                break;
            }
        }
    }
}

void UiAnalyzePanel::select_and_run_file(const std::string& path, bool auto_run) {
    scan_files();
    on_file_selected(path);

    selected_file_idx_ = -1;
    std::error_code ec;
    fs::path target_can = fs::weakly_canonical(fs::path(path), ec);
    for (size_t i = 0; i < discovered_files_.size(); ++i) {
        std::error_code ec2;
        fs::path cand = fs::weakly_canonical(fs::path(discovered_files_[i]), ec2);
        if ((!ec && !ec2 && target_can == cand) || discovered_files_[i] == path) {
            selected_file_idx_ = static_cast<int>(i);
            break;
        }
    }

    if (auto_run) {
        start_analysis();
    }
}

void UiAnalyzePanel::on_file_selected(const std::string& path) {
    std::snprintf(custom_file_path_, sizeof(custom_file_path_), "%s", path.c_str());

    selected_filename_[0] = '\0';
    selected_file_size_bytes_ = 0;
    if (!path.empty()) {
        std::error_code ec;
        fs::path p(path);
        const std::string fn = p.filename().string();
        std::snprintf(selected_filename_, sizeof(selected_filename_), "%s", fn.c_str());
        const auto sz = fs::file_size(p, ec);
        if (!ec) {
            selected_file_size_bytes_ = static_cast<uint64_t>(sz);
        }
    }

    current_sidecar_ = SidecarReader::parse(path);
    ui_center_freq_mhz_ = 1420.5;
    ui_sample_rate_msps_ = 2.4;
    if (current_sidecar_.found_sidecar || current_sidecar_.center_freq_hz > 0.0) {
        if (current_sidecar_.center_freq_hz > 0.0) {
            ui_center_freq_mhz_ = current_sidecar_.center_freq_hz / 1e6;
        }
        if (current_sidecar_.sample_rate_hz > 0.0) {
            ui_sample_rate_msps_ = current_sidecar_.sample_rate_hz / 1e6;
        }
        if (current_sidecar_.found_sidecar) {
            set_status_message("Loaded sidecar metadata (" + current_sidecar_.sidecar_source + "). Ready.");
        } else {
            set_status_message("Parsed metadata from filename. Ready.");
        }
    } else {
        set_status_message("Selected file. Verify LO / Sample Rate and run H I pipeline.");
    }
}

AnalyzeParamsSig UiAnalyzePanel::current_signature() const {
    AnalyzeParamsSig sig;
    sig.filepath = custom_file_path_;
    sig.center_freq_mhz = ui_center_freq_mhz_;
    sig.sample_rate_msps = ui_sample_rate_msps_;
    sig.fft_size_idx = ui_fft_size_idx_;
    sig.baseline_idx = ui_baseline_idx_;
    sig.line_method_idx = ui_line_method_idx_;
    sig.tsys_k = ui_tsys_k_;
    sig.duration_s = ui_duration_s_;
    sig.dc_clear_khz = ui_dc_clear_khz_;
    sig.als_lambda = static_cast<double>(ui_als_lambda_);
    sig.als_p = static_cast<double>(ui_als_p_);
    sig.poly_degree = ui_poly_deg_;
    return sig;
}

bool UiAnalyzePanel::validate_inputs(std::string& out_error) const {
    val_cache_path_.clear(); // Force fresh check before execution
    return validate_inputs_cached(out_error);
}

bool UiAnalyzePanel::validate_inputs_cached(std::string& out_error) const {
    const std::string target = custom_file_path_;
    if (target.empty()) {
        out_error = "Select a .cs16 file first.";
        return false;
    }
    if (app_state_.is_file_actively_recording(target)) {
        out_error = "File is currently being written by Live Recorder.";
        return false;
    }
    const uint64_t now_ms = GetTickCount64();
    if (target != val_cache_path_ || (now_ms - val_cache_tick_ms_) >= 500ULL) {
        std::error_code ec;
        val_cache_file_ok_ = fs::exists(target, ec) && !fs::is_directory(target, ec);
        val_cache_path_ = target;
        val_cache_tick_ms_ = now_ms;
    }
    if (!val_cache_file_ok_) {
        out_error = "Input .cs16 file does not exist on disk.";
        return false;
    }
    if (ui_center_freq_mhz_ < 1.0 || ui_center_freq_mhz_ > 6000.0) {
        out_error = "Center frequency must be between 1 and 6000 MHz.";
        return false;
    }
    if (ui_sample_rate_msps_ < 0.1 || ui_sample_rate_msps_ > 20.0) {
        out_error = "Sample rate must be between 0.1 and 20.0 MSPS.";
        return false;
    }
    if (ui_tsys_k_ < 1.0f) {
        out_error = "System Temperature T_sys must be >= 1.0 K.";
        return false;
    }
    return true;
}

std::string UiAnalyzePanel::get_output_dir_for_source(const std::string& src_path) const {
    std::string stem = fs::path(src_path).stem().string();
    if (stem.empty()) stem = "analysis";
    fs::path out_dir = fs::path(get_exe_parent_output_dir()) / stem;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    return out_dir.string();
}

void UiAnalyzePanel::start_analysis() {
    if (is_analyzing_.load()) return;

    std::string err;
    if (!validate_inputs(err)) {
        set_status_message("Blocked: " + err);
        return;
    }

    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }

    static constexpr int fft_choices[] = {2048, 4096, 8192, 16384, 32768};
    AstroConfig run_cfg = config_;
    run_cfg.center_freq_hz = ui_center_freq_mhz_ * 1e6;
    run_cfg.sample_rate_hz = ui_sample_rate_msps_ * 1e6;
    run_cfg.fft_size = fft_choices[std::clamp(ui_fft_size_idx_, 0, 4)];
    run_cfg.system_temp_k = static_cast<double>(ui_tsys_k_);
    run_cfg.max_duration_s = static_cast<double>(ui_duration_s_);
    run_cfg.dc_clear_hz = static_cast<double>(ui_dc_clear_khz_) * 1000.0;
    run_cfg.als_lambda = static_cast<double>(ui_als_lambda_);
    run_cfg.als_p = static_cast<double>(ui_als_p_);
    run_cfg.poly_degree = std::clamp(ui_poly_deg_, 2, 20);
    run_cfg.verbose = false;
    run_cfg.baseline_method = (ui_baseline_idx_ == 1) ? BaselineMethod::ALS
                            : (ui_baseline_idx_ == 2) ? BaselineMethod::Poly
                                                      : BaselineMethod::RippleFilter;
    static constexpr LineAnalysisMethod kLineMethods[] = {
        LineAnalysisMethod::GaussianLS,
        LineAnalysisMethod::DoubleGaussian,
        LineAnalysisMethod::VoigtProfile,
        LineAnalysisMethod::GaussHermite,
        LineAnalysisMethod::SpectralMoments
    };
    run_cfg.line_method = kLineMethods[std::clamp(ui_line_method_idx_, 0, 4)];

    const std::string target_file = custom_file_path_;
    const AnalyzeParamsSig run_sig = current_signature();
    run_cfg.output_dir = get_output_dir_for_source(target_file);

    cancel_requested_ = false;
    is_analyzing_ = true;
    progress_pct_ = 0.0;
    progress_speed_ = 0.0;
    progress_eta_ = 0.0;
    set_status_message("Streaming FFT Accumulation (AVX2 + FMA)...");
    set_export_message("");

    worker_thread_ = std::make_unique<std::thread>([this, target_file, run_cfg, run_sig]() {
        try {
            StreamingProcessor processor(target_file, run_cfg);
            processor.set_cancel_token(&cancel_requested_);

            auto snap = std::make_shared<AnalysisSnapshot>();
            snap->source_filepath = target_file;
            snap->config_used = run_cfg;
            snap->stream = processor.process([this](double pct, double mbps, double eta) {
                progress_pct_.store(pct);
                progress_speed_.store(mbps);
                progress_eta_.store(eta);
            });

            if (cancel_requested_.load() || snap->stream.averaged_power.empty()) {
                set_status_message(cancel_requested_.load() ? "Analysis cancelled." : "Analysis failed: empty stream.");
                is_analyzing_ = false;
                return;
            }

            set_status_message("DC Excision, RFI Mitigation & Baseline Fit...");
            DCExcisionInfo dc_info;
            auto dc_clean = DCExcision::excise(snap->stream.averaged_power, run_cfg, dc_info);

            int num_rfi = 0;
            snap->rfi_clean_power = RFIFilter::mitigate(dc_clean, snap->stream.freq_mhz, run_cfg, num_rfi);

            // Precompute log10 dB passband once on worker thread for zero-cost GUI plotting
            snap->raw_db.resize(snap->rfi_clean_power.size());
            SimdDsp::power_to_db_f64(snap->rfi_clean_power.data(), snap->raw_db.data(), snap->rfi_clean_power.size());

            auto base_res = BaselineFitter::fit(snap->rfi_clean_power, snap->stream.freq_mhz, run_cfg);
            snap->baseline_power = std::move(base_res.baseline_power);
            snap->baseline_db = std::move(base_res.baseline_db);

            const size_t n = snap->rfi_clean_power.size();
            snap->residual_db.resize(n);
            snap->calibrated_k.resize(n);
            snap->calibrated_db.resize(n);
            for (size_t i = 0; i < n; ++i) {
                snap->residual_db[i] = snap->raw_db[i] - snap->baseline_db[i];
                double bp = std::max(snap->baseline_power[i], 1e-18);
                double ratio = std::max(snap->rfi_clean_power[i] / bp, 1e-12);
                snap->calibrated_k[i] = run_cfg.system_temp_k * (ratio - 1.0);
                snap->calibrated_db[i] = 10.0 * std::log10(ratio);
            }

            snap->vel_kms = Astrophysics::compute_doppler_velocity(snap->stream.freq_hz);

            double local_slope = 0.0, local_intercept = 0.0;
            BaselineFitter::refine_local(snap->vel_kms, snap->calibrated_k, snap->calibrated_db,
                                         run_cfg, local_slope, local_intercept);

            snap->metrics = Astrophysics::calculate_metrics(
                snap->stream.freq_mhz, snap->vel_kms, snap->calibrated_k, snap->calibrated_db, run_cfg);

            set_status_message("Building Normalized Waterfall Texture...");
            build_normalized_waterfall(snap->stream, snap->wf_tex_w, snap->wf_tex_h,
                                       snap->wf_norm_u8, snap->wf_vmin_db, snap->wf_vmax_db);
            snap->wf_duration_s = snap->stream.total_duration_s;

            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                active_snapshot_ = snap;
                last_run_sig_ = run_sig;
            }

            char done_msg[256];
            std::snprintf(done_msg, sizeof(done_msg),
                          "Done in %.2fs (%.0f MB/s) | Peak: %.4f MHz (%.2f K, %.2f sigma)",
                          snap->stream.elapsed_processing_time_s,
                          snap->stream.throughput_mb_s,
                          snap->metrics.peak_freq_mhz,
                          snap->metrics.peak_temp_k,
                          snap->metrics.snr);
            set_status_message(done_msg);
        } catch (const std::exception& ex) {
            set_status_message(std::string("Pipeline error: ") + ex.what());
        }
        is_analyzing_ = false;
    });
}

void UiAnalyzePanel::cancel_analysis() {
    if (is_analyzing_.load()) {
        cancel_requested_ = true;
        set_status_message("Cancelling streaming worker...");
    }
}

void UiAnalyzePanel::start_export(const std::shared_ptr<const AnalysisSnapshot>& snap) {
    if (!snap || is_exporting_.load() || is_analyzing_.load()) return;

    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }

    is_exporting_ = true;
    set_export_message("Exporting 8 scientific artifacts...");

    export_thread_ = std::make_unique<std::thread>([this, snap]() {
        FileMetadata meta;
        meta.filepath = snap->source_filepath;
        meta.filesize = snap->stream.file_size_bytes;
        meta.total_samples = snap->stream.total_samples;
        meta.duration_s = snap->stream.total_duration_s;
        meta.processing_time_s = snap->stream.elapsed_processing_time_s;
        meta.throughput_mb_s = snap->stream.throughput_mb_s;

        std::string out_dir = get_output_dir_for_source(snap->source_filepath);
        bool ok = Exporter::export_all_artifacts(
            out_dir, snap->source_filepath, snap->config_used, meta,
            snap->stream, snap->rfi_clean_power, snap->baseline_power,
            snap->calibrated_k, snap->calibrated_db, snap->metrics);

        set_export_message(ok ? ("Exported 8 artifacts to " + out_dir)
                              : "Export failed (check write permissions).");
        is_exporting_ = false;
    });
}

void UiAnalyzePanel::render_panel(ID3D11DeviceContext* ctx, float width, float height) {
    auto snap = get_snapshot();
    if (snap.get() != last_zoom_snap_) {
        zoom_cal_spec_.active = false;
        zoom_raw_base_.active = false;
        zoom_waterfall_.active = false;
        zoom_q2_db_.active = false;
        last_zoom_snap_ = snap.get();
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(8.0f)));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(8.0f), ui_obs::dp(8.0f)));
    ImGui::BeginChild("##AnalyzeRoot", ImVec2(width, height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const float inner_w = width - ui_obs::dp(20.0f);
    const float top_h = ui_obs::dp(48.0f);
    render_top_bar(inner_w, top_h, snap);

    const float rem_h = std::max(ui_obs::dp(120.0f), height - top_h - ui_obs::dp(24.0f));
    const float sidebar_w = std::min(ui_obs::dp(320.0f), std::max(ui_obs::dp(265.0f), inner_w * 0.24f));
    const float main_w = std::max(ui_obs::dp(260.0f), inner_w - sidebar_w - ui_obs::dp(8.0f));

    render_sidebar_controls(sidebar_w, rem_h, snap);
    ImGui::SameLine();

    ImGui::BeginChild("##AnalyzeMainCol", ImVec2(main_w, rem_h), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    // Non-occluding Pinned KPI Strip + Stale-Parameter Banner
    if (snap) {
        bool params_dirty = false;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            params_dirty = (current_signature() != last_run_sig_);
        }
        if (params_dirty && !is_analyzing_.load()) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Tint(ui_obs::palette::Amber, 0.08f));
            ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(ui_obs::palette::Amber, 0.40f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(4.0f)));
            ImGui::BeginChild("##StaleBanner", ImVec2(main_w, ui_obs::dp(28.0f)), ImGuiChildFlags_Borders,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

            ImDrawList* bdl = ImGui::GetWindowDrawList();
            const ImVec2 bmin = ImGui::GetWindowPos();
            const ImVec2 bmax(bmin.x + ImGui::GetWindowWidth(), bmin.y + ImGui::GetWindowHeight());
            bdl->AddCircleFilled(ImVec2(bmin.x + ui_obs::dp(13.0f), (bmin.y + bmax.y) * 0.5f),
                                 ui_obs::dp(3.0f), ui_obs::palette::U32(ui_obs::palette::Amber), 12);

            ImGui::SetCursorPos(ImVec2(ui_obs::dp(24.0f), ui_obs::dp(4.5f)));
            {
                ui_obs::ScopedFont bf(fonts_.regular, 12.0f);
                ImGui::TextColored(ui_obs::palette::Amber,
                                   "Parameters changed since last run - recompute to update calibrated spectra.");
            }

            const float rerun_w = ui_obs::dp(76.0f);
            ImGui::SameLine(std::max(ui_obs::dp(260.0f), main_w - rerun_w - ui_obs::dp(10.0f)));
            ImGui::SetCursorPosY(ui_obs::dp(3.0f));
            ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::Amber, 0.22f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::Amber, 0.36f));
            ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(ui_obs::palette::Amber, 0.65f));
            if (ImGui::Button("Re-run", ImVec2(rerun_w, ui_obs::dp(22.0f)))) {
                start_analysis();
            }
            ImGui::PopStyleColor(3);

            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
        }

        render_kpi_strip(main_w, snap->metrics, snap->config_used);
    }

    // Tabs Viewport
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(8.0f)));
    ImGui::BeginChild("##AnalyzePlotArea", ImVec2(main_w, std::max(ui_obs::dp(100.0f), ImGui::GetContentRegionAvail().y)),
                      ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (!snap || is_analyzing_.load()) {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        render_empty_state(avail.x, avail.y);
    } else if (ImGui::BeginTabBar("##AnalyzeTabs")) {
        if (ImGui::BeginTabItem("1. Calibrated H I Spectrum")) {
            render_tab_calibrated_spectrum(snap);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("2. Raw Passband & Baseline")) {
            render_tab_raw_and_baseline(snap);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("3. Time-Frequency Waterfall")) {
            render_tab_waterfall(ctx, snap);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("4. Master Dashboard (2x2)")) {
            render_tab_master_dashboard(ctx, snap);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);

    ImGui::EndChild(); // ##AnalyzeMainCol

    ImGui::EndChild(); // ##AnalyzeRoot
    ImGui::PopStyleVar(2);
}

void UiAnalyzePanel::render_empty_state(float width, float height) {
    if (width < ui_obs::dp(180.0f) || height < ui_obs::dp(160.0f)) return;

    const bool running = is_analyzing_.load();
    const float card_w = std::min(width - ui_obs::dp(32.0f), ui_obs::dp(560.0f));
    const float card_x = std::max(0.0f, (width - card_w) * 0.5f);
    const float card_y = std::max(ui_obs::dp(12.0f), (height - ui_obs::dp(290.0f)) * 0.36f);

    ImGui::SetCursorPos(ImVec2(card_x, card_y));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::SurfaceAlt);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Border);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(24.0f), ui_obs::dp(20.0f)));
    ImGui::BeginChild("##EmptyOnboarding", ImVec2(card_w, 0.0f),
                      ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wmin = ImGui::GetWindowPos();
    const float ww = ImGui::GetWindowWidth();

    // Centered instrument icon
    const ImVec2 emblem_c(wmin.x + ww * 0.5f, wmin.y + ui_obs::dp(34.0f));
    const ImVec4 emblem_col = running ? ui_obs::palette::AccentBlue : ui_obs::palette::Cyan;
    ui_obs::icons::draw_spectrum_peak(dl, emblem_c, ui_obs::dp(14.0f), ui_obs::palette::U32(emblem_col));

    ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(42.0f)));

    auto center_text = [&](const char* txt, const ImVec4& col, ImFont* font, float size) {
        ui_obs::ScopedFont sf(font, size);
        const ImVec2 tsz = ImGui::CalcTextSize(txt);
        const float avail = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (avail - tsz.x) * 0.5f));
        ImGui::TextColored(col, "%s", txt);
    };

    if (running) {
        center_text("Running 21cm H I Analysis Pipeline", ui_obs::palette::TextPrimary, fonts_.semibold, 17.0f);
        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(4.0f)));
        const std::string st = get_status_message();
        center_text(st.c_str(), ui_obs::palette::TextMuted, fonts_.regular, 13.0f);

        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(12.0f)));
        const ImVec2 pmin = ImGui::GetCursorScreenPos();
        const float bar_w = ImGui::GetContentRegionAvail().x;
        const float bar_h = ui_obs::dp(24.0f);
        const ImVec2 pmax(pmin.x + bar_w, pmin.y + bar_h);
        const float pct = static_cast<float>(progress_pct_.load() / 100.0);
        char prog_lbl[96];
        std::snprintf(prog_lbl, sizeof(prog_lbl), "%.1f%%  |  %.0f MB/s  |  ETA %.1fs",
                      progress_pct_.load(), progress_speed_.load(), progress_eta_.load());
        draw_gradient_progress_bar(dl, pmin, pmax, pct, prog_lbl, fonts_.mono);
        ImGui::Dummy(ImVec2(bar_w, bar_h));

        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(10.0f)));
        const double p = progress_pct_.load();
        const int active_stage = (p < 70.0) ? 0 : (p < 90.0) ? 1 : 2;
        static const char* stages[3] = {"[1] AVX2 FFT", "[2] RFI & Baseline", "[3] Doppler Fit"};
        if (ImGui::BeginTable("##PipeStages", 3, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            for (int i = 0; i < 3; ++i) {
                ImGui::TableSetColumnIndex(i);
                const bool done = (i < active_stage);
                const bool cur = (i == active_stage);
                const ImVec4 c = done ? ui_obs::palette::Emerald : (cur ? ui_obs::palette::Cyan : ui_obs::palette::TextDim);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, done || cur ? ui_obs::palette::Tint(c, 0.12f) : ui_obs::palette::BgDeep);
                ImGui::PushStyleColor(ImGuiCol_Border, done || cur ? ui_obs::palette::Tint(c, 0.48f) : ui_obs::palette::BorderSubtle);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(5.0f)));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(4.0f), ui_obs::dp(2.0f)));
                ImGui::BeginChild(stages[i], ImVec2(-1.0f, ui_obs::dp(28.0f)), ImGuiChildFlags_Borders,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                center_text(stages[i], c, fonts_.semibold, 12.0f);
                ImGui::EndChild();
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }
            ImGui::EndTable();
        }
    } else {
        center_text("21cm Neutral Hydrogen Analysis Workstation", ui_obs::palette::TextPrimary, fonts_.semibold, 17.0f);
        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(4.0f)));
        {
            ui_obs::ScopedFont df(fonts_.regular, 13.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ui_obs::palette::TextMuted);
            ImGui::TextWrapped(
                "Calibrate raw .cs16 I/Q captures for Galactic 1420.406 MHz emission, "
                "Doppler V_LSR velocity, line profile modeling, and N_HI column density.");
            ImGui::PopStyleColor();
        }

        const std::string cur_status = get_status_message();
        if (cur_status.rfind("Blocked:", 0) == 0 ||
            cur_status.rfind("Pipeline error:", 0) == 0 ||
            cur_status.rfind("Analysis failed", 0) == 0) {
            ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(6.0f)));
            center_text(cur_status.c_str(), ui_obs::palette::Amber, fonts_.semibold, 12.0f);
        }

        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(12.0f)));
        if (ImGui::BeginTable("##WorkflowSteps", 3, ImGuiTableFlags_SizingStretchSame)) {
            struct StepSpec { const char* id; const char* num; const char* title; ImVec4 col; };
            static const StepSpec steps[3] = {
                {"##St1", "1. SELECT .CS16", "Choose IQ file",   ui_obs::palette::Cyan},
                {"##St2", "2. TUNE MODEL",   "Baseline & Fit",   ui_obs::palette::AccentBlue},
                {"##St3", "3. INSPECT",      "Plots & 8x Export",ui_obs::palette::Emerald}
            };
            ImGui::TableNextRow();
            for (int i = 0; i < 3; ++i) {
                ImGui::TableSetColumnIndex(i);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::BgDeep);
                ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(8.0f), ui_obs::dp(5.0f)));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(4.0f), ui_obs::dp(2.0f)));
                ImGui::BeginChild(steps[i].id, ImVec2(-1.0f, ui_obs::dp(48.0f)), ImGuiChildFlags_Borders,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                {
                    ui_obs::ScopedFont sf(fonts_.semibold, 11.0f);
                    ImGui::TextColored(steps[i].col, "%s", steps[i].num);
                }
                {
                    ui_obs::ScopedFont rf(fonts_.regular, 12.0f);
                    ImGui::TextColored(ui_obs::palette::TextMuted, "%s", steps[i].title);
                }
                ImGui::EndChild();
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }
            ImGui::EndTable();
        }

        ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(14.0f)));
        std::string val_err;
        const bool valid = validate_inputs_cached(val_err);
        const bool locked_rec = app_state_.is_file_actively_recording(custom_file_path_);
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float btn_h = ui_obs::dp(34.0f);

        if (valid && !locked_rec && selected_filename_[0] != '\0') {
            std::string short_fn(selected_filename_);
            if (short_fn.size() > 24) {
                short_fn = short_fn.substr(0, 14) + "..." + short_fn.substr(short_fn.size() - 7);
            }
            char cta[128];
            std::snprintf(cta, sizeof(cta), "Run H I Pipeline: %s", short_fn.c_str());
            const float pri_w = std::min(avail_w - ui_obs::dp(116.0f), ui_obs::dp(340.0f));
            const float sec_w = ui_obs::dp(106.0f);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (avail_w - (pri_w + sec_w + ui_obs::dp(10.0f))) * 0.5f));

            ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.30f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.50f));
            ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::AccentBlue);
            {
                ui_obs::ScopedFont cf(fonts_.semibold, 13.0f);
                if (ImGui::Button(cta, ImVec2(pri_w, btn_h))) {
                    start_analysis();
                }
                if (ImGui::IsItemHovered() && std::strlen(selected_filename_) > 24) {
                    ImGui::SetTooltip("%s", selected_filename_);
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
            if (ImGui::Button("Browse...", ImVec2(sec_w, btn_h))) {
                std::string picked;
                if (ui_obs::win_fs::browse_for_cs16_file(owner_hwnd_, ui_obs::win_fs::get_default_recordings_dir(), picked)) {
                    select_and_run_file(picked, false);
                }
            }
        } else {
            const float pri_w = ui_obs::dp(210.0f);
            const float sec_w = ui_obs::dp(190.0f);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (avail_w - (pri_w + sec_w + ui_obs::dp(10.0f))) * 0.5f));

            ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.28f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.48f));
            ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::AccentBlue);
            {
                ui_obs::ScopedFont cf(fonts_.semibold, 13.0f);
                if (ImGui::Button("Browse .cs16 Recording...", ImVec2(pri_w, btn_h))) {
                    std::string picked;
                    if (ui_obs::win_fs::browse_for_cs16_file(owner_hwnd_, ui_obs::win_fs::get_default_recordings_dir(), picked)) {
                        select_and_run_file(picked, false);
                    }
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
            if (ImGui::Button("Open Live SDR Recorder", ImVec2(sec_w, btn_h))) {
                app_state_.switch_to_record();
            }
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

bool UiAnalyzePanel::begin_sidebar_card(const char* id, const char* title, const ImVec4& accent,
                                          const char* badge_text, const ImVec4& badge_col) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(8.0f)));
    const bool open = ui_obs::widgets::BeginCard(id, fonts_.semibold, title, accent, badge_text, badge_col);
    return open;
}

void UiAnalyzePanel::end_sidebar_card() {
    ui_obs::widgets::EndCard();
    ImGui::PopStyleVar();
}

void UiAnalyzePanel::render_top_bar(float width, float height, const std::shared_ptr<const AnalysisSnapshot>& snap) {
    const float btn_h = std::max(ImGui::GetFrameHeight(), height - ui_obs::dp(12.0f));
    const float pad_y = std::max(ui_obs::dp(4.0f), (height - btn_h) * 0.5f);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), pad_y));
    ImGui::BeginChild("##AnalyzeTopBar", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const bool running = is_analyzing_.load();
    const bool locked_rec = app_state_.is_file_actively_recording(custom_file_path_);

    // 1. Recording Combo + Browse + Refresh
    ImGui::BeginDisabled(running);
    const float combo_w = std::clamp(width * 0.28f, ui_obs::dp(190.0f), ui_obs::dp(340.0f));
    const char* disp_name = (selected_filename_[0] != '\0') ? selected_filename_ : "-- Select .cs16 File --";

    ImGui::SetNextItemWidth(combo_w);
    if (ImGui::BeginCombo("##Cs16Selector", disp_name)) {
        for (int i = 0; i < static_cast<int>(discovered_files_.size()); ++i) {
            std::string fn = fs::path(discovered_files_[static_cast<size_t>(i)]).filename().string();
            bool active_rec = app_state_.is_file_actively_recording(discovered_files_[static_cast<size_t>(i)]);
            std::string label = active_rec ? (fn + " [RECORDING]") : fn;
            bool is_sel = (selected_file_idx_ == i);
            if (ImGui::Selectable(label.c_str(), is_sel)) {
                selected_file_idx_ = i;
                on_file_selected(discovered_files_[static_cast<size_t>(i)]);
            }
            if (is_sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine(0.0f, ui_obs::dp(6.0f));
    if (ImGui::Button("Browse...", ImVec2(ui_obs::dp(78.0f), btn_h))) {
        std::string picked;
        if (ui_obs::win_fs::browse_for_cs16_file(owner_hwnd_, ui_obs::win_fs::get_default_recordings_dir(), picked)) {
            select_and_run_file(picked, false);
        }
    }
    ImGui::SameLine(0.0f, ui_obs::dp(6.0f));
    if (ImGui::Button("Refresh", ImVec2(ui_obs::dp(70.0f), btn_h))) {
        scan_files();
    }
    ImGui::EndDisabled();

    // 2. Run / Cancel CTA + Clean Status Box or Progress Bar
    ImGui::SameLine(0.0f, ui_obs::dp(8.0f));
    if (!running) {
        std::string val_err;
        const bool valid = validate_inputs_cached(val_err);
        ImGui::BeginDisabled(!valid || locked_rec);
        ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.34f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.52f));
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.75f));
        {
            ui_obs::ScopedFont cf(fonts_.semibold, 13.0f);
            if (ImGui::Button("Run H I Pipeline", ImVec2(ui_obs::dp(146.0f), btn_h))) {
                start_analysis();
            }
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();

        ImGui::SameLine(0.0f, ui_obs::dp(8.0f));
        const float pill_w = ImGui::GetContentRegionAvail().x;
        if (pill_w > ui_obs::dp(60.0f)) {
            const bool dirty = (snap != nullptr) && (current_signature() != last_run_sig_);
            ImVec4 dot_col = ui_obs::palette::TextMuted;
            ImVec4 border_col = ui_obs::palette::BorderSubtle;
            std::string status_txt;

            if (locked_rec) {
                dot_col = ui_obs::palette::Red;
                border_col = ui_obs::palette::Tint(ui_obs::palette::Red, 0.40f);
                status_txt = "File is actively recording - stop SDR capture before analysis.";
            } else if (!valid) {
                dot_col = ui_obs::palette::Amber;
                border_col = ui_obs::palette::Tint(ui_obs::palette::Amber, 0.40f);
                status_txt = val_err;
            } else if (dirty) {
                dot_col = ui_obs::palette::Amber;
                border_col = ui_obs::palette::Tint(ui_obs::palette::Amber, 0.40f);
                status_txt = "Parameters modified - click Run H I Pipeline";
            } else if (snap) {
                dot_col = ui_obs::palette::Emerald;
                status_txt = get_status_message();
            } else {
                status_txt = get_status_message();
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 pmin = ImGui::GetCursorScreenPos();
            const ImVec2 pmax(pmin.x + pill_w, pmin.y + btn_h);
            dl->AddRectFilled(pmin, pmax, ui_obs::palette::BezelBg32, ui_obs::dp(4.0f));
            dl->AddRect(pmin, pmax, ui_obs::palette::U32(border_col), ui_obs::dp(4.0f));
            dl->AddCircleFilled(ImVec2(pmin.x + ui_obs::dp(11.0f), (pmin.y + pmax.y) * 0.5f),
                                ui_obs::dp(3.0f), ui_obs::palette::U32(dot_col), 12);

            const ImVec2 tclip_min(pmin.x + ui_obs::dp(21.0f), pmin.y);
            const ImVec2 tclip_max(pmax.x - ui_obs::dp(8.0f), pmax.y);
            dl->PushClipRect(tclip_min, tclip_max, true);
            {
                ui_obs::ScopedFont sf(fonts_.regular, 12.0f);
                const float th = ImGui::GetTextLineHeight();
                dl->AddText(ImVec2(tclip_min.x, pmin.y + (btn_h - th) * 0.5f),
                            ui_obs::palette::U32(locked_rec || !valid ? dot_col : ui_obs::palette::TextMuted),
                            status_txt.c_str());
            }
            dl->PopClipRect();
            ImGui::Dummy(ImVec2(pill_w, btn_h));
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::Red, 0.30f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::Red, 0.48f));
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(ui_obs::palette::Red, 0.70f));
        if (ImGui::Button("Cancel", ImVec2(ui_obs::dp(80.0f), btn_h))) {
            cancel_analysis();
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine(0.0f, ui_obs::dp(8.0f));
        const float prog_w = std::max(ui_obs::dp(140.0f), ImGui::GetContentRegionAvail().x);
        const ImVec2 pmin = ImGui::GetCursorScreenPos();
        const ImVec2 pmax(pmin.x + prog_w, pmin.y + btn_h);
        const float pct = static_cast<float>(progress_pct_.load() / 100.0);
        char prog_lbl[96];
        std::snprintf(prog_lbl, sizeof(prog_lbl), "%.1f%%  |  %.0f MB/s  |  ETA %.1fs",
                      progress_pct_.load(), progress_speed_.load(), progress_eta_.load());
        draw_gradient_progress_bar(ImGui::GetWindowDrawList(), pmin, pmax, pct, prog_lbl, fonts_.mono);
        ImGui::Dummy(ImVec2(prog_w, btn_h));
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

void UiAnalyzePanel::render_sidebar_controls(float width, float height, const std::shared_ptr<const AnalysisSnapshot>& snap) {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(6.0f)));
    ImGui::BeginChild("##AnalyzeSidebar", ImVec2(width, height), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);

    const bool running = is_analyzing_.load();
    auto param_label = [&](const char* lbl) {
        ui_obs::ScopedFont lf(fonts_.regular, 11.5f);
        ImGui::TextColored(ui_obs::palette::TextMuted, "%s", lbl);
    };
    auto param_gap = [&]() { ImGui::Dummy(ImVec2(0.0f, ui_obs::dp(2.0f))); };

    // Card 1: File & Sidecar Metadata
    if (begin_sidebar_card("##CardMeta", "RECORDING METADATA", ui_obs::palette::TextMuted)) {
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(ui_obs::dp(3.0f), ui_obs::dp(1.5f)));
        if (ImGui::BeginTable("##MetaTbl", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("K", ImGuiTableColumnFlags_WidthFixed, ui_obs::dp(58.0f));
            ImGui::TableSetupColumn("V", ImGuiTableColumnFlags_WidthStretch);

            auto meta_row = [&](const char* key, const char* val, const ImVec4& col = ui_obs::palette::TextPrimary) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                {
                    ui_obs::ScopedFont kf(fonts_.regular, 11.5f);
                    ImGui::TextColored(ui_obs::palette::TextMuted, "%s", key);
                }
                ImGui::TableSetColumnIndex(1);
                {
                    ui_obs::ScopedFont vf(fonts_.mono, 11.5f);
                    ImGui::TextColored(col, "%s", val);
                }
            };

            std::string meta_fn = selected_filename_[0] != '\0' ? std::string(selected_filename_) : std::string("None selected");
            if (meta_fn.size() > 24) {
                meta_fn = meta_fn.substr(0, 13) + "..." + meta_fn.substr(meta_fn.size() - 8);
            }
            meta_row("File", meta_fn.c_str(),
                     selected_filename_[0] != '\0' ? ui_obs::palette::TextPrimary : ui_obs::palette::TextMuted);
            if (ImGui::IsItemHovered() && std::strlen(selected_filename_) > 24) {
                ImGui::SetTooltip("%s", selected_filename_);
            }

            if (selected_file_size_bytes_ > 0) {
                char sz_buf[32];
                format_byte_size(selected_file_size_bytes_, sz_buf, sizeof(sz_buf));
                meta_row("Size", sz_buf);
            }

            if (current_sidecar_.found_sidecar) {
                meta_row("Source", current_sidecar_.sidecar_source.c_str(), ui_obs::palette::Emerald);
                if (!current_sidecar_.device_name.empty()) {
                    std::string dev_str = current_sidecar_.device_name;
                    if (dev_str.size() > 24) {
                        dev_str = dev_str.substr(0, 21) + "...";
                    }
                    meta_row("Device", dev_str.c_str());
                    if (ImGui::IsItemHovered() && current_sidecar_.device_name.size() > 24) {
                        ImGui::SetTooltip("%s", current_sidecar_.device_name.c_str());
                    }
                }
                if (current_sidecar_.has_gain || std::abs(current_sidecar_.lo_offset_khz) > 1e-3) {
                    char g_buf[48];
                    std::snprintf(g_buf, sizeof(g_buf), "%.1f dB | LO %+.0f kHz",
                                  current_sidecar_.gain_db, current_sidecar_.lo_offset_khz);
                    meta_row("RF", g_buf);
                }
                meta_row("Bias-T", current_sidecar_.bias_tee ? "ON 4.5V" : "OFF",
                         current_sidecar_.bias_tee ? ui_obs::palette::Amber : ui_obs::palette::TextMuted);
            } else {
                meta_row("Sidecar", "None (manual RF)", ui_obs::palette::Amber);
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }
    end_sidebar_card();

    // Card 2: RF & Calibration Parameters (compact 2-col layout)
    if (begin_sidebar_card("##CardCalib", "RF & CALIBRATION", ui_obs::palette::TextMuted)) {
        ImGui::BeginDisabled(running);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(2.5f)));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(ui_obs::dp(3.0f), ui_obs::dp(2.0f)));

        if (ImGui::BeginTable("##CalibGrid", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            param_label("Freq (MHz)");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputDouble("##AnaFreq", &ui_center_freq_mhz_, 0.0, 0.0, "%.4f");

            ImGui::TableSetColumnIndex(1);
            param_label("Rate (MSPS)");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputDouble("##AnaRate", &ui_sample_rate_msps_, 0.0, 0.0, "%.3f");

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            param_label("T_sys (K)");
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::InputFloat("##AnaTsys", &ui_tsys_k_, 0.0f, 0.0f, "%.1f")) {
                if (ui_tsys_k_ < 1.0f) ui_tsys_k_ = 1.0f;
            }

            ImGui::TableSetColumnIndex(1);
            param_label("Window (s, 0=All)");
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::InputFloat("##AnaDur", &ui_duration_s_, 0.0f, 0.0f, "%.1f")) {
                if (ui_duration_s_ < 0.0f) ui_duration_s_ = 0.0f;
            }

            ImGui::EndTable();
        }

        ImGui::PopStyleVar(2);
        ImGui::EndDisabled();
    }
    end_sidebar_card();

    // Card 3: Scientific Pipeline Configuration
    if (begin_sidebar_card("##CardPipe", "DSP & BASELINE MODEL", ui_obs::palette::TextMuted)) {
        ImGui::BeginDisabled(running);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(2.5f)));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(ui_obs::dp(3.0f), ui_obs::dp(2.0f)));

        if (ImGui::BeginTable("##PipeGrid", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            static const char* fft_labels[] = {"2048", "4096", "8192", "16384", "32768"};
            param_label("FFT Bins");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::Combo("##AnaFft", &ui_fft_size_idx_, fft_labels, 5);

            ImGui::TableSetColumnIndex(1);
            param_label("DC Excise (kHz)");
            ImGui::SetNextItemWidth(-1.0f);
            if (ui_obs::widgets::SliderFloatAccent("##AnaDc", &ui_dc_clear_khz_, 0.0f, 30.0f, "%.1f", ui_obs::palette::AccentBlue)) {
                if (ui_dc_clear_khz_ < 0.0f) ui_dc_clear_khz_ = 0.0f;
            }
            ImGui::EndTable();
        }
        param_gap();

        static const char* base_labels[] = {"Ripple Filter (Gaussian)", "Asymmetric Least Squares", "Sigma-Clipped Polynomial"};
        param_label("Baseline Removal Method");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::Combo("##AnaBase", &ui_baseline_idx_, base_labels, 3);

        if (ui_baseline_idx_ == 1) {
            param_gap();
            if (ImGui::BeginTable("##AlsGrid", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                param_label("ALS Lambda");
                ImGui::SetNextItemWidth(-1.0f);
                ui_obs::widgets::SliderFloatAccent("##AlsLam", &ui_als_lambda_, 100.0f, 50000.0f, "%.0f", ui_obs::palette::Orange);

                ImGui::TableSetColumnIndex(1);
                param_label("ALS p");
                ImGui::SetNextItemWidth(-1.0f);
                ui_obs::widgets::SliderFloatAccent("##AlsP", &ui_als_p_, 0.001f, 0.1f, "%.3f", ui_obs::palette::Orange);
                ImGui::EndTable();
            }
        } else if (ui_baseline_idx_ == 2) {
            param_gap();
            param_label("Polynomial Degree");
            ImGui::SetNextItemWidth(-1.0f);
            ui_obs::widgets::SliderIntAccent("##PolyDeg", &ui_poly_deg_, 3, 18, "%d", ui_obs::palette::Orange);
        }

        param_gap();
        static const char* line_method_labels[] = {
            "Single Gaussian (L-M)",
            "Double Gaussian (2-Arm)",
            "Voigt Profile (Core+Wings)",
            "Gauss-Hermite (h3/h4 Asym)",
            "Spectral Moments (M0/M1/M2)"
        };
        param_label("Line Profile Analysis");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::Combo("##AnaLineMethod", &ui_line_method_idx_, line_method_labels, 5);

        ImGui::PopStyleVar(2);
        ImGui::EndDisabled();
    }
    end_sidebar_card();

    // Card 4: Waterfall Colormap & Artifact Export
    if (begin_sidebar_card("##CardExport", "COLORMAP & EXPORT", ui_obs::palette::TextMuted)) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(4.0f)));

        static const char* cmaps[] = {"Viridis", "Inferno", "Plasma", "Turbo"};
        param_label("Waterfall Colormap");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("##AnaCmap", &ui_colormap_idx_, cmaps, 4)) {
            active_lut256_ = ui_obs::build_colormap_lut256(ui_colormap_idx_);
        }
        param_gap();

        const bool exporting = is_exporting_.load();
        const bool can_export = (snap != nullptr && !running && !exporting);
        ImGui::BeginDisabled(!can_export);
        ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::Emerald, 0.28f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::Emerald, 0.44f));
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(ui_obs::palette::Emerald, 0.65f));
        {
            ui_obs::ScopedFont bf(fonts_.semibold, 12.5f);
            if (ImGui::Button(exporting ? "Exporting Artifacts..." : "Export All 8 Artifacts", ImVec2(-1.0f, ui_obs::dp(28.0f)))) {
                start_export(snap);
            }
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();

        if (ImGui::Button("Open Output Directory", ImVec2(-1.0f, ui_obs::dp(26.0f)))) {
            std::string out_dir = snap ? get_output_dir_for_source(snap->source_filepath) : get_exe_parent_output_dir();
            ui_obs::win_fs::open_folder_in_explorer(out_dir);
        }

        const std::string exp_msg = get_export_message();
        if (!exp_msg.empty()) {
            const bool is_err = (exp_msg.find("fail") != std::string::npos) || (exp_msg.find("Error") != std::string::npos);
            const ImVec4 st_col = is_err ? ui_obs::palette::Red : ui_obs::palette::Emerald;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Tint(st_col, 0.10f));
            ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::Tint(st_col, 0.40f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(8.0f), ui_obs::dp(5.0f)));
            ImGui::BeginChild("##ExpMsgBox", ImVec2(-1.0f, 0.0f), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
            {
                ui_obs::ScopedFont ef(fonts_.regular, 11.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, st_col);
                ImGui::TextWrapped("%s", exp_msg.c_str());
                ImGui::PopStyleColor();
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
        }

        ImGui::PopStyleVar();
    }
    end_sidebar_card();

    ImGui::EndChild(); // ##AnalyzeSidebar
    ImGui::PopStyleVar();
}

void UiAnalyzePanel::render_kpi_strip(float width, const AstrophysicalMetrics& m, const AstroConfig& cfg) {
    (void)width;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(ui_obs::dp(4.0f), 0.0f));
    if (!ImGui::BeginTable("##AnaKpiStrip", 6, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::PopStyleVar();
        return;
    }

    auto kpi_card = [&](int col, const char* child_id, const char* label,
                        const char* val_str, const char* unit_str, const char* sub_str,
                        ImVec4 val_col, const char* status_tag = nullptr, ImVec4 tag_col = ui_obs::palette::TextMuted) {
        ImGui::TableSetColumnIndex(col);

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(6.0f)));
        ImGui::BeginChild(child_id, ImVec2(-1.0f, ui_obs::dp(60.0f)), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 cmin = ImGui::GetWindowPos();
        const ImVec2 cmax(cmin.x + ImGui::GetWindowWidth(), cmin.y + ImGui::GetWindowHeight());

        dl->PushClipRect(cmin, cmax, true);

        // Row 1: Uppercase micro-label + optional plain inline status tag
        {
            ui_obs::ScopedFont lf(fonts_.semibold, 10.5f);
            dl->AddText(ImVec2(cmin.x + ui_obs::dp(10.0f), cmin.y + ui_obs::dp(6.0f)),
                        ui_obs::palette::U32(ui_obs::palette::TextMuted), label);
            if (status_tag && *status_tag) {
                const ImVec2 psz = ImGui::CalcTextSize(status_tag);
                dl->AddText(ImVec2(cmax.x - psz.x - ui_obs::dp(9.0f), cmin.y + ui_obs::dp(6.0f)),
                            ui_obs::palette::U32(tag_col), status_tag);
            }
        }

        // Row 2: Mono primary readout + muted unit
        ImVec2 vsz(0.0f, 0.0f);
        const ImVec2 vpos(cmin.x + ui_obs::dp(10.0f), cmin.y + ui_obs::dp(20.0f));
        {
            ui_obs::ScopedFont vf(fonts_.mono, 18.0f);
            vsz = ImGui::CalcTextSize(val_str);
            dl->AddText(vpos, ui_obs::palette::U32(val_col), val_str);
        }
        if (unit_str && *unit_str) {
            ui_obs::ScopedFont uf(fonts_.regular, 11.0f);
            const ImVec2 usz = ImGui::CalcTextSize(unit_str);
            dl->AddText(ImVec2(vpos.x + vsz.x + ui_obs::dp(4.0f), vpos.y + vsz.y - usz.y - ui_obs::dp(2.0f)),
                        ui_obs::palette::U32(ui_obs::palette::TextMuted), unit_str);
        }

        // Row 3: Physical diagnostic sub-caption
        if (sub_str && *sub_str) {
            ui_obs::ScopedFont sf(fonts_.mono, 10.5f);
            dl->AddText(ImVec2(cmin.x + ui_obs::dp(10.0f), cmax.y - ImGui::GetFontSize() - ui_obs::dp(5.0f)),
                        ui_obs::palette::U32(ui_obs::palette::TextDim), sub_str);
        }
        dl->PopClipRect();

        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    };

    ImGui::TableNextRow();
    char val[48], sub[64];

    std::snprintf(val, sizeof(val), "%.4f", m.peak_freq_mhz);
    std::snprintf(sub, sizeof(sub), "LO off: %+.1f kHz", m.lo_offset_khz);
    kpi_card(0, "##KpiFreq", "PEAK FREQ", val, "MHz", sub, ui_obs::palette::TextPrimary);

    std::snprintf(val, sizeof(val), "%+.2f", m.peak_vel_kms);
    std::snprintf(sub, sizeof(sub), "Sep: %d bins", m.bin_separation);
    kpi_card(1, "##KpiVel", "DOPPLER V_LSR", val, "km/s", sub, ui_obs::palette::TextPrimary);

    std::snprintf(val, sizeof(val), "%.2f", m.peak_temp_k);
    std::snprintf(sub, sizeof(sub), "%+.2f dB | rms %.2f K", m.peak_power_db, m.noise_std_k);
    kpi_card(2, "##KpiTa", "BRIGHTNESS T_A", val, "K", sub, ui_obs::palette::TextPrimary);

    const ImVec4 snr_col = (m.snr >= 5.0) ? ui_obs::palette::Emerald
                         : (m.snr >= 3.0) ? ui_obs::palette::Cyan
                         : (m.snr >= 2.0) ? ui_obs::palette::Amber
                                          : ui_obs::palette::Red;
    const char* snr_tag = (m.snr >= 5.0) ? "DETECTED"
                        : (m.snr >= 3.0) ? "MARGINAL"
                        : (m.snr >= 2.0) ? "WEAK"
                                         : "NOISE";
    std::snprintf(val, sizeof(val), "%.2f", m.snr);
    std::snprintf(sub, sizeof(sub), "Floor: %.3f dB", m.noise_std_db);
    kpi_card(3, "##KpiSnr", "DETECTION SNR", val, "sigma", sub, snr_col, snr_tag, snr_col);

    std::snprintf(val, sizeof(val), "%.2f", m.fwhm_kms);
    if (!m.gauss_fit.success) {
        std::snprintf(sub, sizeof(sub), "Raw estimate");
    } else if (m.gauss_fit.method == LineAnalysisMethod::DoubleGaussian && m.gauss_fit.has_secondary) {
        std::snprintf(sub, sizeof(sub), "2-Arm: %+.1f / %+.1f", m.gauss_fit.v_center_kms, m.gauss_fit.v2_center_kms);
    } else if (m.gauss_fit.method == LineAnalysisMethod::VoigtProfile) {
        std::snprintf(sub, sizeof(sub), "Voigt eta: %.2f | sig %.1f", m.gauss_fit.voigt_eta, m.sigma_kms);
    } else if (m.gauss_fit.method == LineAnalysisMethod::GaussHermite) {
        std::snprintf(sub, sizeof(sub), "h3:%+.2f h4:%+.2f", m.gauss_fit.gh_h3, m.gauss_fit.gh_h4);
    } else if (m.gauss_fit.method == LineAnalysisMethod::SpectralMoments) {
        std::snprintf(sub, sizeof(sub), "M2 sig: %.2f | M1 %+.1f", m.gauss_fit.moment2_sigma_kms, m.gauss_fit.moment1_kms);
    } else {
        std::snprintf(sub, sizeof(sub), "sigma: %.2f | Fit OK", m.sigma_kms);
    }
    kpi_card(4, "##KpiFwhm", "LINE FWHM", val, "km/s", sub, ui_obs::palette::TextPrimary);

    const bool has_tsys = (cfg.system_temp_k > 0.0);
    std::snprintf(val, sizeof(val), "%.2e", m.column_density_cm2);
    std::snprintf(sub, sizeof(sub), "W: %.1f K*km/s", m.integrated_flux_k_kms);
    kpi_card(5, "##KpiNhi", "N_HI COLUMN", val, "cm^-2", sub,
             has_tsys ? ui_obs::palette::TextPrimary : ui_obs::palette::TextMuted);

    ImGui::EndTable();
    ImGui::PopStyleVar();
}

void UiAnalyzePanel::render_tab_calibrated_spectrum(const std::shared_ptr<const AnalysisSnapshot>& snap) {
    render_calibrated_spectrum_plot(snap, ImGui::GetContentRegionAvail(), false, "##CalSpecHit", zoom_cal_spec_);
}

void UiAnalyzePanel::render_calibrated_spectrum_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                                     ImVec2 avail, bool compact,
                                                     const char* hit_id, ui_obs::PlotZoomState& zoom) {
    if (avail.x < ui_obs::dp(80.0f) || avail.y < ui_obs::dp(60.0f) || snap->vel_kms.size() < 2) return;

    const size_t n = snap->vel_kms.size();
    const size_t edge_bins = std::min(n / 4, static_cast<size_t>(n * std::clamp(snap->config_used.passband_edge_fraction, 0.0, 0.45)));
    const double v_a = snap->vel_kms[edge_bins];
    const double v_b = snap->vel_kms[n - 1 - edge_bins];
    const double v_lo_valid = std::min(v_a, v_b);
    const double v_hi_valid = std::max(v_a, v_b);
    const double v_lo_full = std::min(snap->vel_kms.front(), snap->vel_kms.back());
    const double v_hi_full = std::max(snap->vel_kms.front(), snap->vel_kms.back());
    const double def_xmin = (v_hi_valid - v_lo_valid > 400.0) ? std::max(v_lo_valid, -200.0) : v_lo_valid;
    const double def_xmax = (v_hi_valid - v_lo_valid > 400.0) ? std::min(v_hi_valid,  200.0) : v_hi_valid;

    double def_ymin = 1e9, def_ymax = -1e9;
    for (size_t i = 0; i < n && i < snap->calibrated_k.size(); ++i) {
        const double v = snap->vel_kms[i];
        if (v < def_xmin || v > def_xmax) continue;
        def_ymin = std::min(def_ymin, snap->calibrated_k[i]);
        def_ymax = std::max(def_ymax, snap->calibrated_k[i]);
    }
    if (def_ymin >= def_ymax) { def_ymin = -5.0; def_ymax = 15.0; }
    const double pad = std::max(1.0, (def_ymax - def_ymin) * 0.14);
    def_ymin -= pad;
    def_ymax += pad;

    const double x_min = zoom.active ? zoom.x_min : def_xmin;
    const double x_max = zoom.active ? zoom.x_max : def_xmax;
    const double ymin  = zoom.active ? zoom.y_min : def_ymin;
    const double ymax  = zoom.active ? zoom.y_max : def_ymax;

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    ui_obs::PlotMargin m;
    if (compact) {
        m.left = ui_obs::dp(54.0f); m.right = ui_obs::dp(16.0f);
        m.top  = ui_obs::dp(22.0f); m.bottom = ui_obs::dp(28.0f);
    } else {
        m.left = ui_obs::dp(72.0f); m.right = ui_obs::dp(22.0f);
        m.top  = ui_obs::dp(42.0f); m.bottom = ui_obs::dp(38.0f);
    }

    auto vp = ui_obs::PlotViewport2D::create(origin, avail, x_min, x_max, ymin, ymax, m);
    vp.draw_background_and_grid(dl, compact ? 5 : 7, compact ? 4 : 5, "%.0f", "%.1f",
                                compact ? "V_LSR (km/s)" : "Doppler Radial Velocity V_LSR (km/s)",
                                "T_A (K)", fonts_.mono);

    if (!compact) {
        const double f_top_left  = HYDROGEN_REST_FREQ_MHZ * (1.0 - x_min / SPEED_OF_LIGHT_KMS);
        const double f_top_right = HYDROGEN_REST_FREQ_MHZ * (1.0 - x_max / SPEED_OF_LIGHT_KMS);
        vp.draw_top_axis(dl, f_top_left, f_top_right, 7, "%.3f", "Topocentric Sky Frequency (MHz)", fonts_.mono);
    }

    // 1. Reference 0 K line & 0 km/s rest-frame marker
    dl->PushClipRect(vp.plot_min, vp.plot_max, true);
    if (ymin < 0.0 && ymax > 0.0) {
        const float zy = vp.map_y(0.0);
        dl->AddLine(ImVec2(vp.plot_min.x, zy), ImVec2(vp.plot_max.x, zy),
                    ui_obs::palette::U32(ui_obs::palette::TextDim, 0.78f), 1.0f);
    }
    if (0.0 >= x_min && 0.0 <= x_max) {
        const float zx = vp.map_x(0.0);
        ui_obs::widgets::draw_dashed_vline(dl, zx, vp.plot_min.y, vp.plot_max.y,
                                           ui_obs::palette::U32(ui_obs::palette::Amber, 0.65f), 1.2f);
    }

    auto vel_at = [&](size_t i) { return snap->vel_kms[i]; };
    auto ta_at  = [&](size_t i) { return snap->calibrated_k[i]; };

    // 2. +T_A gradient glow fill
    vp.draw_area_fill_above_ref(dl, n, vel_at, ta_at, 0.0,
                                ui_obs::palette::Cyan, 0.02f, 0.34f, fill_scratch_);

    // 3. Main calibrated spectrum trace
    vp.draw_decimated_xy(dl, n, vel_at, ta_at,
                         ui_obs::palette::U32(ui_obs::palette::Cyan), ui_obs::dp(1.6f), trace_scratch_);

    // 4. Line profile fit overlay + secondary component curves + FWHM half-max bracket
    const auto& gfit = snap->metrics.gauss_fit;
    if (gfit.success && gfit.fitted_curve.size() == n) {
        if (gfit.method == LineAnalysisMethod::DoubleGaussian && gfit.has_secondary &&
            gfit.component1_curve.size() == n && gfit.component2_curve.size() == n) {
            vp.draw_decimated_xy(dl, n, vel_at,
                                 [&](size_t i) { return gfit.component1_curve[i]; },
                                 ui_obs::palette::U32(ui_obs::palette::Orange, 0.80f), ui_obs::dp(1.3f), trace_scratch_);
            vp.draw_decimated_xy(dl, n, vel_at,
                                 [&](size_t i) { return gfit.component2_curve[i]; },
                                 ui_obs::palette::U32(ui_obs::palette::Purple, 0.85f), ui_obs::dp(1.3f), trace_scratch_);
        }

        vp.draw_decimated_xy(dl, n, vel_at,
                             [&](size_t i) { return gfit.fitted_curve[i]; },
                             ui_obs::palette::U32(ui_obs::palette::Pink), ui_obs::dp(1.8f), trace_scratch_);

        if (gfit.fwhm_kms > 0.0 && gfit.amplitude_k > 0.0) {
            const float hy = vp.map_y(0.5 * gfit.amplitude_k);
            const float x_l = vp.map_x(gfit.v_center_kms - 0.5 * gfit.fwhm_kms);
            const float x_r = vp.map_x(gfit.v_center_kms + 0.5 * gfit.fwhm_kms);
            const ImU32 brk_col = ui_obs::palette::U32(ui_obs::palette::Pink, 0.80f);
            ui_obs::widgets::draw_dashed_hline(dl, std::min(x_l, x_r), std::max(x_l, x_r), hy, brk_col, 1.3f, 4.0f, 3.0f);
            dl->AddLine(ImVec2(x_l, hy - ui_obs::dp(4.0f)), ImVec2(x_l, hy + ui_obs::dp(4.0f)), brk_col, 1.3f);
            dl->AddLine(ImVec2(x_r, hy - ui_obs::dp(4.0f)), ImVec2(x_r, hy + ui_obs::dp(4.0f)), brk_col, 1.3f);
        }
    }

    // 5. Peak marker drop-line & dot
    const double pk_v = snap->metrics.peak_vel_kms;
    const double pk_t = snap->metrics.peak_temp_k;
    if (pk_v >= x_min && pk_v <= x_max && pk_t >= ymin && pk_t <= ymax) {
        const float px = vp.map_x(pk_v);
        const float py = vp.map_y(pk_t);
        const float py0 = std::clamp(vp.map_y(0.0), vp.plot_min.y, vp.plot_max.y);
        if (py0 > py) {
            ui_obs::widgets::draw_dashed_vline(dl, px, py, py0, ui_obs::palette::U32(ui_obs::palette::Red, 0.50f), 1.0f);
        }
        dl->AddCircleFilled(ImVec2(px, py), ui_obs::dp(3.8f), ui_obs::palette::U32(ui_obs::palette::Red), 16);
        dl->AddCircle(ImVec2(px, py), ui_obs::dp(4.2f), IM_COL32(255, 255, 255, 220), 16, 1.1f);
    }

    // 6. Smart Fit Summary Card (full mode only, auto-placed opposite peak side)
    if (!compact) {
        const float cw = ui_obs::dp(212.0f);
        const float ch = ui_obs::dp(128.0f);
        const bool peak_on_left = (pk_v < 0.5 * (x_min + x_max));
        const float cx = peak_on_left ? (vp.plot_max.x - cw - ui_obs::dp(10.0f)) : (vp.plot_min.x + ui_obs::dp(10.0f));
        const float cy = vp.plot_min.y + ui_obs::dp(10.0f);
        const ImVec2 cmin(cx, cy), cmax(cx + cw, cy + ch);

        dl->AddRectFilled(cmin, cmax, ui_obs::palette::CardBg32, ui_obs::dp(4.0f));
        dl->AddRect(cmin, cmax, ui_obs::palette::PlotBorder32, ui_obs::dp(4.0f), 1.0f, ImDrawFlags_None);
        const ImVec4 title_col = gfit.success ? ui_obs::palette::Pink : ui_obs::palette::Cyan;

        const char* card_hdr = "PEAK H I ESTIMATE";
        if (gfit.success) {
            switch (gfit.method) {
                case LineAnalysisMethod::DoubleGaussian:  card_hdr = "DOUBLE GAUSSIAN (2-ARM)"; break;
                case LineAnalysisMethod::VoigtProfile:    card_hdr = "PSEUDO-VOIGT PROFILE"; break;
                case LineAnalysisMethod::GaussHermite:    card_hdr = "GAUSS-HERMITE (h3/h4)"; break;
                case LineAnalysisMethod::SpectralMoments: card_hdr = "SPECTRAL MOMENTS (M0-M2)"; break;
                default:                                  card_hdr = "GAUSSIAN H I FIT (L-M)"; break;
            }
        }
        {
            ui_obs::ScopedFont hf(fonts_.semibold, 11.0f);
            dl->AddText(ImVec2(cmin.x + ui_obs::dp(10.0f), cmin.y + ui_obs::dp(7.0f)),
                        ui_obs::palette::U32(title_col), card_hdr);
        }

        char v1[32], v2[32], v3[32], v4[32], v5[32], v6[32];
        const char* k6 = "RMSE";
        std::snprintf(v1, sizeof(v1), "%.2f K", gfit.success ? gfit.amplitude_k : snap->metrics.peak_temp_k);
        std::snprintf(v2, sizeof(v2), "%+.2f km/s", gfit.success ? gfit.v_center_kms : snap->metrics.peak_vel_kms);
        std::snprintf(v3, sizeof(v3), "%.2f km/s", snap->metrics.fwhm_kms);
        std::snprintf(v4, sizeof(v4), "%.2f sigma", snap->metrics.snr);
        std::snprintf(v5, sizeof(v5), "%.2e", snap->metrics.column_density_cm2);
        if (gfit.method == LineAnalysisMethod::DoubleGaussian && gfit.has_secondary) {
            k6 = "Arm 2";
            std::snprintf(v6, sizeof(v6), "%.1fK @ %+.1f", gfit.amp2_k, gfit.v2_center_kms);
        } else if (gfit.method == LineAnalysisMethod::VoigtProfile) {
            k6 = "Wings";
            std::snprintf(v6, sizeof(v6), "eta=%.2f g=%.1f", gfit.voigt_eta, gfit.lorentz_gamma_kms);
        } else if (gfit.method == LineAnalysisMethod::GaussHermite) {
            k6 = "h3/h4";
            std::snprintf(v6, sizeof(v6), "%+.2f / %+.2f", gfit.gh_h3, gfit.gh_h4);
        } else if (gfit.method == LineAnalysisMethod::SpectralMoments) {
            k6 = "M1/M2";
            std::snprintf(v6, sizeof(v6), "%+.1f / %.1f", gfit.moment1_kms, gfit.moment2_sigma_kms);
        } else {
            std::snprintf(v6, sizeof(v6), "%.3f K", gfit.rmse_k);
        }

        const struct { const char* k; const char* v; ImU32 c; } rows[6] = {
            {"Amp",   v1, ui_obs::palette::TextPrimary32},
            {"V_LSR", v2, ui_obs::palette::TextPrimary32},
            {"FWHM",  v3, ui_obs::palette::TextPrimary32},
            {k6,      v6, ui_obs::palette::U32(ui_obs::palette::Orange)},
            {"SNR",   v4, ui_obs::palette::U32(snap->metrics.snr >= 5.0 ? ui_obs::palette::Emerald : ui_obs::palette::Amber)},
            {"N_HI",  v5, ui_obs::palette::TextPrimary32}
        };

        float ry = cmin.y + ui_obs::dp(25.0f);
        const float row_h = ui_obs::dp(16.0f);
        for (int r = 0; r < 6; ++r) {
            {
                ui_obs::ScopedFont uf(fonts_.regular, 12.0f);
                dl->AddText(ImVec2(cmin.x + ui_obs::dp(10.0f), ry), ui_obs::palette::TextMuted32, rows[r].k);
            }
            {
                ui_obs::ScopedFont mf(fonts_.mono, 12.0f);
                const ImVec2 vsz = ImGui::CalcTextSize(rows[r].v);
                dl->AddText(ImVec2(cmax.x - ui_obs::dp(10.0f) - vsz.x, ry), rows[r].c, rows[r].v);
            }
            ry += row_h;
        }
    }

    if (zoom.active) {
        const auto pill = ui_obs::widgets::compute_clamped_pill_rect(
            "ZOOMED | Dbl-click reset", fonts_.semibold, 10.5f,
            vp.plot_max.x - ui_obs::dp(82.0f), vp.plot_max.y - ui_obs::dp(24.0f),
            vp.plot_min, vp.plot_max);
        ui_obs::widgets::draw_pill_badge(dl, pill, "ZOOMED | Dbl-click reset",
                                         fonts_.semibold, 10.5f,
                                         ui_obs::palette::PillBg32,
                                         ui_obs::palette::U32(ui_obs::palette::Cyan, 0.50f),
                                         ui_obs::palette::U32(ui_obs::palette::Cyan));
    }
    dl->PopClipRect();

    // 7. Interactive zoom/pan & HUD crosshair card
    ImGui::SetCursorScreenPos(origin);
    ImGui::InvisibleButton(hit_id, avail);
    ui_obs::handle_plot_zoom_pan(vp.plot_min, vp.plot_max, v_lo_full, v_hi_full, def_ymin, def_ymax, zoom, true, false);

    if (ImGui::IsItemHovered() && vp.contains(ImGui::GetIO().MousePos)) {
        const ImVec2 mp = ImGui::GetIO().MousePos;
        ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp.plot_min.y, vp.plot_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
        ui_obs::widgets::draw_dashed_hline(dl, vp.plot_min.x, vp.plot_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);

        const double vh = vp.unmap_x(mp.x);
        const size_t idx = find_nearest_index(snap->vel_kms, vh);
        const double v_bin = snap->vel_kms[idx];
        const double ta_bin = snap->calibrated_k[idx];
        const double fh = HYDROGEN_REST_FREQ_MHZ * (1.0 - v_bin / SPEED_OF_LIGHT_KMS);

        char b_vel[32], b_freq[32], b_ta[32], b_fit[32];
        std::snprintf(b_vel,  sizeof(b_vel),  "%+.2f km/s", v_bin);
        std::snprintf(b_freq, sizeof(b_freq), "%.4f MHz",  fh);
        std::snprintf(b_ta,   sizeof(b_ta),   "%.2f K",    ta_bin);

        ui_obs::widgets::TelemetryRow rows[4] = {
            {"V_LSR", b_vel,  ui_obs::palette::TextPrimary32},
            {"Freq",  b_freq, ui_obs::palette::U32(ui_obs::palette::Cyan)},
            {"T_A",   b_ta,   ui_obs::palette::U32(ui_obs::palette::Emerald)},
            {"Fit",   b_fit,  ui_obs::palette::U32(ui_obs::palette::Pink)}
        };
        int num_rows = 3;
        if (gfit.success && gfit.fitted_curve.size() == n) {
            std::snprintf(b_fit, sizeof(b_fit), "%.2f K", gfit.fitted_curve[idx]);
            num_rows = 4;
        }
        ui_obs::widgets::draw_crosshair_card(dl, mp, vp.plot_min, vp.plot_max,
                                             "CALIBRATED H I", ui_obs::palette::U32(ui_obs::palette::Cyan),
                                             rows, num_rows, fonts_.regular, fonts_.mono);
    }
}

void UiAnalyzePanel::render_tab_raw_and_baseline(const std::shared_ptr<const AnalysisSnapshot>& snap) {
    render_raw_baseline_plot(snap, ImGui::GetContentRegionAvail(), false, "##RawBaseHit", zoom_raw_base_);
}

void UiAnalyzePanel::render_raw_baseline_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                              ImVec2 avail, bool compact,
                                              const char* hit_id, ui_obs::PlotZoomState& zoom) {
    if (avail.x < ui_obs::dp(80.0f) || avail.y < ui_obs::dp(60.0f) || snap->stream.freq_mhz.size() < 2) return;

    const size_t n = snap->stream.freq_mhz.size();
    const double f_min_full = snap->stream.freq_mhz.front();
    const double f_max_full = snap->stream.freq_mhz.back();
    const double x_min = zoom.active ? zoom.x_min : f_min_full;
    const double x_max = zoom.active ? zoom.x_max : f_max_full;

    size_t i_lo = find_nearest_index(snap->stream.freq_mhz, x_min);
    size_t i_hi = find_nearest_index(snap->stream.freq_mhz, x_max);
    if (i_lo > i_hi) std::swap(i_lo, i_hi);
    if (i_hi <= i_lo) { i_lo = 0; i_hi = n - 1; }

    double y1_min = 1e9, y1_max = -1e9;
    double y2_min = 1e9, y2_max = -1e9;
    const bool has_base = (snap->baseline_db.size() == n);
    const bool has_res  = (snap->residual_db.size() == n);
    for (size_t i = i_lo; i <= i_hi; ++i) {
        const double r = snap->raw_db[i];
        y1_min = std::min(y1_min, r);
        y1_max = std::max(y1_max, r);
        if (has_base) {
            const double b = snap->baseline_db[i];
            y1_min = std::min(y1_min, b);
            y1_max = std::max(y1_max, b);
            const double d = has_res ? snap->residual_db[i] : (r - b);
            y2_min = std::min(y2_min, d);
            y2_max = std::max(y2_max, d);
        }
    }
    if (y1_min >= y1_max) { y1_min = -80.0; y1_max = 0.0; }
    const double y1_pad = std::max(0.8, (y1_max - y1_min) * 0.10);
    y1_min -= y1_pad; y1_max += y1_pad;

    if (y2_min >= y2_max) { y2_min = -0.5; y2_max = 0.8; }
    else {
        const double y2_pad = std::max(0.15, (y2_max - y2_min) * 0.14);
        y2_min = std::min(y2_min - y2_pad, -0.35);
        y2_max = std::max(y2_max + y2_pad,  0.55);
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    auto draw_passband_overlays = [&](const ui_obs::PlotViewport2D& vp, bool draw_labels) {
        dl->PushClipRect(vp.plot_min, vp.plot_max, true);
        const double span_f = f_max_full - f_min_full;
        const double edge_f = std::clamp(snap->config_used.passband_edge_fraction, 0.0, 0.25) * span_f;
        if (edge_f > 0.0) {
            const ImU32 skirt_col = ui_obs::palette::U32(ui_obs::palette::TextDim, 0.11f);
            if (f_min_full + edge_f > x_min) {
                const float xr = std::min(vp.plot_max.x, vp.map_x(f_min_full + edge_f));
                dl->AddRectFilled(vp.plot_min, ImVec2(xr, vp.plot_max.y), skirt_col);
                if (draw_labels && (xr - vp.plot_min.x) > ui_obs::dp(64.0f)) {
                    ui_obs::ScopedFont mf(fonts_.semibold, 10.5f);
                    dl->AddText(ImVec2(vp.plot_min.x + ui_obs::dp(6.0f), vp.plot_min.y + ui_obs::dp(6.0f)),
                                ui_obs::palette::TextMuted32, "Filter Skirt");
                }
            }
            if (f_max_full - edge_f < x_max) {
                const float xl = std::max(vp.plot_min.x, vp.map_x(f_max_full - edge_f));
                dl->AddRectFilled(ImVec2(xl, vp.plot_min.y), vp.plot_max, skirt_col);
            }
        }

        const double hi_mhz = HYDROGEN_REST_FREQ_MHZ;
        const double hi_hw  = snap->config_used.hi_protection_half_width_khz * 1e-3;
        if (hi_mhz + hi_hw >= x_min && hi_mhz - hi_hw <= x_max) {
            const float x0 = std::clamp(vp.map_x(hi_mhz - hi_hw), vp.plot_min.x, vp.plot_max.x);
            const float x1 = std::clamp(vp.map_x(hi_mhz + hi_hw), vp.plot_min.x, vp.plot_max.x);
            dl->AddRectFilled(ImVec2(x0, vp.plot_min.y), ImVec2(x1, vp.plot_max.y),
                              ui_obs::palette::U32(ui_obs::palette::Amber, 0.08f));
            const float hx = vp.map_x(hi_mhz);
            ui_obs::widgets::draw_dashed_vline(dl, hx, vp.plot_min.y, vp.plot_max.y,
                                               ui_obs::palette::U32(ui_obs::palette::Amber, 0.72f), 1.2f);
            if (draw_labels) {
                const auto p = ui_obs::widgets::compute_clamped_pill_rect(
                    "H I 1420.406", fonts_.mono, 10.5f, hx, vp.plot_min.y + ui_obs::dp(6.0f),
                    vp.plot_min, vp.plot_max);
                ui_obs::widgets::draw_pill_badge(dl, p, "H I 1420.406", fonts_.mono, 10.5f,
                                                 ui_obs::palette::PillBg32,
                                                 ui_obs::palette::U32(ui_obs::palette::Amber, 0.55f),
                                                 ui_obs::palette::U32(ui_obs::palette::Amber));
            }
        }

        const double lo_mhz = snap->config_used.center_freq_hz / 1e6;
        if (lo_mhz >= x_min && lo_mhz <= x_max) {
            const float lox = vp.map_x(lo_mhz);
            ui_obs::widgets::draw_dashed_vline(dl, lox, vp.plot_min.y, vp.plot_max.y,
                                               ui_obs::palette::U32(ui_obs::palette::Red, 0.55f), 1.0f);
        }
        dl->PopClipRect();
    };

    if (compact) {
        ui_obs::PlotMargin m{ui_obs::dp(54.0f), ui_obs::dp(16.0f), ui_obs::dp(22.0f), ui_obs::dp(28.0f)};
        auto vp = ui_obs::PlotViewport2D::create(origin, avail, x_min, x_max, y1_min, y1_max, m);
        vp.draw_background_and_grid(dl, 5, 4, "%.2f", "%.1f", "Frequency (MHz)", "dB", fonts_.mono);
        draw_passband_overlays(vp, false);

        vp.draw_decimated_trace(dl, n, f_min_full, f_max_full,
                                [&](size_t i) { return snap->raw_db[i]; },
                                ui_obs::palette::U32(ui_obs::palette::Cyan), ui_obs::dp(1.5f), trace_scratch_);
        if (has_base) {
            vp.draw_decimated_trace(dl, n, f_min_full, f_max_full,
                                    [&](size_t i) { return snap->baseline_db[i]; },
                                    ui_obs::palette::U32(ui_obs::palette::Orange), ui_obs::dp(2.0f), trace_scratch_);
        }

        ImGui::SetCursorScreenPos(origin);
        ImGui::InvisibleButton(hit_id, avail);
        ui_obs::handle_plot_zoom_pan(vp.plot_min, vp.plot_max, f_min_full, f_max_full, y1_min, y1_max, zoom, true, true);
        if (ImGui::IsItemHovered() && vp.contains(ImGui::GetIO().MousePos)) {
            const ImVec2 mp = ImGui::GetIO().MousePos;
            ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp.plot_min.y, vp.plot_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
            ui_obs::widgets::draw_dashed_hline(dl, vp.plot_min.x, vp.plot_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);
            const size_t idx = find_nearest_index(snap->stream.freq_mhz, vp.unmap_x(mp.x));
            char b_f[32], b_r[32], b_b[32], b_d[32];
            std::snprintf(b_f, sizeof(b_f), "%.4f MHz", snap->stream.freq_mhz[idx]);
            std::snprintf(b_r, sizeof(b_r), "%.2f dB",  snap->raw_db[idx]);
            const double bval = has_base ? snap->baseline_db[idx] : 0.0;
            std::snprintf(b_b, sizeof(b_b), "%.2f dB",  bval);
            std::snprintf(b_d, sizeof(b_d), "%+.2f dB", snap->raw_db[idx] - bval);
            const ui_obs::widgets::TelemetryRow rows[4] = {
                {"Freq",  b_f, ui_obs::palette::U32(ui_obs::palette::Cyan)},
                {"Raw",   b_r, ui_obs::palette::TextPrimary32},
                {"Base",  b_b, ui_obs::palette::U32(ui_obs::palette::Orange)},
                {"Delta", b_d, ui_obs::palette::U32(ui_obs::palette::Emerald)}
            };
            ui_obs::widgets::draw_crosshair_card(dl, mp, vp.plot_min, vp.plot_max,
                                                 "PASSBAND", ui_obs::palette::U32(ui_obs::palette::Orange),
                                                 rows, has_base ? 4 : 2, fonts_.regular, fonts_.mono);
        }
        return;
    }

    // Full mode: synchronized 64% Top (Passband + Baseline) / 36% Bottom (Flattened Residuals)
    const float split_gap = ui_obs::dp(6.0f);
    const float top_h = std::floor((avail.y - split_gap) * 0.64f);
    const float bot_h = std::max(ui_obs::dp(40.0f), avail.y - split_gap - top_h);
    const ImVec2 bot_origin(origin.x, origin.y + top_h + split_gap);

    ui_obs::PlotMargin m_top{ui_obs::dp(64.0f), ui_obs::dp(22.0f), ui_obs::dp(16.0f), ui_obs::dp(16.0f)};
    ui_obs::PlotMargin m_bot{ui_obs::dp(64.0f), ui_obs::dp(22.0f), ui_obs::dp(16.0f), ui_obs::dp(36.0f)};

    auto vp_top = ui_obs::PlotViewport2D::create(origin, ImVec2(avail.x, top_h), x_min, x_max, y1_min, y1_max, m_top);
    vp_top.draw_background_and_grid(dl, 7, 5, "%.2f", "%.1f", nullptr, "Power (dB)", fonts_.mono);
    draw_passband_overlays(vp_top, true);

    vp_top.draw_decimated_trace(dl, n, f_min_full, f_max_full,
                                [&](size_t i) { return snap->raw_db[i]; },
                                ui_obs::palette::U32(ui_obs::palette::Cyan), ui_obs::dp(1.5f), trace_scratch_);
    if (has_base) {
        vp_top.draw_decimated_trace(dl, n, f_min_full, f_max_full,
                                    [&](size_t i) { return snap->baseline_db[i]; },
                                    ui_obs::palette::U32(ui_obs::palette::Orange), ui_obs::dp(1.8f), trace_scratch_);
    }

    // Compact legend pill top-right
    {
        const float lw = ui_obs::dp(248.0f);
        const float lh = ui_obs::dp(22.0f);
        const ImVec2 lmin(vp_top.plot_max.x - lw - ui_obs::dp(8.0f), vp_top.plot_min.y + ui_obs::dp(6.0f));
        const ImVec2 lmax(lmin.x + lw, lmin.y + lh);
        dl->AddRectFilled(lmin, lmax, ui_obs::palette::PillBg32, ui_obs::dp(4.0f));
        dl->AddRect(lmin, lmax, ui_obs::palette::PlotBorder32, ui_obs::dp(4.0f), 1.0f, ImDrawFlags_None);
        ui_obs::ScopedFont lf(fonts_.regular, 11.0f);
        const struct { const char* lbl; ImVec4 col; float dx; } leg_items[3] = {
            {"Raw (dB)",  ui_obs::palette::Cyan,    ui_obs::dp(8.0f)},
            {"Baseline",  ui_obs::palette::Orange,  ui_obs::dp(82.0f)},
            {"Residuals", ui_obs::palette::Emerald, ui_obs::dp(162.0f)}
        };
        const float cy = lmin.y + lh * 0.5f;
        const float ty = lmin.y + (lh - ImGui::GetTextLineHeight()) * 0.5f;
        for (int k = 0; k < 3; ++k) {
            dl->AddLine(ImVec2(lmin.x + leg_items[k].dx, cy),
                        ImVec2(lmin.x + leg_items[k].dx + ui_obs::dp(10.0f), cy),
                        ui_obs::palette::U32(leg_items[k].col), ui_obs::dp(2.2f));
            dl->AddText(ImVec2(lmin.x + leg_items[k].dx + ui_obs::dp(14.0f), ty),
                        ui_obs::palette::TextPrimary32, leg_items[k].lbl);
        }
    }

    auto vp_bot = ui_obs::PlotViewport2D::create(bot_origin, ImVec2(avail.x, bot_h), x_min, x_max, y2_min, y2_max, m_bot);
    vp_bot.draw_background_and_grid(dl, 7, 4, "%.2f", "%+.2f", "Frequency (MHz)", "Delta (dB)", fonts_.mono);
    draw_passband_overlays(vp_bot, false);

    dl->PushClipRect(vp_bot.plot_min, vp_bot.plot_max, true);
    if (y2_min < 0.0 && y2_max > 0.0) {
        const float zy = vp_bot.map_y(0.0);
        dl->AddLine(ImVec2(vp_bot.plot_min.x, zy), ImVec2(vp_bot.plot_max.x, zy),
                    ui_obs::palette::U32(ui_obs::palette::TextDim, 0.80f), 1.0f);
    }
    if (has_base) {
        const double df = (f_max_full - f_min_full) / static_cast<double>(n - 1);
        auto res_at = [&](size_t i) {
            return has_res ? snap->residual_db[i] : (snap->raw_db[i] - snap->baseline_db[i]);
        };
        vp_bot.draw_area_fill_above_ref(dl, n,
                                        [&](size_t i) { return f_min_full + static_cast<double>(i) * df; },
                                        res_at,
                                        0.0, ui_obs::palette::Emerald, 0.02f, 0.24f, fill_scratch_);
        vp_bot.draw_decimated_trace(dl, n, f_min_full, f_max_full,
                                    res_at,
                                    ui_obs::palette::U32(ui_obs::palette::Emerald), ui_obs::dp(1.4f), trace_scratch_);
    }
    dl->PopClipRect();

    if (zoom.active) {
        const auto pill = ui_obs::widgets::compute_clamped_pill_rect(
            "ZOOMED | Dbl-click reset", fonts_.semibold, 10.5f,
            vp_bot.plot_max.x - ui_obs::dp(82.0f), vp_bot.plot_min.y + ui_obs::dp(6.0f),
            vp_bot.plot_min, vp_bot.plot_max);
        ui_obs::widgets::draw_pill_badge(dl, pill, "ZOOMED | Dbl-click reset",
                                         fonts_.semibold, 10.5f,
                                         ui_obs::palette::PillBg32,
                                         ui_obs::palette::U32(ui_obs::palette::Cyan, 0.50f),
                                         ui_obs::palette::U32(ui_obs::palette::Cyan));
    }

    ImGui::SetCursorScreenPos(origin);
    ImGui::InvisibleButton(hit_id, avail);
    ui_obs::handle_plot_zoom_pan(ImVec2(vp_top.plot_min.x, vp_top.plot_min.y),
                                 ImVec2(vp_bot.plot_max.x, vp_bot.plot_max.y),
                                 f_min_full, f_max_full, y1_min, y1_max, zoom, true, true);

    if (ImGui::IsItemHovered()) {
        const ImVec2 mp = ImGui::GetIO().MousePos;
        const bool in_top = vp_top.contains(mp);
        const bool in_bot = vp_bot.contains(mp);
        if (in_top || in_bot) {
            ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp_top.plot_min.y, vp_top.plot_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
            ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp_bot.plot_min.y, vp_bot.plot_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
            if (in_top) {
                ui_obs::widgets::draw_dashed_hline(dl, vp_top.plot_min.x, vp_top.plot_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);
            } else {
                ui_obs::widgets::draw_dashed_hline(dl, vp_bot.plot_min.x, vp_bot.plot_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);
            }

            const double freq_h = in_top ? vp_top.unmap_x(mp.x) : vp_bot.unmap_x(mp.x);
            const size_t idx = find_nearest_index(snap->stream.freq_mhz, freq_h);
            const double raw_v = snap->raw_db[idx];
            const double base_v = has_base ? snap->baseline_db[idx] : 0.0;
            char b_f[32], b_r[32], b_b[32], b_d[32];
            std::snprintf(b_f, sizeof(b_f), "%.4f MHz", snap->stream.freq_mhz[idx]);
            std::snprintf(b_r, sizeof(b_r), "%.2f dB",  raw_v);
            std::snprintf(b_b, sizeof(b_b), "%.2f dB",  base_v);
            std::snprintf(b_d, sizeof(b_d), "%+.3f dB", raw_v - base_v);

            const ui_obs::widgets::TelemetryRow rows[4] = {
                {"Freq",  b_f, ui_obs::palette::U32(ui_obs::palette::Cyan)},
                {"Raw",   b_r, ui_obs::palette::TextPrimary32},
                {"Base",  b_b, ui_obs::palette::U32(ui_obs::palette::Orange)},
                {"Delta", b_d, ui_obs::palette::U32(ui_obs::palette::Emerald)}
            };
            const ImVec2 b_min = in_top ? vp_top.plot_min : vp_bot.plot_min;
            const ImVec2 b_max = in_top ? vp_top.plot_max : vp_bot.plot_max;
            ui_obs::widgets::draw_crosshair_card(dl, mp, b_min, b_max,
                                                 "BANDPASS & BASELINE",
                                                 ui_obs::palette::U32(ui_obs::palette::Orange),
                                                 rows, has_base ? 4 : 2, fonts_.regular, fonts_.mono);
        }
    }
}

void UiAnalyzePanel::render_q2_calibrated_db_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                                  ImVec2 avail, const char* hit_id, ui_obs::PlotZoomState& zoom) {
    if (avail.x < ui_obs::dp(60.0f) || avail.y < ui_obs::dp(40.0f) ||
        snap->stream.freq_mhz.size() < 2 || snap->calibrated_db.size() != snap->stream.freq_mhz.size()) {
        return;
    }

    const size_t n = snap->stream.freq_mhz.size();
    const size_t edge_bins = std::min(n / 4, static_cast<size_t>(n * std::clamp(snap->config_used.passband_edge_fraction, 0.0, 0.45)));
    const double f0 = snap->stream.freq_mhz[edge_bins];
    const double f1 = snap->stream.freq_mhz[n - 1 - edge_bins];
    const bool has_hi = (HYDROGEN_REST_FREQ_MHZ >= f0 && HYDROGEN_REST_FREQ_MHZ <= f1);
    const double def_xmin = has_hi ? std::max(f0, HYDROGEN_REST_FREQ_MHZ - 0.6) : f0;
    const double def_xmax = has_hi ? std::min(f1, HYDROGEN_REST_FREQ_MHZ + 0.6) : f1;

    const double x_min = zoom.active ? zoom.x_min : def_xmin;
    const double x_max = zoom.active ? zoom.x_max : def_xmax;

    double def_ymin = 1e9, def_ymax = -1e9;
    for (size_t i = 0; i < n; ++i) {
        const double f = snap->stream.freq_mhz[i];
        if (f < x_min || f > x_max) continue;
        def_ymin = std::min(def_ymin, snap->calibrated_db[i]);
        def_ymax = std::max(def_ymax, snap->calibrated_db[i]);
    }
    if (def_ymin >= def_ymax) { def_ymin = -0.2; def_ymax = 0.5; }
    const double pad = std::max(0.08, (def_ymax - def_ymin) * 0.14);
    def_ymin = std::min(def_ymin - pad, -0.10);
    def_ymax = def_ymax + pad;

    const double ymin = zoom.active ? zoom.y_min : def_ymin;
    const double ymax = zoom.active ? zoom.y_max : def_ymax;

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    ui_obs::PlotMargin m{ui_obs::dp(54.0f), ui_obs::dp(16.0f), ui_obs::dp(22.0f), ui_obs::dp(28.0f)};
    auto vp = ui_obs::PlotViewport2D::create(origin, avail, x_min, x_max, ymin, ymax, m);
    vp.draw_background_and_grid(dl, 5, 4, "%.2f", "%.2f", "Frequency (MHz)", "dB", fonts_.mono);

    dl->PushClipRect(vp.plot_min, vp.plot_max, true);
    if (ymin < 0.0 && ymax > 0.0) {
        const float zy = vp.map_y(0.0);
        dl->AddLine(ImVec2(vp.plot_min.x, zy), ImVec2(vp.plot_max.x, zy),
                    ui_obs::palette::U32(ui_obs::palette::TextDim, 0.80f), 1.0f);
    }
    if (HYDROGEN_REST_FREQ_MHZ >= x_min && HYDROGEN_REST_FREQ_MHZ <= x_max) {
        const float hx = vp.map_x(HYDROGEN_REST_FREQ_MHZ);
        ui_obs::widgets::draw_dashed_vline(dl, hx, vp.plot_min.y, vp.plot_max.y,
                                           ui_obs::palette::U32(ui_obs::palette::Amber, 0.72f), 1.2f);
    }

    const double df = (f1 - f0) / static_cast<double>(n - 1);
    vp.draw_area_fill_above_ref(dl, n,
                                [&](size_t i) { return f0 + static_cast<double>(i) * df; },
                                [&](size_t i) { return snap->calibrated_db[i]; },
                                0.0, ui_obs::palette::Emerald, 0.02f, 0.30f, fill_scratch_);
    vp.draw_decimated_trace(dl, n, f0, f1,
                            [&](size_t i) { return snap->calibrated_db[i]; },
                            ui_obs::palette::U32(ui_obs::palette::Emerald), ui_obs::dp(1.6f), trace_scratch_);

    if (snap->metrics.peak_freq_mhz >= x_min && snap->metrics.peak_freq_mhz <= x_max &&
        snap->metrics.peak_power_db >= ymin && snap->metrics.peak_power_db <= ymax) {
        const float px = vp.map_x(snap->metrics.peak_freq_mhz);
        const float py = vp.map_y(snap->metrics.peak_power_db);
        dl->AddCircleFilled(ImVec2(px, py), ui_obs::dp(3.5f), ui_obs::palette::U32(ui_obs::palette::Red), 14);
        dl->AddCircle(ImVec2(px, py), ui_obs::dp(4.0f), IM_COL32(255, 255, 255, 220), 14, 1.0f);
    }
    dl->PopClipRect();

    ImGui::SetCursorScreenPos(origin);
    ImGui::InvisibleButton(hit_id, avail);
    ui_obs::handle_plot_zoom_pan(vp.plot_min, vp.plot_max, f0, f1, def_ymin, def_ymax, zoom, true, false);

    if (ImGui::IsItemHovered() && vp.contains(ImGui::GetIO().MousePos)) {
        const ImVec2 mp = ImGui::GetIO().MousePos;
        ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp.plot_min.y, vp.plot_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
        ui_obs::widgets::draw_dashed_hline(dl, vp.plot_min.x, vp.plot_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);
        const size_t idx = find_nearest_index(snap->stream.freq_mhz, vp.unmap_x(mp.x));
        const double f_bin = snap->stream.freq_mhz[idx];
        const double v_bin = SPEED_OF_LIGHT_KMS * (HYDROGEN_REST_FREQ_MHZ - f_bin) / HYDROGEN_REST_FREQ_MHZ;
        char b_f[32], b_v[32], b_d[32];
        std::snprintf(b_f, sizeof(b_f), "%.4f MHz", f_bin);
        std::snprintf(b_v, sizeof(b_v), "%+.1f km/s", v_bin);
        std::snprintf(b_d, sizeof(b_d), "%+.3f dB", snap->calibrated_db[idx]);
        const ui_obs::widgets::TelemetryRow rows[3] = {
            {"Freq",  b_f, ui_obs::palette::U32(ui_obs::palette::Cyan)},
            {"V_LSR", b_v, ui_obs::palette::TextPrimary32},
            {"Power", b_d, ui_obs::palette::U32(ui_obs::palette::Emerald)}
        };
        ui_obs::widgets::draw_crosshair_card(dl, mp, vp.plot_min, vp.plot_max,
                                             "H I LINE (dB)", ui_obs::palette::U32(ui_obs::palette::Emerald),
                                             rows, 3, fonts_.regular, fonts_.mono);
    }
}

void UiAnalyzePanel::render_waterfall_view(ID3D11DeviceContext* ctx, ImVec2 avail,
                                           const std::shared_ptr<const AnalysisSnapshot>& snap,
                                           const char* img_id,
                                           bool compact) {
    update_waterfall_texture(ctx, snap);
    if (!waterfall_srv_ || avail.x < ui_obs::dp(120.0f) || avail.y < ui_obs::dp(60.0f)) {
        ImGui::TextColored(ui_obs::palette::TextMuted, "No waterfall integration data available.");
        return;
    }

    const float axis_left = ui_obs::dp(54.0f);
    const float colorbar_right = ui_obs::dp(72.0f);
    const float top_pad = compact ? ui_obs::dp(22.0f) : ui_obs::dp(26.0f);
    const float bottom_axis = compact ? ui_obs::dp(28.0f) : ui_obs::dp(38.0f);

    const float img_w = std::max(ui_obs::dp(40.0f), avail.x - axis_left - colorbar_right);
    const float img_h = std::max(ui_obs::dp(40.0f), avail.y - top_pad - bottom_axis);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 img_min(std::floor(origin.x + axis_left), std::floor(origin.y + top_pad));
    const ImVec2 img_max(img_min.x + img_w, img_min.y + img_h);

    const double u0 = zoom_waterfall_.active ? zoom_waterfall_.x_min : 0.0;
    const double u1 = zoom_waterfall_.active ? zoom_waterfall_.x_max : 1.0;
    const double v0 = zoom_waterfall_.active ? zoom_waterfall_.y_min : 0.0;
    const double v1 = zoom_waterfall_.active ? zoom_waterfall_.y_max : 1.0;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage(reinterpret_cast<ImTextureID>(waterfall_srv_), img_min, img_max,
                 ImVec2(static_cast<float>(u0), static_cast<float>(v0)),
                 ImVec2(static_cast<float>(u1), static_cast<float>(v1)));
    dl->AddRect(img_min, img_max, ui_obs::palette::PlotBorder32, ui_obs::dp(2.0f), 1.0f, ImDrawFlags_None);

    const double f_full0 = snap->stream.freq_mhz.empty() ? 0.0 : snap->stream.freq_mhz.front();
    const double f_full1 = snap->stream.freq_mhz.empty() ? 1.0 : snap->stream.freq_mhz.back();
    const double f0 = f_full0 + u0 * (f_full1 - f_full0);
    const double f1 = f_full0 + u1 * (f_full1 - f_full0);
    const double t0 = v0 * snap->wf_duration_s;
    const double t1 = v1 * snap->wf_duration_s;

    // Clipped 1420.406 MHz rest marker + small "H I" pill
    if (HYDROGEN_REST_FREQ_MHZ >= f0 && HYDROGEN_REST_FREQ_MHZ <= f1 && (f1 > f0)) {
        const float hx = img_min.x + static_cast<float>((HYDROGEN_REST_FREQ_MHZ - f0) / (f1 - f0)) * img_w;
        dl->PushClipRect(img_min, img_max, true);
        ui_obs::widgets::draw_dashed_vline(dl, hx, img_min.y, img_max.y,
                                           ui_obs::palette::U32(ui_obs::palette::Amber, 0.62f), 1.2f);
        dl->PopClipRect();

        const ImVec2 pill_bounds_min(img_min.x, origin.y);
        const ImVec2 pill_bounds_max(img_max.x, img_min.y);
        const auto p = ui_obs::widgets::compute_clamped_pill_rect(
            "H I", fonts_.semibold, 10.5f, hx, img_min.y - ui_obs::dp(20.0f),
            pill_bounds_min, pill_bounds_max, 5.0f, 1.5f);
        ui_obs::widgets::draw_pill_badge(dl, p, "H I", fonts_.semibold, 10.5f,
                                         ui_obs::palette::PillBg32,
                                         ui_obs::palette::U32(ui_obs::palette::Amber, 0.55f),
                                         ui_obs::palette::U32(ui_obs::palette::Amber), 0, 5.0f, 1.5f);
    }

    // Axis ticks & titles
    {
        ui_obs::ScopedFont tf(fonts_.mono, 11.5f);
        const float tick_h = ImGui::GetFontSize();
        char buf[32];
        const int x_ticks = compact ? 5 : 6;
        float last_right = -1e9f;
        for (int i = 0; i < x_ticks; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(x_ticks - 1);
            const float px = img_min.x + static_cast<float>(frac) * img_w;
            dl->AddLine(ImVec2(px, img_max.y), ImVec2(px, img_max.y + ui_obs::dp(3.5f)),
                        ui_obs::palette::PlotBorder32, 1.0f);
            std::snprintf(buf, sizeof(buf), "%.2f", f0 + frac * (f1 - f0));
            const ImVec2 ts = ImGui::CalcTextSize(buf);
            const float lx = std::clamp(px - ts.x * 0.5f, img_min.x, img_max.x - ts.x);
            if (lx >= last_right + ui_obs::dp(6.0f)) {
                dl->AddText(ImVec2(lx, img_max.y + ui_obs::dp(3.5f)), ui_obs::palette::TextMuted32, buf);
                last_right = lx + ts.x;
            }
        }
        const int y_ticks = compact ? 4 : 5;
        for (int i = 0; i < y_ticks; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(y_ticks - 1);
            const float py = img_min.y + static_cast<float>(frac) * img_h;
            dl->AddLine(ImVec2(img_min.x - ui_obs::dp(3.5f), py), ImVec2(img_min.x, py),
                        ui_obs::palette::PlotBorder32, 1.0f);
            std::snprintf(buf, sizeof(buf), "%.0fs", t0 + frac * (t1 - t0));
            const ImVec2 ts = ImGui::CalcTextSize(buf);
            dl->AddText(ImVec2(img_min.x - ts.x - ui_obs::dp(6.0f), py - ts.y * 0.5f),
                        ui_obs::palette::TextMuted32, buf);
        }

        if (!compact) {
            ui_obs::ScopedFont lf(fonts_.regular, 12.0f);
            const ImVec2 xts = ImGui::CalcTextSize("Frequency (MHz)");
            dl->AddText(ImVec2(img_min.x + (img_w - xts.x) * 0.5f, img_max.y + tick_h + ui_obs::dp(4.5f)),
                        ui_obs::palette::U32(ui_obs::palette::TextDim), "Frequency (MHz)");
        }
    }
    {
        ui_obs::ScopedFont ylf(fonts_.regular, 11.5f);
        const ImVec2 yts = ImGui::CalcTextSize("Time (s)");
        dl->AddText(ImVec2(std::max(origin.x + ui_obs::dp(2.0f), img_min.x - ui_obs::dp(6.0f) - yts.x),
                           std::max(origin.y + ui_obs::dp(1.0f), img_min.y - yts.y - ui_obs::dp(3.0f))),
                    ui_obs::palette::TextMuted32, "Time (s)");
    }

    const uint32_t* lut_ptr = (active_lut256_.size() >= 256) ? active_lut256_.data() : nullptr;
    ui_obs::draw_vertical_colorbar(
        dl,
        ImVec2(img_max.x + ui_obs::dp(10.0f), img_min.y),
        ImVec2(img_max.x + ui_obs::dp(22.0f), img_max.y),
        snap->wf_vmin_db, snap->wf_vmax_db, "%.1f", lut_ptr, fonts_.mono, "Rel dB"
    );

    if (zoom_waterfall_.active) {
        const auto pill = ui_obs::widgets::compute_clamped_pill_rect(
            "ZOOMED | Dbl-click reset", fonts_.semibold, 10.5f,
            img_max.x - ui_obs::dp(82.0f), img_min.y + ui_obs::dp(8.0f),
            img_min, img_max);
        ui_obs::widgets::draw_pill_badge(dl, pill, "ZOOMED | Dbl-click reset",
                                         fonts_.semibold, 10.5f,
                                         ui_obs::palette::PillBg32,
                                         ui_obs::palette::U32(ui_obs::palette::Cyan, 0.50f),
                                         ui_obs::palette::U32(ui_obs::palette::Cyan));
    }

    // UV Zoom/Pan Interaction (texture V=0 is top, V=1 is bottom)
    ImGui::SetCursorScreenPos(img_min);
    ImGui::InvisibleButton(img_id, ImVec2(img_w, img_h));
    const ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsItemHovered()) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            zoom_waterfall_.active = false;
        } else if (io.MouseWheel != 0.0f &&
                   io.MousePos.x >= img_min.x && io.MousePos.x <= img_max.x &&
                   io.MousePos.y >= img_min.y && io.MousePos.y <= img_max.y) {
            if (!zoom_waterfall_.active) {
                zoom_waterfall_.x_min = 0.0; zoom_waterfall_.x_max = 1.0;
                zoom_waterfall_.y_min = 0.0; zoom_waterfall_.y_max = 1.0;
                zoom_waterfall_.active = true;
            }
            const double factor = std::pow(0.86, static_cast<double>(io.MouseWheel));
            const double fx = static_cast<double>(io.MousePos.x - img_min.x) / img_w;
            const double fy = static_cast<double>(io.MousePos.y - img_min.y) / img_h;
            const double ax = zoom_waterfall_.x_min + fx * (zoom_waterfall_.x_max - zoom_waterfall_.x_min);
            const double ay = zoom_waterfall_.y_min + fy * (zoom_waterfall_.y_max - zoom_waterfall_.y_min);

            if (!io.KeyCtrl) {
                const double span = std::clamp((zoom_waterfall_.x_max - zoom_waterfall_.x_min) * factor, 0.02, 1.0);
                zoom_waterfall_.x_min = ax - fx * span;
                zoom_waterfall_.x_max = zoom_waterfall_.x_min + span;
                if (zoom_waterfall_.x_min < 0.0) { zoom_waterfall_.x_max -= zoom_waterfall_.x_min; zoom_waterfall_.x_min = 0.0; }
                if (zoom_waterfall_.x_max > 1.0) { zoom_waterfall_.x_min -= (zoom_waterfall_.x_max - 1.0); zoom_waterfall_.x_max = 1.0; }
                zoom_waterfall_.x_min = std::clamp(zoom_waterfall_.x_min, 0.0, 1.0);
                zoom_waterfall_.x_max = std::clamp(zoom_waterfall_.x_max, 0.0, 1.0);
            }
            if (!io.KeyShift) {
                const double span = std::clamp((zoom_waterfall_.y_max - zoom_waterfall_.y_min) * factor, 0.02, 1.0);
                zoom_waterfall_.y_min = ay - fy * span;
                zoom_waterfall_.y_max = zoom_waterfall_.y_min + span;
                if (zoom_waterfall_.y_min < 0.0) { zoom_waterfall_.y_max -= zoom_waterfall_.y_min; zoom_waterfall_.y_min = 0.0; }
                if (zoom_waterfall_.y_max > 1.0) { zoom_waterfall_.y_min -= (zoom_waterfall_.y_max - 1.0); zoom_waterfall_.y_max = 1.0; }
                zoom_waterfall_.y_min = std::clamp(zoom_waterfall_.y_min, 0.0, 1.0);
                zoom_waterfall_.y_max = std::clamp(zoom_waterfall_.y_max, 0.0, 1.0);
            }
        }
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
        if (!zoom_waterfall_.active) {
            zoom_waterfall_.x_min = 0.0; zoom_waterfall_.x_max = 1.0;
            zoom_waterfall_.y_min = 0.0; zoom_waterfall_.y_max = 1.0;
            zoom_waterfall_.active = true;
        }
        const double ux_span = zoom_waterfall_.x_max - zoom_waterfall_.x_min;
        const double vy_span = zoom_waterfall_.y_max - zoom_waterfall_.y_min;
        zoom_waterfall_.x_min -= (static_cast<double>(io.MouseDelta.x) / img_w) * ux_span;
        zoom_waterfall_.x_max  = zoom_waterfall_.x_min + ux_span;
        if (zoom_waterfall_.x_min < 0.0) { zoom_waterfall_.x_min = 0.0; zoom_waterfall_.x_max = ux_span; }
        if (zoom_waterfall_.x_max > 1.0) { zoom_waterfall_.x_max = 1.0; zoom_waterfall_.x_min = 1.0 - ux_span; }

        zoom_waterfall_.y_min -= (static_cast<double>(io.MouseDelta.y) / img_h) * vy_span;
        zoom_waterfall_.y_max  = zoom_waterfall_.y_min + vy_span;
        if (zoom_waterfall_.y_min < 0.0) { zoom_waterfall_.y_min = 0.0; zoom_waterfall_.y_max = vy_span; }
        if (zoom_waterfall_.y_max > 1.0) { zoom_waterfall_.y_max = 1.0; zoom_waterfall_.y_min = 1.0 - vy_span; }
    }

    if (ImGui::IsItemHovered()) {
        const ImVec2 mp = io.MousePos;
        if (mp.x >= img_min.x && mp.x <= img_max.x && mp.y >= img_min.y && mp.y <= img_max.y) {
            ui_obs::widgets::draw_dashed_vline(dl, mp.x, img_min.y, img_max.y, ui_obs::palette::PlotCrosshair, 1.0f);
            ui_obs::widgets::draw_dashed_hline(dl, img_min.x, img_max.x, mp.y, ui_obs::palette::PlotCrosshair, 1.0f);

            const double u = std::clamp(u0 + ((mp.x - img_min.x) / img_w) * (u1 - u0), 0.0, 1.0);
            const double v = std::clamp(v0 + ((mp.y - img_min.y) / img_h) * (v1 - v0), 0.0, 1.0);
            const double freq = f_full0 + u * (f_full1 - f_full0);
            const double v_lsr = SPEED_OF_LIGHT_KMS * (HYDROGEN_REST_FREQ_MHZ - freq) / HYDROGEN_REST_FREQ_MHZ;
            const double t_s = v * snap->wf_duration_s;

            double rel_db = snap->wf_vmin_db;
            if (snap->wf_tex_w > 0 && snap->wf_tex_h > 0 &&
                snap->wf_norm_u8.size() >= static_cast<size_t>(snap->wf_tex_w) * static_cast<size_t>(snap->wf_tex_h)) {
                const int col = std::clamp(static_cast<int>(u * (snap->wf_tex_w - 1) + 0.5), 0, snap->wf_tex_w - 1);
                const int row = std::clamp(static_cast<int>(v * (snap->wf_tex_h - 1) + 0.5), 0, snap->wf_tex_h - 1);
                const uint8_t q = snap->wf_norm_u8[static_cast<size_t>(row) * static_cast<size_t>(snap->wf_tex_w) + static_cast<size_t>(col)];
                rel_db = snap->wf_vmin_db + (static_cast<double>(q) / 255.0) * (snap->wf_vmax_db - snap->wf_vmin_db);
            }

            char b_f[32], b_v[32], b_t[32], b_db[32];
            std::snprintf(b_f,  sizeof(b_f),  "%.4f MHz",  freq);
            std::snprintf(b_v,  sizeof(b_v),  "%+.1f km/s", v_lsr);
            std::snprintf(b_t,  sizeof(b_t),  "%.1f s",    t_s);
            std::snprintf(b_db, sizeof(b_db), "%+.1f dB",  rel_db);
            const ui_obs::widgets::TelemetryRow rows[4] = {
                {"Freq",  b_f,  ui_obs::palette::U32(ui_obs::palette::Cyan)},
                {"V_LSR", b_v,  ui_obs::palette::TextPrimary32},
                {"Time",  b_t,  ui_obs::palette::TextPrimary32},
                {"Power", b_db, ui_obs::palette::U32(ui_obs::palette::Amber)}
            };
            ui_obs::widgets::draw_crosshair_card(dl, mp, img_min, img_max,
                                                 "WATERFALL SAMPLE", ui_obs::palette::U32(ui_obs::palette::Amber),
                                                 rows, 4, fonts_.regular, fonts_.mono);
        }
    }
}

void UiAnalyzePanel::render_tab_waterfall(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap) {
    render_waterfall_view(ctx, ImGui::GetContentRegionAvail(), snap, "##TabWfImg", false);
}

void UiAnalyzePanel::render_tab_master_dashboard(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < ui_obs::dp(160.0f) || avail.y < ui_obs::dp(140.0f)) return;

    const ImVec2 sp = ImGui::GetStyle().ItemSpacing;
    const float cell_w = std::floor((avail.x - sp.x) * 0.5f);
    const float cell_h = std::floor((avail.y - sp.y) * 0.5f);

    auto begin_quad = [&](const char* child_id, const char* title, const QuadSwatch* sw, int sw_count) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::BgDeep);
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(8.0f), ui_obs::dp(6.0f)));
        ImGui::BeginChild(child_id, ImVec2(cell_w, cell_h), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        draw_quad_header(ImGui::GetWindowDrawList(), fonts_.semibold, fonts_.regular, title, sw, sw_count);
    };
    auto end_quad = [&]() {
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    };

    const QuadSwatch q1_sw[2] = {{"Raw", ui_obs::palette::Cyan}, {"Base", ui_obs::palette::Orange}};
    begin_quad("##Q1", "Q1: PASSBAND & BASELINE", q1_sw, 2);
    render_raw_baseline_plot(snap, ImGui::GetContentRegionAvail(), true, "##Q1RawHit", zoom_raw_base_);
    end_quad();

    ImGui::SameLine();

    const QuadSwatch q2_sw[2] = {{"+dB", ui_obs::palette::Emerald}, {"1420.4", ui_obs::palette::Amber}};
    begin_quad("##Q2", "Q2: CALIBRATED H I (dB)", q2_sw, 2);
    render_q2_calibrated_db_plot(snap, ImGui::GetContentRegionAvail(), "##Q2DbHit", zoom_q2_db_);
    end_quad();

    const QuadSwatch q3_sw[2] = {{"T_A", ui_obs::palette::Cyan}, {"Gauss", ui_obs::palette::Pink}};
    begin_quad("##Q3", "Q3: DOPPLER PROFILE (T_A)", q3_sw, 2);
    render_calibrated_spectrum_plot(snap, ImGui::GetContentRegionAvail(), true, "##Q3CalHit", zoom_cal_spec_);
    end_quad();

    ImGui::SameLine();

    const QuadSwatch q4_sw[1] = {{"5%-99.5%", ui_obs::palette::Amber}};
    begin_quad("##Q4", "Q4: WATERFALL HISTORY", q4_sw, 1);
    render_waterfall_view(ctx, ImGui::GetContentRegionAvail(), snap, "##Q4WfImg", true);
    end_quad();
}

void UiAnalyzePanel::update_waterfall_texture(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap) {
    if (!d3d_device_ || !ctx || !snap || snap->wf_tex_w <= 0 || snap->wf_tex_h <= 0 || snap->wf_norm_u8.empty()) {
        return;
    }

    const int w = snap->wf_tex_w;
    const int h = snap->wf_tex_h;

    // Fast early-out if GPU texture already matches snapshot and colormap!
    if (waterfall_texture_ && last_uploaded_snap_ == snap.get() && last_colormap_idx_ == ui_colormap_idx_ &&
        tex_width_ == w && tex_height_ == h) {
        return;
    }

    if (!waterfall_texture_ || tex_width_ != w || tex_height_ != h) {
        release_d3d_resources();
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = static_cast<UINT>(w);
        desc.Height = static_cast<UINT>(h);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        if (SUCCEEDED(d3d_device_->CreateTexture2D(&desc, nullptr, &waterfall_texture_)) && waterfall_texture_) {
            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc{};
            srv_desc.Format = desc.Format;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = 1;
            d3d_device_->CreateShaderResourceView(waterfall_texture_, &srv_desc, &waterfall_srv_);
            tex_width_ = w;
            tex_height_ = h;
        }
    }

    if (waterfall_texture_) {
        const uint32_t* lut = active_lut256_.data();
        const uint8_t* src = snap->wf_norm_u8.data();

        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(ctx->Map(waterfall_texture_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            uint8_t* dst = static_cast<uint8_t*>(mapped.pData);
            for (int y = 0; y < h; ++y) {
                const uint8_t* srow = src + static_cast<size_t>(y) * static_cast<size_t>(w);
                uint32_t* drow = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(y) * mapped.RowPitch);
                for (int x = 0; x < w; ++x) {
                    drow[x] = lut[srow[x]];
                }
            }
            ctx->Unmap(waterfall_texture_, 0);
        }

        last_uploaded_snap_ = snap.get();
        last_colormap_idx_ = ui_colormap_idx_;
    }
}

} // namespace astro
