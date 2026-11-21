#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <d3d11.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>

#include "app_state.h"
#include "astrophysics.h"
#include "config.h"
#include "constants.h"
#include "dc_excision.h"
#include "sidecar_reader.h"
#include "streaming_processor.h"
#include "ui_observatory.h"

namespace astro {

// Stage-A streaming FFT cache key (disk read + AVX2 Welch FFT accumulation)
struct StreamCacheKey {
    std::string filepath;
    uint64_t file_size_bytes = 0;
    double sample_rate_msps = 2.4;
    int fft_size_idx = 2;
    float duration_s = 0.0f;
    WindowType window = WindowType::Blackman;
    bool enable_welch_overlap = true;

    bool operator==(const StreamCacheKey& o) const {
        return filepath == o.filepath &&
               file_size_bytes == o.file_size_bytes &&
               std::abs(sample_rate_msps - o.sample_rate_msps) < 1e-6 &&
               fft_size_idx == o.fft_size_idx &&
               std::abs(duration_s - o.duration_s) < 1e-3f &&
               window == o.window &&
               enable_welch_overlap == o.enable_welch_overlap;
    }
    bool operator!=(const StreamCacheKey& o) const { return !(*this == o); }
};

// Immutable Stage-A cached FFT & pre-normalized waterfall data shared across live Stage-B re-fits
struct CachedStreamData {
    StreamCacheKey key;
    StreamingResults stream;
    double base_center_freq_hz = 1420.5e6;

    // Pre-normalized 8-bit waterfall (downsampled <= 2048x1024) + percentile bounds
    int wf_tex_w = 0;
    int wf_tex_h = 0;
    std::vector<uint8_t> wf_norm_u8;
    double wf_vmin_db = -10.0;
    double wf_vmax_db = 10.0;
    double wf_duration_s = 0.0;
};

// Stage-B live post-processing parameter signature (<5 ms re-fit without disk I/O)
struct StageBParamsSig {
    double center_freq_mhz = 1420.5;
    float dc_clear_khz = 5.0f;
    float rfi_threshold_mad = 4.5f;
    int baseline_idx = 0;
    float hi_protection_half_width_khz = 190.0f;
    float filter_smoothing_sigma_khz = 35.0f;
    float passband_edge_pct = 8.0f;
    bool auto_skirt_trim = true;
    float als_lambda = 200.0f;
    float als_p = 0.50f;
    int poly_degree = 12;
    int sw_continuum_order = 4;
    float line_free_velocity_kms = 40.0f;
    int line_method_idx = 0;
    float tsys_k = 100.0f;
    float spin_temp_k = static_cast<float>(DEFAULT_HI_SPIN_TEMP_K);

    bool operator==(const StageBParamsSig& o) const {
        return std::abs(center_freq_mhz - o.center_freq_mhz) < 1e-6 &&
               std::abs(dc_clear_khz - o.dc_clear_khz) < 1e-3f &&
               std::abs(rfi_threshold_mad - o.rfi_threshold_mad) < 1e-3f &&
               baseline_idx == o.baseline_idx &&
               std::abs(hi_protection_half_width_khz - o.hi_protection_half_width_khz) < 1e-3f &&
               std::abs(filter_smoothing_sigma_khz - o.filter_smoothing_sigma_khz) < 1e-3f &&
               std::abs(passband_edge_pct - o.passband_edge_pct) < 1e-3f &&
               auto_skirt_trim == o.auto_skirt_trim &&
               std::abs(als_lambda - o.als_lambda) < 1e-2f &&
               std::abs(als_p - o.als_p) < 1e-4f &&
               poly_degree == o.poly_degree &&
               sw_continuum_order == o.sw_continuum_order &&
               std::abs(line_free_velocity_kms - o.line_free_velocity_kms) < 1e-3f &&
               line_method_idx == o.line_method_idx &&
               std::abs(tsys_k - o.tsys_k) < 1e-3f &&
               std::abs(spin_temp_k - o.spin_temp_k) < 1e-3f;
    }
    bool operator!=(const StageBParamsSig& o) const { return !(*this == o); }
};

struct AnalysisSnapshot {
    std::string source_filepath;
    AstroConfig config_used;
    std::shared_ptr<const CachedStreamData> stream_data;

    // Shifted frequency axes matching config_used.center_freq_hz
    std::vector<double> freq_hz;
    std::vector<double> freq_mhz;

    std::vector<double> raw_db;
    std::vector<double> rfi_clean_power;
    std::vector<double> baseline_power;
    std::vector<double> baseline_db;
    std::vector<double> residual_db;
    std::vector<double> calibrated_k;
    std::vector<double> calibrated_db;
    std::vector<double> vel_kms;

    // Effective skirt mask & baseline diagnostics
    std::vector<uint8_t> effective_passband_mask;
    int effective_edge_left_bins = 0;
    int effective_edge_right_bins = 0;
    int dc_cleared_bins = 0;
    int rfi_mitigated_bins = 0;
    double base_off_line_rms_db = 0.0;
    double base_thermal_rms_db = 0.0;
    double base_quality_ratio = 1.0;
    double base_lag1_autocorr = 0.0;
    int standing_wave_components_used = 0;
    double standing_wave_period1_mhz = 0.0;
    double standing_wave_period2_mhz = 0.0;
    double post_process_ms = 0.0;

    AstrophysicalMetrics metrics;
};

class UiAnalyzePanel {
public:
    explicit UiAnalyzePanel(AppState& app_state);
    ~UiAnalyzePanel();

    void set_fonts(const ui_obs::UiFonts& fonts) { fonts_ = fonts; }
    void set_owner_hwnd(HWND hwnd) { owner_hwnd_ = hwnd; }

    void init_d3d_resources(ID3D11Device* device);
    void release_d3d_resources();

    void render_panel(ID3D11DeviceContext* ctx, float width, float height);

    void select_and_run_file(const std::string& path, bool auto_run = true);
    void scan_files();

    bool is_analyzing() const { return is_analyzing_.load(); }
    bool has_cached_fft() const { return has_cached_fft_.load(std::memory_order_relaxed); }
    double get_channel_resolution_khz() const { return cached_chan_res_khz_.load(std::memory_order_relaxed); }
    double get_last_refit_ms() const { return last_refit_ms_.load(std::memory_order_relaxed); }

private:
    static std::shared_ptr<AnalysisSnapshot> run_post_processing(
        const std::shared_ptr<const CachedStreamData>& cached,
        const AstroConfig& run_cfg);

    void on_file_selected(const std::string& path);
    void start_analysis();
    void cancel_analysis();
    void start_export(const std::shared_ptr<const AnalysisSnapshot>& snap);
    void schedule_live_refit(const std::shared_ptr<const CachedStreamData>& stream,
                             const AstroConfig& cfg,
                             const StageBParamsSig& sig);
    void refit_worker_loop();
    void reset_advanced_defaults();

    AstroConfig build_run_config() const;
    StreamCacheKey current_stream_key() const;
    StageBParamsSig current_stage_b_sig() const;
    bool validate_inputs(std::string& out_error) const;
    std::string get_output_dir_for_source(const std::string& src_path) const;

    std::shared_ptr<const AnalysisSnapshot> get_snapshot();
    std::shared_ptr<const CachedStreamData> get_cached_stream();
    std::string get_status_message();
    void set_status_message(const std::string& msg);
    std::string get_export_message();
    void set_export_message(const std::string& msg);

    void render_top_bar(float width, float height, const std::shared_ptr<const AnalysisSnapshot>& snap);
    void render_empty_state(float width, float height);
    void render_sidebar_controls(float width, float height, const std::shared_ptr<const AnalysisSnapshot>& snap);
    void render_kpi_strip(float width, const AstrophysicalMetrics& m, const AstroConfig& cfg);
    void render_waterfall_view(ID3D11DeviceContext* ctx, ImVec2 avail,
                               const std::shared_ptr<const AnalysisSnapshot>& snap,
                               const char* img_id,
                               ui_obs::PlotZoomState& zoom,
                               bool compact = false);

    void render_tab_calibrated_spectrum(const std::shared_ptr<const AnalysisSnapshot>& snap);
    void render_tab_raw_and_baseline(const std::shared_ptr<const AnalysisSnapshot>& snap);
    void render_tab_waterfall(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap);
    void render_tab_master_dashboard(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap);

    void render_calibrated_spectrum_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                         ImVec2 avail, bool compact,
                                         const char* hit_id, ui_obs::PlotZoomState& zoom);
    void render_raw_baseline_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                  ImVec2 avail, bool compact,
                                  const char* hit_id,
                                  ui_obs::PlotZoomState& zoom_top,
                                  ui_obs::PlotZoomState* zoom_bot = nullptr);
    void render_q2_calibrated_db_plot(const std::shared_ptr<const AnalysisSnapshot>& snap,
                                      ImVec2 avail, const char* hit_id, ui_obs::PlotZoomState& zoom);
    bool begin_sidebar_card(const char* id, const char* title, const ImVec4& accent,
                            const char* badge_text = nullptr,
                            const ImVec4& badge_col = ui_obs::palette::TextMuted);
    void end_sidebar_card();

    void update_waterfall_texture(ID3D11DeviceContext* ctx, const std::shared_ptr<const AnalysisSnapshot>& snap);

    AppState& app_state_;
    ui_obs::UiFonts fonts_;
    HWND owner_hwnd_ = nullptr;

    // Direct3D 11 Resources
    ID3D11Device* d3d_device_ = nullptr;
    ID3D11Texture2D* waterfall_texture_ = nullptr;
    ID3D11ShaderResourceView* waterfall_srv_ = nullptr;
    int tex_width_ = 0;
    int tex_height_ = 0;
    const CachedStreamData* last_uploaded_stream_ = nullptr;
    int last_colormap_idx_ = -1;

    // File Selection State
    std::vector<std::string> discovered_files_;
    int selected_file_idx_ = -1;
    char custom_file_path_[512] = "";
    char selected_filename_[256] = "";
    uint64_t selected_file_size_bytes_ = 0;
    ObservationMetadata current_sidecar_;

    // Configuration Controls
    AstroConfig config_;
    double ui_center_freq_mhz_ = 1420.5;
    double ui_sample_rate_msps_ = 2.4;
    int ui_fft_size_idx_ = 2;       // 8192
    int ui_baseline_idx_ = 0;       // RippleFilter
    int ui_line_method_idx_ = 0;    // GaussianLS
    float ui_tsys_k_ = 100.0f;
    float ui_spin_temp_k_ = static_cast<float>(DEFAULT_HI_SPIN_TEMP_K);
    float ui_duration_s_ = 0.0f;
    float ui_dc_clear_khz_ = 5.0f;
    float ui_rfi_mad_thresh_ = 4.5f;
    float ui_hi_protect_khz_ = 190.0f;
    float ui_smooth_sigma_khz_ = 35.0f;
    float ui_passband_edge_pct_ = 8.0f;
    bool ui_auto_skirt_trim_ = true;
    float ui_als_lambda_ = 200.0f;
    float ui_als_p_ = 0.50f;
    int ui_poly_deg_ = 12;
    int ui_sw_order_ = 4;
    float ui_line_free_vel_kms_ = 40.0f;
    int ui_colormap_idx_ = 0;       // Viridis
    bool ui_live_refit_enabled_ = true;
    bool ui_collapse_fit_card_ = false;
    bool ui_link_dashboard_x_ = true;
    int pending_tab_select_ = -1;

    // Asynchronous Stage-A Execution & Export Threads
    std::atomic<bool> is_analyzing_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> is_exporting_{false};
    std::atomic<double> progress_pct_{0.0};
    std::atomic<double> progress_speed_{0.0};
    std::atomic<double> progress_eta_{0.0};
    std::atomic<bool> has_cached_fft_{false};
    std::atomic<double> cached_chan_res_khz_{0.0};
    std::atomic<double> last_refit_ms_{0.0};

    std::mutex status_mutex_;
    std::string status_message_ = "Ready. Select a .cs16 recording to begin.";
    std::string export_status_msg_;

    std::unique_ptr<std::thread> worker_thread_;
    std::unique_ptr<std::thread> export_thread_;

    // Coalesced Asynchronous Stage-B Live Re-Fit Worker
    std::mutex refit_mutex_;
    std::condition_variable refit_cv_;
    bool refit_stop_ = false;
    uint64_t refit_gen_ = 0;
    std::shared_ptr<const CachedStreamData> pending_refit_stream_;
    AstroConfig pending_refit_cfg_;
    StageBParamsSig pending_refit_sig_;
    std::unique_ptr<std::thread> refit_thread_;

    mutable uint64_t val_cache_tick_ms_ = 0;
    mutable std::string val_cache_path_;
    mutable bool val_cache_file_ok_ = false;
    bool validate_inputs_cached(std::string& out_error) const;

    // Snapshot & Cached Stream State
    std::mutex snapshot_mutex_;
    std::shared_ptr<const CachedStreamData> cached_stream_;
    std::shared_ptr<const AnalysisSnapshot> active_snapshot_;
    StageBParamsSig last_run_stage_b_sig_;
    StageBParamsSig last_submitted_stage_b_sig_;

    // Interactive plot zoom/pan states (preserved across Stage-B live re-fits)
    ui_obs::PlotZoomState zoom_cal_spec_;
    ui_obs::PlotZoomState zoom_raw_top_;
    ui_obs::PlotZoomState zoom_raw_bot_;
    ui_obs::PlotZoomState zoom_waterfall_;
    ui_obs::PlotZoomState zoom_q1_raw_;
    ui_obs::PlotZoomState zoom_q2_db_;
    ui_obs::PlotZoomState zoom_q3_cal_;
    ui_obs::PlotZoomState zoom_q4_wf_;
    const CachedStreamData* last_zoom_stream_ = nullptr;
    double last_zoom_center_freq_hz_ = 0.0;

    // Reusable UI plot vertex & colormap scratch buffers
    std::vector<ImVec2> trace_scratch_;
    std::vector<ImVec2> fill_scratch_;
    std::vector<uint32_t> active_lut256_;
};

} // namespace astro
