#include "ui_record_panel.h"
#include "constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>

namespace {

constexpr float kAlignedLeftMarginBase  = 60.0f;
constexpr float kAlignedRightMarginBase = 76.0f;
constexpr float kSharedBarOffsetXBase   = 12.0f;
constexpr float kSharedBarWidthBase     = 14.0f;

static std::string build_timestamped_cs16_path(const std::string& output_dir,
                                               const std::string& prefix,
                                               uint64_t center_freq_hz) {
    std::string clean_prefix = prefix.empty() ? "capture" : prefix;
    for (char& ch : clean_prefix) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '*' ||
            ch == '?' || ch == '"' || ch == '<' || ch == '>' || ch == '|' || ch == ' ') {
            ch = '_';
        }
    }

    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
#ifdef _WIN32
    localtime_s(&tm_now, &tt);
#else
    tm_now = *std::localtime(&tt);
#endif
    char stamp[128];
    const double freq_mhz = static_cast<double>(center_freq_hz) / 1e6;
    std::snprintf(stamp, sizeof(stamp), "%s_%04d%02d%02d_%02d%02d%02d_%.3fMHz.cs16",
                  clean_prefix.c_str(),
                  tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
                  tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec, freq_mhz);

    std::filesystem::path dir(output_dir.empty() ? "recordings" : output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return (dir / stamp).string();
}

} // namespace

UiRecordPanel::UiRecordPanel(SdrController& controller, DspPipeline& dsp, Cs16Writer& writer,
                             Waterfall& waterfall, AppState& app_state)
    : sdr_(controller), dsp_(dsp), writer_(writer), waterfall_(waterfall), app_state_(app_state)
{
    spectrum_data_.reserve(16384);
    peak_hold_data_.reserve(16384);
    trace_scratch_.reserve(2048);
    peak_scratch_.reserve(2048);

    waterfall_.set_palette(WaterfallPalette::Viridis);
    refresh_devices();
    sync_device_selection();
}

UiRecordPanel::~UiRecordPanel() {
    release_d3d_resources();
}

void UiRecordPanel::init_d3d_resources(ID3D11Device* device) {
    d3d_device_ = device;
}

void UiRecordPanel::release_d3d_resources() {
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
    wf_uploaded_rev_ = 0;
}

void UiRecordPanel::refresh_devices() {
    device_entries_.clear();

    auto hw_names = RtlSdrDevice::scan_devices();
    for (size_t i = 0; i < hw_names.size(); ++i) {
        // RtlSdrDevice::scan_devices() returns a placeholder string when no hardware is present;
        // skip it so we don't show two Virtual SDR entries in the device combo.
        if (hw_names[i].find("Virtual SDR") != std::string::npos) {
            continue;
        }
        DeviceEntry entry;
        entry.label = hw_names[i];
        entry.is_virtual = false;
        entry.hw_index = static_cast<uint32_t>(i);
        device_entries_.push_back(entry);
    }

    DeviceEntry sim;
    sim.label = "Virtual 21cm H I Simulator (Internal)";
    sim.is_virtual = true;
    sim.hw_index = 0;
    device_entries_.push_back(sim);

    sync_device_selection();
}

void UiRecordPanel::sync_device_selection() {
    const bool sim_active = sdr_.is_simulated();
    selected_device_idx_ = static_cast<int>(device_entries_.size()) - 1;
    if (!sim_active) {
        for (size_t i = 0; i < device_entries_.size(); ++i) {
            if (!device_entries_[i].is_virtual) {
                selected_device_idx_ = static_cast<int>(i);
                break;
            }
        }
    }

    valid_gains_ = sdr_.get_valid_gains();
    if (valid_gains_.empty()) {
        valid_gains_ = {0.0f, 9.0f, 14.0f, 25.4f, 36.4f, 49.6f};
    }

    lna_gain_db_ = snap_gain_to_valid(sdr_.get_current_gain());
    agc_enabled_ = sdr_.is_auto_gain();
    bias_tee_enabled_ = sdr_.is_bias_tee_enabled();
    ppm_correction_ = sdr_.get_current_ppm();
    device_status_text_ = sdr_.get_status_detail();

    const uint64_t f_hz = sdr_.get_current_frequency();
    std::snprintf(tune_input_buf_, sizeof(tune_input_buf_), "%.6f", static_cast<double>(f_hz) / 1e6);
    selected_preset_idx_ = -1;
    const auto& presets = SdrConstants::GetPresets();
    for (size_t i = 0; i < presets.size(); ++i) {
        if (presets[i].freq_hz == f_hz) {
            selected_preset_idx_ = static_cast<int>(i);
            break;
        }
    }

    const uint32_t rate_hz = sdr_.get_current_sample_rate();
    const auto& rates = SdrConstants::GetSampleRates();
    for (size_t i = 0; i < rates.size(); ++i) {
        if (rates[i].rate_hz == rate_hz) {
            selected_sample_rate_idx_ = static_cast<int>(i);
            break;
        }
    }

    const int cur_fft = dsp_.get_fft_size();
    const auto& ffts = SdrConstants::GetFftSizes();
    for (size_t i = 0; i < ffts.size(); ++i) {
        if (ffts[i] == cur_fft) {
            selected_fft_size_idx_ = static_cast<int>(i);
            break;
        }
    }
    avg_factor_ = dsp_.get_averaging();
}

float UiRecordPanel::snap_gain_to_valid(float requested_db) const {
    if (valid_gains_.empty()) return requested_db;
    float best = valid_gains_[0];
    float best_diff = std::abs(best - requested_db);
    for (float g : valid_gains_) {
        const float d = std::abs(g - requested_db);
        if (d < best_diff) {
            best_diff = d;
            best = g;
        }
    }
    return best;
}

void UiRecordPanel::set_frequency(uint64_t freq_hz) {
    const uint64_t clamped = std::clamp<uint64_t>(freq_hz, 24000000ULL, 1766000000ULL);
    sdr_.request_frequency(clamped);
    std::snprintf(tune_input_buf_, sizeof(tune_input_buf_), "%.6f", static_cast<double>(clamped) / 1e6);

    selected_preset_idx_ = -1;
    const auto& presets = SdrConstants::GetPresets();
    for (size_t i = 0; i < presets.size(); ++i) {
        if (presets[i].freq_hz == clamped) {
            selected_preset_idx_ = static_cast<int>(i);
            break;
        }
    }
}

void UiRecordPanel::apply_tuned_frequency_hz(double new_freq_hz) {
    const double clamped = std::clamp(new_freq_hz, 24.0e6, 1766.0e6);
    set_frequency(static_cast<uint64_t>(std::llround(clamped)));
}

void UiRecordPanel::auto_fit_db_range() {
    const float nf = dsp_.get_noise_floor_db();
    if (std::isfinite(nf) && nf > -150.0f && nf < 10.0f) {
        fft_min_db_ = std::clamp(nf - 8.0f, -140.0f, -20.0f);
        fft_max_db_ = std::clamp(std::max(peak_dbfs_ + 8.0f, fft_min_db_ + 32.0f), fft_min_db_ + 10.0f, 5.0f);
        spec_interact_zoom_.y_manual = false;
        spec_interact_zoom_.sync_active();
        waterfall_.recolor(fft_min_db_, fft_max_db_);
    }
}

void UiRecordPanel::step_frequency(int64_t delta_hz) {
    const int64_t cur = static_cast<int64_t>(sdr_.get_current_frequency());
    const int64_t next = std::max<int64_t>(24000000LL, cur + delta_hz);
    set_frequency(static_cast<uint64_t>(next));
}

void UiRecordPanel::update_peak_hold(double center_hz, double rate_hz) {
    if (spectrum_data_.empty()) return;

    const bool invalidated =
        peak_hold_data_.size() != spectrum_data_.size() ||
        peak_hold_bins_ != spectrum_data_.size() ||
        std::abs(center_hz - peak_hold_freq_hz_) > 1.0 ||
        std::abs(rate_hz - peak_hold_rate_hz_) > 1.0;

    if (invalidated) {
        peak_hold_data_.assign(spectrum_data_.begin(), spectrum_data_.end());
        peak_hold_freq_hz_ = center_hz;
        peak_hold_rate_hz_ = rate_hz;
        peak_hold_bins_ = spectrum_data_.size();
        peak_hold_max_db_ = *std::max_element(peak_hold_data_.begin(), peak_hold_data_.end());
        return;
    }

    if (show_peak_hold_) {
        const size_t n = spectrum_data_.size();
        float max_v = peak_hold_max_db_;
        for (size_t i = 0; i < n; ++i) {
            if (spectrum_data_[i] > peak_hold_data_[i]) {
                peak_hold_data_[i] = spectrum_data_[i];
                if (spectrum_data_[i] > max_v) {
                    max_v = spectrum_data_[i];
                }
            }
        }
        peak_hold_max_db_ = max_v;
    }
}

void UiRecordPanel::sync_recording_state() {
    if (!writer_.is_recording() && app_state_.is_recording_active()) {
        std::string last = writer_.get_last_recorded_filepath();
        app_state_.set_recording_active(false);
        if (!last.empty()) {
            app_state_.set_last_recorded_file(last);
        }
    }
}

void UiRecordPanel::render_panel(ID3D11DeviceContext* context, float width, float height) {
    sync_recording_state();

    // Live SDR keyboard shortcuts (guarded against text input, inline LO edit, and active popups)
    const ImGuiIO& io = ImGui::GetIO();
    const bool any_popup = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup);
    if (!io.WantTextInput && !editing_freq_inline_ && !any_popup && !io.KeyCtrl && !io.KeyAlt) {
        if (ImGui::IsKeyPressed(ImGuiKey_Space) && !writer_.is_recording()) {
            if (sdr_.is_streaming()) sdr_.stop_streaming();
            else sdr_.start_streaming();
        } else if (ImGui::IsKeyPressed(ImGuiKey_P)) {
            show_peak_hold_ = !show_peak_hold_;
        } else if (ImGui::IsKeyPressed(ImGuiKey_W)) {
            show_waterfall_ = !show_waterfall_;
        } else if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket)) {
            step_frequency(io.KeyShift ? -10000LL : -100000LL);
        } else if (ImGui::IsKeyPressed(ImGuiKey_RightBracket)) {
            step_frequency(io.KeyShift ? +10000LL : +100000LL);
        }
    }

    const float pad_x = ui_obs::dp(10.0f);
    const float pad_y = ui_obs::dp(8.0f);
    const float gap = ui_obs::dp(8.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(pad_x, pad_y));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
    ImGui::BeginChild("##RecPanelRoot", ImVec2(width, height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const float inner_w = std::max(100.0f, width - pad_x * 2.0f);
    const float ribbon_h = ui_obs::dp(48.0f);
    const float content_h = std::max(120.0f, height - ribbon_h - pad_y * 2.0f - gap);

    render_top_ribbon(inner_w, ribbon_h);

    const float sidebar_w = std::clamp(inner_w * 0.24f, ui_obs::dp(260.0f), ui_obs::dp(320.0f));
    const float main_w = std::max(240.0f, inner_w - sidebar_w - gap);

    render_sidebar(sidebar_w, content_h);
    ImGui::SameLine();
    render_display_area(context, main_w, content_h);

    render_bias_tee_modal();

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
}

void UiRecordPanel::render_segmented_lo_tuner(float box_h) {
    static constexpr double kStepWeightsHz[10] = {
        1e9, 1e8, 1e7, 1e6,
        1e5, 1e4, 1e3,
        1e2, 1e1, 1e0
    };

    double lo_hz = static_cast<double>(sdr_.get_current_frequency());
    const long long hz_int = std::llround(lo_hz);

    if (editing_freq_inline_) {
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ui_obs::dp(5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(6.0f)));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ui_obs::palette::BezelBg32);
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::U32(ui_obs::palette::Cyan));
        ui_obs::ScopedFont edit_f(fonts_.mono, 18.0f);
        ImGui::SetNextItemWidth(ui_obs::dp(218.0f));
        if (focus_freq_inline_) {
            ImGui::SetKeyboardFocusHere();
            focus_freq_inline_ = false;
        }
        const bool committed = ImGui::InputText(
            "##InlineFreqMHz", tune_input_buf_, sizeof(tune_input_buf_),
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_AutoSelectAll
        );
        if (committed) {
            const double mhz = std::strtod(tune_input_buf_, nullptr);
            if (mhz >= 24.0 && mhz <= 1766.0) {
                apply_tuned_frequency_hz(mhz * 1.0e6);
            }
            editing_freq_inline_ = false;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                   (!ImGui::IsItemActive() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
            editing_freq_inline_ = false;
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
        return;
    }

    int digits[10]{};
    long long rem = std::clamp<long long>(hz_int, 0, 9999999999LL);
    for (int i = 9; i >= 0; --i) {
        digits[i] = static_cast<int>(rem % 10LL);
        rem /= 10LL;
    }

    int first_nonzero = 9;
    for (int i = 0; i < 10; ++i) {
        if (digits[i] != 0) {
            first_nonzero = i;
            break;
        }
    }

    const float digit_w = ui_obs::dp(13.0f);
    const float dot_w   = ui_obs::dp(8.0f);
    const float unit_w  = ui_obs::dp(34.0f);
    const float pad_x   = ui_obs::dp(10.0f);
    const float total_w = pad_x * 2.0f + 10.0f * digit_w + 2.0f * dot_w + unit_w;

    const ImVec2 bmin = ImGui::GetCursorScreenPos();
    const ImVec2 bmax(bmin.x + total_w, bmin.y + box_h);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    const bool bezel_hovered = ImGui::IsMouseHoveringRect(bmin, bmax, true);
    dl->AddRectFilled(bmin, bmax, ui_obs::palette::BezelBg32, ui_obs::dp(6.0f), ImDrawFlags_RoundAll);
    dl->AddRect(bmin, bmax,
                bezel_hovered ? ui_obs::palette::WithAlphaF(ui_obs::palette::U32(ui_obs::palette::Cyan), 0.72f)
                              : ui_obs::palette::BezelBorder32,
                ui_obs::dp(6.0f), 1.0f, ImDrawFlags_RoundAll);

    ImFont* mf = fonts_.mono ? fonts_.mono : ImGui::GetFont();
    const float mfs = ui_obs::font_dp(20.0f);
    float cx = bmin.x + pad_x;
    const float cy = std::floor(bmin.y + (box_h - mfs) * 0.5f);

    for (int d = 0; d < 10; ++d) {
        if (d == 4 || d == 7) {
            dl->AddText(mf, mfs, ImVec2(std::floor(cx + 1.0f), cy), ui_obs::palette::TextMuted32, ".");
            cx += dot_w;
        }

        const ImVec2 dmin(cx, bmin.y + ui_obs::dp(3.0f));
        const ImVec2 dmax(cx + digit_w, bmax.y - ui_obs::dp(3.0f));
        ImGui::SetCursorScreenPos(dmin);
        ImGui::PushID(d);
        ImGui::InvisibleButton("##LO_Dig", ImVec2(digit_w, box_h - ui_obs::dp(6.0f)));
        const bool d_hov = ImGui::IsItemHovered();

        if (d_hov) {
            ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            dl->AddRectFilled(dmin, dmax, ui_obs::palette::WithAlphaF(ui_obs::palette::U32(ui_obs::palette::Cyan), 0.16f),
                              ui_obs::dp(3.0f), ImDrawFlags_RoundAll);
            dl->AddRect(dmin, dmax, ui_obs::palette::WithAlphaF(ui_obs::palette::U32(ui_obs::palette::Cyan), 0.48f),
                        ui_obs::dp(3.0f), 1.0f, ImDrawFlags_RoundAll);

            const ImVec2 mp = ImGui::GetMousePos();
            const float wheel = ImGui::GetIO().MouseWheel;
            double delta = 0.0;
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (ImGui::GetTime() - last_digit_click_time_ < 0.45 && pre_click_lo_hz_ > 0.0) {
                    lo_hz = pre_click_lo_hz_;
                    apply_tuned_frequency_hz(lo_hz);
                }
                std::snprintf(tune_input_buf_, sizeof(tune_input_buf_), "%.6f", lo_hz * 1e-6);
                editing_freq_inline_ = true;
                focus_freq_inline_ = true;
            } else if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                pre_click_lo_hz_ = lo_hz;
                last_digit_click_time_ = ImGui::GetTime();
                delta = (mp.y < (dmin.y + dmax.y) * 0.5f) ? kStepWeightsHz[d] : -kStepWeightsHz[d];
            } else if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                delta = -kStepWeightsHz[d];
            } else if (wheel != 0.0f) {
                delta = (wheel > 0.0f ? 1.0 : -1.0) * kStepWeightsHz[d];
            }

            if (delta != 0.0 && !editing_freq_inline_) {
                apply_tuned_frequency_hz(lo_hz + delta);
            }

            char tip[80];
            if (kStepWeightsHz[d] >= 1e6) {
                std::snprintf(tip, sizeof(tip), "+/- %.0f MHz (Wheel / L-click +/- | Dbl-click edit)", kStepWeightsHz[d] * 1e-6);
            } else if (kStepWeightsHz[d] >= 1e3) {
                std::snprintf(tip, sizeof(tip), "+/- %.0f kHz (Wheel / L-click +/- | Dbl-click edit)", kStepWeightsHz[d] * 1e-3);
            } else {
                std::snprintf(tip, sizeof(tip), "+/- %.0f Hz (Wheel / L-click +/- | Dbl-click edit)", kStepWeightsHz[d]);
            }
            ImGui::SetTooltip("%s", tip);
        }

        const char ch[2] = { static_cast<char>('0' + digits[d]), '\0' };
        const bool is_lead_zero = (d < first_nonzero && d < 3);
        const ImU32 dcol = d_hov ? ui_obs::palette::TextPrimary32
                                 : (is_lead_zero ? ui_obs::palette::WithAlphaF(ui_obs::palette::TextMuted32, 0.35f)
                                                 : (d >= 7 ? ui_obs::palette::WithAlphaF(ui_obs::palette::TextMuted32, 0.90f)
                                                           : ui_obs::palette::U32(ui_obs::palette::Cyan)));
        dl->AddText(mf, mfs, ImVec2(std::floor(cx + 1.0f), cy), dcol, ch);
        ImGui::PopID();
        cx += digit_w;
    }

    const float ufs = ui_obs::font_dp(13.0f);
    ImGui::SetCursorScreenPos(ImVec2(cx, bmin.y + ui_obs::dp(3.0f)));
    if (ImGui::InvisibleButton("##LO_Unit", ImVec2(unit_w, box_h - ui_obs::dp(6.0f)))) {
        std::snprintf(tune_input_buf_, sizeof(tune_input_buf_), "%.6f", lo_hz * 1e-6);
        editing_freq_inline_ = true;
        focus_freq_inline_ = true;
    }
    const bool unit_hov = ImGui::IsItemHovered();
    if (unit_hov) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("Click to type LO frequency in MHz");
    }
    dl->AddText(ImGui::GetFont(), ufs,
                ImVec2(std::floor(cx + ui_obs::dp(4.0f)), std::floor(bmin.y + (box_h - ufs) * 0.5f + ui_obs::dp(1.5f))),
                unit_hov ? ui_obs::palette::U32(ui_obs::palette::Cyan) : ui_obs::palette::TextMuted32, "MHz");

    ImGui::SetCursorScreenPos(bmin);
    ImGui::Dummy(ImVec2(total_w, box_h));
}

void UiRecordPanel::render_top_ribbon(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(10.0f), ui_obs::dp(6.0f)));
    ImGui::BeginChild("##RecTopRibbon", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wpos = ImGui::GetWindowPos();

    const bool streaming = sdr_.is_streaming();
    const float control_h = std::max(ui_obs::dp(28.0f), height - ui_obs::dp(12.0f));
    const float row_top = ImGui::GetCursorScreenPos().y;

    // 1. Start / Stop RX Stream Toggle (disabled while CS16 recording is active)
    const bool rec_active = writer_.is_recording();
    const ImVec4 rx_col = streaming ? ui_obs::palette::Red : ui_obs::palette::Emerald;
    ImGui::BeginDisabled(streaming && rec_active);
    ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(rx_col, 0.18f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(rx_col, 0.30f));
    ImGui::PushStyleColor(ImGuiCol_Border, rx_col);
    {
        ui_obs::ScopedFont btn_f(fonts_.semibold, 14.0f);
        const char* rx_label = streaming ? "    Stop RX" : "    Start RX";
        if (ImGui::Button(rx_label, ImVec2(ui_obs::dp(116.0f), control_h))) {
            if (streaming) sdr_.stop_streaming();
            else sdr_.start_streaming();
        }
        const ImVec2 bmin = ImGui::GetItemRectMin();
        const ImVec2 bmax = ImGui::GetItemRectMax();
        const ImVec2 ic(bmin.x + ui_obs::dp(16.0f), (bmin.y + bmax.y) * 0.5f);
        if (streaming) {
            ui_obs::icons::draw_stop_square(dl, ic, ui_obs::dp(6.0f), ui_obs::palette::U32(ui_obs::palette::Red));
        } else {
            ui_obs::icons::draw_play_triangle(dl, ic, ui_obs::dp(6.5f), ui_obs::palette::U32(ui_obs::palette::Emerald));
        }
    }
    ImGui::PopStyleColor(3);
    ImGui::EndDisabled();

    // 2. Segmented Clickable/Scrollable 10-Digit LO Display
    ImGui::SameLine(0.0f, ui_obs::dp(12.0f));
    render_segmented_lo_tuner(control_h);

    // 3. Step Pills
    ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ui_obs::dp(4.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(3.0f), 0.0f));
    {
        ui_obs::ScopedFont step_f(fonts_.mono, 12.5f);
        auto step_btn = [&](const char* lbl, int64_t delta) {
            if (ImGui::Button(lbl, ImVec2(ui_obs::dp(48.0f), control_h))) {
                step_frequency(delta);
            }
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::SameLine();
        };
        if (width > ui_obs::dp(1080.0f)) {
            step_btn("-1M", -1000000LL);
            step_btn("-100k", -100000LL);
            step_btn("-10k", -10000LL);
            ImGui::Dummy(ImVec2(ui_obs::dp(4.0f), 0.0f));
            ImGui::SameLine();
            step_btn("+10k", 10000LL);
            step_btn("+100k", 100000LL);
            step_btn("+1M", 1000000LL);
        } else if (width > ui_obs::dp(880.0f)) {
            step_btn("-100k", -100000LL);
            step_btn("-10k", -10000LL);
            step_btn("+10k", 10000LL);
            step_btn("+100k", 100000LL);
        } else {
            step_btn("-100k", -100000LL);
            step_btn("+100k", 100000LL);
        }
    }
    ImGui::PopStyleVar(2);

    // 4. Band Preset Selector + Interactive Bias-Tee Indicator Pill
    const float bias_pill_w = ui_obs::dp(104.0f);
    ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
    const auto& presets = SdrConstants::GetPresets();
    const char* cur_preset_name = (selected_preset_idx_ >= 0 && selected_preset_idx_ < static_cast<int>(presets.size()))
        ? presets[static_cast<size_t>(selected_preset_idx_)].name
        : "Custom LO";
    const float avail_preset_w = ImGui::GetContentRegionAvail().x - bias_pill_w - ui_obs::dp(12.0f);
    ImGui::SetNextItemWidth(std::clamp(avail_preset_w, ui_obs::dp(80.0f), ui_obs::dp(220.0f)));
    if (ImGui::BeginCombo("##BandPresets", cur_preset_name)) {
        for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
            const bool is_sel = (selected_preset_idx_ == i);
            if (ImGui::Selectable(presets[static_cast<size_t>(i)].name, is_sel)) {
                set_frequency(presets[static_cast<size_t>(i)].freq_hz);
            }
            if (is_sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // Right-aligned live Bias-Tee pill
    const float right_x = wpos.x + width - bias_pill_w - ui_obs::dp(12.0f);
    if (right_x > ImGui::GetItemRectMax().x + ui_obs::dp(8.0f)) {
        ImGui::SetCursorScreenPos(ImVec2(right_x, row_top));
    } else {
        ImGui::SameLine(0.0f, ui_obs::dp(8.0f));
    }
    const ImVec2 bp_min = ImGui::GetCursorScreenPos();
    const ImVec2 bp_max(bp_min.x + bias_pill_w, bp_min.y + control_h);
    ImGui::InvisibleButton("##TopBiasPill", ImVec2(bias_pill_w, control_h));
    const bool bp_hov = ImGui::IsItemHovered();
    if (bp_hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (bias_tee_enabled_) {
            bias_tee_enabled_ = false;
            sdr_.request_bias_tee(false);
        } else {
            open_bias_modal_ = true;
        }
    }
    const char* bias_txt = bias_tee_enabled_ ? "BIAS 4.5V" : "BIAS OFF";
    const ImU32 bp_bg = bias_tee_enabled_
        ? ui_obs::palette::WithAlphaF(ui_obs::palette::U32(ui_obs::palette::Amber), bp_hov ? 0.24f : 0.16f)
        : (bp_hov ? IM_COL32(28, 36, 48, 255) : ui_obs::palette::PillBg32);
    const ImU32 bp_bd = bias_tee_enabled_ ? ui_obs::palette::U32(ui_obs::palette::Amber) : ui_obs::palette::PlotBorder32;
    const ImU32 bp_tx = bias_tee_enabled_ ? ui_obs::palette::U32(ui_obs::palette::Amber) : ui_obs::palette::TextMuted32;
    const ImU32 bp_dot = bias_tee_enabled_ ? ui_obs::palette::U32(ui_obs::palette::Amber) : IM_COL32(82, 92, 108, 255);
    dl->AddRectFilled(bp_min, bp_max, bp_bg, ui_obs::dp(4.0f), ImDrawFlags_RoundAll);
    dl->AddRect(bp_min, bp_max, bp_bd, ui_obs::dp(4.0f), 1.0f, ImDrawFlags_RoundAll);
    ui_obs::icons::draw_bolt(dl, ImVec2(bp_min.x + ui_obs::dp(15.0f), (bp_min.y + bp_max.y) * 0.5f),
                             ui_obs::dp(6.0f), bp_dot);
    {
        ui_obs::ScopedFont bf(fonts_.semibold, 12.0f);
        const ImVec2 bts = ImGui::CalcTextSize(bias_txt);
        dl->AddText(ImVec2(bp_min.x + ui_obs::dp(25.0f), bp_min.y + (control_h - bts.y) * 0.5f - 0.5f),
                    bp_tx, bias_txt);
    }
    if (bp_hov) {
        ImGui::SetTooltip(bias_tee_enabled_
            ? "4.5V Bias-Tee ACTIVE (Click to power OFF)"
            : "Bias-Tee OFF (Click to enable with confirmation)");
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

void UiRecordPanel::render_sidebar(float width, float height) {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ui_obs::dp(6.0f), ui_obs::dp(5.0f)));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(8.0f), ui_obs::dp(6.0f)));
    ImGui::BeginChild("##RecSidebar", ImVec2(width, height), false);
    render_device_card();
    render_radio_card();
    render_dsp_card();
    render_recording_card();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
}

void UiRecordPanel::render_device_card() {
    const bool sim = sdr_.is_simulated();
    if (ui_obs::widgets::BeginCard(
            "##CardSource", fonts_.semibold, "SOURCE HARDWARE",
            ui_obs::palette::Cyan,
            sim ? "Simulated" : "RTL-SDR",
            sim ? ui_obs::palette::Amber : ui_obs::palette::Emerald)) {
        const char* cur_label = (!device_entries_.empty() && selected_device_idx_ >= 0 &&
                                 selected_device_idx_ < static_cast<int>(device_entries_.size()))
            ? device_entries_[static_cast<size_t>(selected_device_idx_)].label.c_str()
            : "No Source";

        const float scan_w = ui_obs::dp(52.0f);
        const bool is_rec_busy = writer_.is_recording();
        ImGui::BeginDisabled(is_rec_busy);
        ImGui::SetNextItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x - scan_w - ui_obs::dp(6.0f)));
        if (ImGui::BeginCombo("##SdrDeviceCombo", cur_label)) {
            for (int i = 0; i < static_cast<int>(device_entries_.size()); ++i) {
                ImGui::PushID(i);
                const bool is_sel = (selected_device_idx_ == i);
                if (ImGui::Selectable(device_entries_[static_cast<size_t>(i)].label.c_str(), is_sel)) {
                    if (selected_device_idx_ != i) {
                        const auto& entry = device_entries_[static_cast<size_t>(i)];
                        const bool was_streaming = sdr_.is_streaming();
                        sdr_.stop_streaming();
                        sdr_.initialize(entry.is_virtual, entry.hw_index);
                        if (was_streaming) sdr_.start_streaming();
                        sync_device_selection();
                    }
                }
                if (is_sel) ImGui::SetItemDefaultFocus();
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        if (is_rec_busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Stop active CS16 recording before switching SDR source.");
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(is_rec_busy);
        if (ImGui::Button("Scan", ImVec2(scan_w, 0.0f))) {
            refresh_devices();
        }
        ImGui::EndDisabled();

        const ImVec4 st_col = sim ? ui_obs::palette::TextMuted : ui_obs::palette::Emerald;
        ImGui::TextColored(st_col, "%s", device_status_text_.c_str());

        const auto now = std::chrono::steady_clock::now();
        if (last_blocker_check_.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_blocker_check_).count() >= 1500) {
            cached_blockers_ = SdrController::check_blocking_apps();
            last_blocker_check_ = now;
        }
        if (!cached_blockers_.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ui_obs::palette::Red);
            ImGui::TextWrapped("USB busy: %s", cached_blockers_[0].c_str());
            ImGui::PopStyleColor();
        }
    }
    ui_obs::widgets::EndCard();
}

void UiRecordPanel::render_radio_card() {
    if (ui_obs::widgets::BeginCard(
            "##CardRadio", fonts_.semibold, "RADIO & GAIN",
            ui_obs::palette::Cyan,
            agc_enabled_ ? "AGC" : "Manual",
            agc_enabled_ ? ui_obs::palette::Emerald : ui_obs::palette::TextMuted)) {
        const auto& rates = SdrConstants::GetSampleRates();
        const char* cur_rate = (selected_sample_rate_idx_ >= 0 && selected_sample_rate_idx_ < static_cast<int>(rates.size()))
            ? rates[static_cast<size_t>(selected_sample_rate_idx_)].label
            : "2.4 Msps";
        const float half_w = std::max(70.0f, (ImGui::GetContentRegionAvail().x - ui_obs::dp(6.0f)) * 0.5f);

        ImGui::TextColored(ui_obs::palette::TextMuted, "Sample Rate");
        ImGui::SameLine(half_w + ui_obs::dp(8.0f));
        ImGui::TextColored(ui_obs::palette::TextMuted, "Calibration (PPM)");

        ImGui::BeginDisabled(writer_.is_recording());
        ImGui::SetNextItemWidth(half_w);
        if (ImGui::BeginCombo("##SampleRate", cur_rate)) {
            for (int i = 0; i < static_cast<int>(rates.size()); ++i) {
                const bool is_sel = (selected_sample_rate_idx_ == i);
                if (ImGui::Selectable(rates[static_cast<size_t>(i)].label, is_sel)) {
                    selected_sample_rate_idx_ = i;
                    sdr_.request_sample_rate(rates[static_cast<size_t>(i)].rate_hz);
                    dsp_.flush();
                }
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        ImGui::SameLine(0.0f, ui_obs::dp(6.0f));
        ImGui::SetNextItemWidth(half_w);
        if (ui_obs::widgets::SliderIntAccent("##PpmOffset", &ppm_correction_, -100, 100, "%d ppm", ui_obs::palette::AccentBlue)) {
            sdr_.request_ppm(ppm_correction_);
        }

        ImGui::BeginDisabled(agc_enabled_);
        ImGui::SetNextItemWidth(-1.0f);
        const float g_min = valid_gains_.empty() ? 0.0f : valid_gains_.front();
        const float g_max = valid_gains_.empty() ? 49.6f : valid_gains_.back();
        if (ui_obs::widgets::SliderFloatAccent("##LnaGain", &lna_gain_db_, g_min, g_max, "LNA Gain: %.1f dB", ui_obs::palette::Cyan)) {
            lna_gain_db_ = snap_gain_to_valid(lna_gain_db_);
            sdr_.request_gain(lna_gain_db_, false);
        }
        ImGui::EndDisabled();

        if (ui_obs::widgets::ToggleSwitch("AGC", &agc_enabled_, ui_obs::palette::Emerald)) {
            sdr_.request_gain(lna_gain_db_, agc_enabled_);
        }
        ImGui::SameLine(0.0f, ui_obs::dp(12.0f));
        bool bias_ui = bias_tee_enabled_;
        if (ui_obs::widgets::ToggleSwitch("Bias-Tee 4.5V", &bias_ui, ui_obs::palette::Amber)) {
            if (bias_ui && !bias_tee_enabled_) {
                open_bias_modal_ = true;
            } else if (!bias_ui) {
                bias_tee_enabled_ = false;
                sdr_.request_bias_tee(false);
            }
        }
    }
    ui_obs::widgets::EndCard();
}

void UiRecordPanel::render_dsp_card() {
    const auto& ffts = SdrConstants::GetFftSizes();
    char fft_info[24];
    std::snprintf(fft_info, sizeof(fft_info), "%d bins", ffts[static_cast<size_t>(selected_fft_size_idx_)]);

    if (ui_obs::widgets::BeginCard(
            "##CardDsp", fonts_.semibold, "SPECTRUM & DISPLAY",
            ui_obs::palette::Cyan, fft_info, ui_obs::palette::TextMuted)) {
        const float half_w = std::max(70.0f, (ImGui::GetContentRegionAvail().x - ui_obs::dp(6.0f)) * 0.5f);

        ImGui::TextColored(ui_obs::palette::TextMuted, "FFT Size");
        ImGui::SameLine(half_w + ui_obs::dp(8.0f));
        ImGui::TextColored(ui_obs::palette::TextMuted, "Waterfall Palette");

        char fft_lbl[24];
        std::snprintf(fft_lbl, sizeof(fft_lbl), "%d", ffts[static_cast<size_t>(selected_fft_size_idx_)]);
        ImGui::SetNextItemWidth(half_w);
        if (ImGui::BeginCombo("##FftBins", fft_lbl)) {
            for (int i = 0; i < static_cast<int>(ffts.size()); ++i) {
                char opt[24];
                std::snprintf(opt, sizeof(opt), "%d bins", ffts[static_cast<size_t>(i)]);
                const bool is_sel = (selected_fft_size_idx_ == i);
                if (ImGui::Selectable(opt, is_sel)) {
                    selected_fft_size_idx_ = i;
                    dsp_.set_fft_size(ffts[static_cast<size_t>(i)]);
                }
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine(0.0f, ui_obs::dp(6.0f));

        const auto& palettes = Waterfall::get_palette_names();
        const char* cur_pal = (selected_palette_idx_ >= 0 && selected_palette_idx_ < static_cast<int>(palettes.size()))
            ? palettes[static_cast<size_t>(selected_palette_idx_)].c_str()
            : "Viridis";
        ImGui::SetNextItemWidth(half_w);
        if (ImGui::BeginCombo("##WfPalette", cur_pal)) {
            for (int i = 0; i < static_cast<int>(palettes.size()); ++i) {
                const bool is_sel = (selected_palette_idx_ == i);
                if (ImGui::Selectable(palettes[static_cast<size_t>(i)].c_str(), is_sel)) {
                    selected_palette_idx_ = i;
                    waterfall_.set_palette(static_cast<WaterfallPalette>(i));
                }
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Full-width PSD Exponential Averaging (0.00 = Instantaneous FFT -> 0.95 = Deep Integration)
        ImGui::SetNextItemWidth(-1.0f);
        if (ui_obs::widgets::SliderFloatAccent("##AvgFactor", &avg_factor_, 0.0f, 0.95f, "PSD Smoothing: %.2f", ui_obs::palette::Cyan)) {
            dsp_.set_averaging(avg_factor_);
        }

        // Dynamic range header + Auto Fit button
        ImGui::TextColored(ui_obs::palette::TextMuted, "Range (dBFS)");
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - ui_obs::dp(58.0f));
        if (ImGui::SmallButton("Auto Fit")) {
            auto_fit_db_range();
        }

        // Side-by-side Min / Max sliders (eliminates clipped right-hand labels)
        ImGui::SetNextItemWidth(half_w);
        if (ui_obs::widgets::SliderFloatAccent("##MinDb", &fft_min_db_, -140.0f, -15.0f, "Min %.0f dB", ui_obs::palette::AccentBlue)) {
            if (fft_min_db_ > fft_max_db_ - 5.0f) fft_min_db_ = fft_max_db_ - 5.0f;
            waterfall_.recolor(fft_min_db_, fft_max_db_);
        }
        ImGui::SameLine(0.0f, ui_obs::dp(6.0f));
        ImGui::SetNextItemWidth(half_w);
        if (ui_obs::widgets::SliderFloatAccent("##MaxDb", &fft_max_db_, -130.0f, 10.0f, "Max %.0f dB", ui_obs::palette::AccentBlue)) {
            if (fft_max_db_ < fft_min_db_ + 5.0f) fft_max_db_ = fft_min_db_ + 5.0f;
            waterfall_.recolor(fft_min_db_, fft_max_db_);
        }

        if (ImGui::BeginTable("##DspToggles", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ui_obs::widgets::ToggleSwitch("Fill", &fill_spectrum_, ui_obs::palette::Cyan);
            ImGui::TableSetColumnIndex(1);
            ui_obs::widgets::ToggleSwitch("Waterfall", &show_waterfall_, ui_obs::palette::AccentBlue);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ui_obs::widgets::ToggleSwitch("Pk Hold", &show_peak_hold_, ui_obs::palette::Amber);
            ImGui::EndTable();
        }
    }
    ui_obs::widgets::EndCard();
}

void UiRecordPanel::render_recording_card() {
    auto now = std::chrono::steady_clock::now();
    if (last_stats_time_.time_since_epoch().count() == 0 ||
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time_).count() >= 180) {
        last_stats_ = writer_.get_stats();
        last_stats_time_ = now;
    }

    const bool is_rec = writer_.is_recording();

    if (ui_obs::widgets::BeginCard(
            "##CardRecord", fonts_.semibold, "IQ CAPTURE",
            is_rec ? ui_obs::palette::Red : ui_obs::palette::Emerald,
            is_rec ? "Recording" : "CS16",
            is_rec ? ui_obs::palette::Red : ui_obs::palette::TextMuted)) {
        const float w_avail = ImGui::GetContentRegionAvail().x;
        const float dur_w = std::clamp(w_avail * 0.36f, ui_obs::dp(76.0f), ui_obs::dp(102.0f));
        const float pref_w = std::max(80.0f, w_avail - dur_w - ui_obs::dp(6.0f));

        ImGui::BeginDisabled(is_rec);
        ImGui::TextColored(ui_obs::palette::TextMuted, "Prefix");
        ImGui::SameLine(pref_w + ui_obs::dp(6.0f));
        ImGui::TextColored(ui_obs::palette::TextMuted, "Duration (0=Man)");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Auto-stop timer in seconds (0 s = manual stop)");
        }

        ImGui::SetNextItemWidth(pref_w);
        ImGui::InputText("##RecPrefix", record_prefix_buf_, sizeof(record_prefix_buf_));
        ImGui::SameLine(0.0f, ui_obs::dp(6.0f));
        ImGui::SetNextItemWidth(dur_w);
        if (ImGui::InputFloat("##RecDur", &record_duration_s_, 0.0f, 0.0f, "%.0f s")) {
            if (record_duration_s_ < 0.0f) record_duration_s_ = 0.0f;
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Auto-stop timer in seconds (0 s = manual stop)");
        }
        ImGui::EndDisabled();

        const ImVec4 cap_col = is_rec ? ui_obs::palette::Red : ui_obs::palette::AccentBlue;
        ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(cap_col, 0.24f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(cap_col, 0.42f));
        ImGui::PushStyleColor(ImGuiCol_Border, cap_col);
        {
            ui_obs::ScopedFont bf(fonts_.semibold, 13.5f);
            const char* rec_btn_lbl = is_rec ? "Stop Recording CS16" : "Start Recording CS16";
            if (ImGui::Button(rec_btn_lbl, ImVec2(-1.0f, ui_obs::dp(30.0f)))) {
                if (!is_rec) {
                    if (!sdr_.is_streaming()) sdr_.start_streaming();
                    const std::string custom_path = build_timestamped_cs16_path(
                        writer_.get_output_dir(), record_prefix_buf_, sdr_.get_current_frequency());
                    const std::string path = writer_.start_recording(
                        sdr_.get_current_sample_rate(),
                        sdr_.get_current_frequency(),
                        sdr_.get_device_name(),
                        sdr_.get_current_gain(),
                        sdr_.is_bias_tee_enabled(),
                        custom_path,
                        static_cast<double>(record_duration_s_)
                    );
                    if (!path.empty()) {
                        app_state_.set_recording_active(true, path);
                        app_state_.set_last_recorded_file(path);
                    }
                } else {
                    writer_.stop_recording();
                    app_state_.set_recording_active(false);
                    const std::string last = writer_.get_last_recorded_filepath();
                    if (!last.empty()) app_state_.set_last_recorded_file(last);
                }
                last_stats_ = writer_.get_stats();
            }
        }
        ImGui::PopStyleColor(3);

        if (is_rec && record_duration_s_ > 0.0f) {
            const float target_s = std::max(0.1f, record_duration_s_);
            const float el_s = static_cast<float>(std::max(0.0, last_stats_.elapsed_seconds));
            const float frac = std::clamp(el_s / target_s, 0.0f, 1.0f);
            const float rem_s = std::max(0.0f, target_s - el_s);
            char prog_lbl[64];
            std::snprintf(prog_lbl, sizeof(prog_lbl), "%.1f / %.0f s (%.0f%%)  |  -%.1f s",
                          el_s, target_s, frac * 100.0f, rem_s);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ui_obs::palette::Tint(ui_obs::palette::Red, 0.72f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ui_obs::palette::BgDeep);
            ImGui::ProgressBar(frac, ImVec2(-1.0f, ui_obs::dp(18.0f)), prog_lbl);
            ImGui::PopStyleColor(2);
        }

        if (ImGui::BeginTable("##RecStatsTbl", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(is_rec ? ui_obs::palette::Red : ui_obs::palette::TextMuted, "State: %s", last_stats_.status.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ui_obs::palette::TextMuted, "Size: %s", last_stats_.size_str.c_str());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ui_obs::palette::TextMuted, "Time: %s", last_stats_.elapsed_str.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(last_stats_.dropped_blocks > 0 ? ui_obs::palette::Red : ui_obs::palette::TextMuted,
                               "Drops: %llu", static_cast<unsigned long long>(last_stats_.dropped_blocks));
            ImGui::EndTable();
        }

        std::string last_file = app_state_.get_last_recorded_file();
        if (last_file.empty()) last_file = writer_.get_last_recorded_filepath();
        const bool can_bridge = (!is_rec && !last_file.empty());

        const float folder_w = ui_obs::dp(78.0f);
        ImGui::BeginDisabled(!can_bridge);
        ImGui::PushStyleColor(ImGuiCol_Button, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.22f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ui_obs::palette::Tint(ui_obs::palette::AccentBlue, 0.40f));
        ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::AccentBlue);
        if (ImGui::Button("Analyze Capture ->", ImVec2(std::max(80.0f, ImGui::GetContentRegionAvail().x - folder_w - ui_obs::dp(6.0f)), ui_obs::dp(28.0f)))) {
            app_state_.switch_to_analyze(last_file, true);
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Folder...", ImVec2(folder_w, ui_obs::dp(28.0f)))) {
            ui_obs::win_fs::open_folder_in_explorer(writer_.get_output_dir());
        }
    }
    ui_obs::widgets::EndCard();
}

void UiRecordPanel::render_bias_tee_modal() {
    if (open_bias_modal_) {
        ImGui::OpenPopup("Confirm +4.5V Bias-Tee");
        open_bias_modal_ = false;
    }
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Confirm +4.5V Bias-Tee", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ui_obs::palette::Amber, "WARNING: DC Power Injection (+4.5V)");
        ImGui::Spacing();
        ImGui::TextWrapped("Enabling the RTL-SDR Bias-Tee sends +4.5V DC directly onto the RF SMA center pin.");
        ImGui::TextWrapped("Only enable if your antenna feed has an active LNA or DC block. Shorting DC to ground can damage the tuner.");
        ImGui::Separator();

        if (ImGui::Button("Enable +4.5V Bias-Tee", ImVec2(ui_obs::dp(175.0f), ui_obs::dp(30.0f)))) {
            bias_tee_enabled_ = true;
            sdr_.request_bias_tee(true);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(ui_obs::dp(100.0f), ui_obs::dp(30.0f)))) {
            bias_tee_enabled_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void UiRecordPanel::render_display_area(ID3D11DeviceContext* context, float width, float height) {
    ImGui::BeginChild("##RecDisplayCol", ImVec2(width, height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const uint64_t f_hz = sdr_.get_current_frequency();
    const uint32_t rate_hz = sdr_.get_current_sample_rate();
    const bool has_new = dsp_.get_power_spectrum(spectrum_data_, peak_freq_hz_, peak_dbfs_, f_hz, rate_hz);
    if (has_new) {
        pending_wf_sample_ = true;
    }
    if (has_new || peak_hold_data_.empty()) {
        update_peak_hold(static_cast<double>(f_hz), static_cast<double>(rate_hz));
    }

    const float gap = ui_obs::dp(6.0f);
    if (!show_waterfall_) hover_wf_active_ = false;

    if (show_waterfall_) {
        const float spec_h = std::max(160.0f, std::floor((height - gap) * 0.52f));
        const float wf_h   = std::max(140.0f, height - spec_h - gap);
        render_spectrum_card(width, spec_h);
        ImGui::Dummy(ImVec2(0.0f, gap));
        render_waterfall_card(context, width, wf_h);
    } else {
        render_spectrum_card(width, height);
    }

    ImGui::EndChild();
}

void UiRecordPanel::render_spectrum_card(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
    ImGui::BeginChild("##RecSpectrumCard", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wpos = ImGui::GetWindowPos();

    // Inline Header Strip + Right Quick Peak-Hold Controls
    {
        ui_obs::ScopedFont hf(fonts_.semibold, 13.0f);
        ImGui::TextColored(ui_obs::palette::TextPrimary, "POWER SPECTRUM (PSD)");
    }
    ImGui::SameLine(0.0f, ui_obs::dp(12.0f));
    {
        ui_obs::ScopedFont mf(fonts_.mono, 12.0f);
        ImGui::TextColored(ui_obs::palette::Cyan, "Peak: %.4f MHz (%.1f dBFS)", peak_freq_hz_ / 1e6, peak_dbfs_);
        ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
        ImGui::TextColored(ui_obs::palette::TextMuted, "Floor: %.1f dBFS | SNR: %.1f dB",
                           dsp_.get_noise_floor_db(), dsp_.get_snr_db());
    }

    const float pk_w = ui_obs::dp(68.0f);
    const float clr_w = show_peak_hold_ ? ui_obs::dp(40.0f) : 0.0f;
    const float total_ctrl_w = pk_w + (clr_w > 0.0f ? clr_w + ui_obs::dp(4.0f) : 0.0f);
    const float right_ctrl_x = wpos.x + width - total_ctrl_w - ui_obs::dp(10.0f);
    if (right_ctrl_x > ImGui::GetItemRectMax().x + ui_obs::dp(8.0f)) {
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(right_ctrl_x, ImGui::GetItemRectMin().y - 1.0f));
        ui_obs::ScopedFont bf(fonts_.semibold, 11.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ui_obs::dp(3.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ui_obs::dp(5.0f), ui_obs::dp(1.5f)));
        ImGui::PushStyleColor(ImGuiCol_Button, show_peak_hold_
            ? ui_obs::palette::Tint(ui_obs::palette::Amber, 0.22f)
            : ui_obs::palette::BgWell);
        ImGui::PushStyleColor(ImGuiCol_Border, show_peak_hold_
            ? ui_obs::palette::Tint(ui_obs::palette::Amber, 0.65f)
            : ui_obs::palette::BorderSubtle);
        ImGui::PushStyleColor(ImGuiCol_Text, show_peak_hold_ ? ui_obs::palette::Amber : ui_obs::palette::TextMuted);
        if (ImGui::Button("PK HOLD", ImVec2(pk_w, 0.0f))) {
            show_peak_hold_ = !show_peak_hold_;
        }
        ImGui::PopStyleColor(3);
        if (show_peak_hold_) {
            ImGui::SameLine(0.0f, ui_obs::dp(4.0f));
            if (ImGui::Button("CLR", ImVec2(clr_w, 0.0f))) {
                peak_hold_data_.clear();
            }
        }
        ImGui::PopStyleVar(2);
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x > 100.0f && avail.y > 54.0f) {
        const ImVec2 origin = ImGui::GetCursorScreenPos();

        const double fc_mhz = static_cast<double>(sdr_.get_current_frequency()) / 1e6;
        const double bw_mhz = std::max(1e-6, static_cast<double>(sdr_.get_current_sample_rate()) / 1e6);
        const double full_x_min = fc_mhz - 0.5 * bw_mhz;
        const double full_x_max = fc_mhz + 0.5 * bw_mhz;

        double view_x_min = full_x_min;
        double view_x_max = full_x_max;
        if (live_zoom_.x_active) {
            live_zoom_.x_min = std::clamp(live_zoom_.x_min, 0.0, 1.0);
            live_zoom_.x_max = std::clamp(live_zoom_.x_max, 0.0, 1.0);
            if (live_zoom_.x_max - live_zoom_.x_min < 1e-4) {
                live_zoom_.reset();
            } else {
                view_x_min = full_x_min + live_zoom_.x_min * bw_mhz;
                view_x_max = full_x_min + live_zoom_.x_max * bw_mhz;
            }
        }

        ui_obs::PlotMargin m;
        m.left   = ui_obs::dp(kAlignedLeftMarginBase);
        m.right  = ui_obs::dp(kAlignedRightMarginBase);
        m.top    = ui_obs::dp(14.0f);
        m.bottom = ui_obs::dp(30.0f);

        auto vp = ui_obs::PlotViewport2D::create(origin, avail, view_x_min, view_x_max, fft_min_db_, fft_max_db_, m);
        vp.plot_min.x = std::floor(vp.plot_min.x);
        vp.plot_max.x = std::floor(vp.plot_max.x);
        vp.plot_w = std::max(1.0f, vp.plot_max.x - vp.plot_min.x);

        vp.draw_background_and_grid(dl, 7, 5, "%.2f", "%.0f", show_waterfall_ ? nullptr : "Frequency (MHz)", "dBFS", fonts_.mono);

        const ImU32 cyan_stroke   = ui_obs::palette::U32(ui_obs::palette::Cyan);
        const ImU32 peak_top_fill = ui_obs::palette::WithAlphaF(cyan_stroke, 0.36f);
        const ImU32 base_top_fill = ui_obs::palette::WithAlphaF(cyan_stroke, 0.12f);
        const ImU32 floor_fill    = ui_obs::palette::WithAlphaF(cyan_stroke, 0.015f);

        if (spectrum_data_.size() >= 2) {
            vp.draw_decimated_trace_with_fill(
                dl, spectrum_data_.size(), full_x_min, full_x_max,
                [this](size_t i) -> double { return static_cast<double>(spectrum_data_[i]); },
                cyan_stroke, 1.65f,
                fill_spectrum_, peak_top_fill, base_top_fill, floor_fill,
                trace_scratch_
            );
        }

        if (show_peak_hold_ && peak_hold_data_.size() == spectrum_data_.size() && peak_hold_data_.size() >= 2) {
            vp.draw_decimated_trace_with_fill(
                dl, peak_hold_data_.size(), full_x_min, full_x_max,
                [this](size_t i) -> double { return static_cast<double>(peak_hold_data_[i]); },
                ui_obs::palette::U32(ui_obs::palette::Amber, 0.85f), 1.25f,
                false, 0, 0, 0,
                peak_scratch_
            );
        }

        // Dashed H I Rest Frequency Marker (clean flat tag, no overlapping capsule boxes)
        const double hi_mhz = SdrConstants::HI_REST_FREQ_HZ * 1e-6;
        ImVec2 hi_box_min(0, 0), hi_box_max(0, 0);
        bool has_hi_tag = false;
        if (hi_mhz >= view_x_min && hi_mhz <= view_x_max) {
            const float hx = std::floor(vp.map_x(hi_mhz)) + 0.5f;
            const ImU32 hi_col = IM_COL32(63, 185, 80, 225);
            ui_obs::widgets::draw_dashed_vline(dl, hx, vp.plot_min.y, vp.plot_max.y,
                                               ui_obs::palette::WithAlphaF(hi_col, 0.65f), 1.15f, 5.0f, 4.0f);
            ImFont* mf = fonts_.mono ? fonts_.mono : ImGui::GetFont();
            const float mfs = ui_obs::font_dp(11.0f);
            const char* hi_lbl = "HI 1420.406";
            const ImVec2 hsz = mf->CalcTextSizeA(mfs, FLT_MAX, 0.0f, hi_lbl);
            const float hi_lo_x = vp.plot_min.x + ui_obs::dp(4.0f);
            const float tx = std::floor(std::clamp(hx + ui_obs::dp(4.0f), hi_lo_x, std::max(hi_lo_x, vp.plot_max.x - hsz.x - ui_obs::dp(6.0f))));
            const float ty = std::floor(vp.plot_min.y + ui_obs::dp(4.0f));
            hi_box_min = ImVec2(tx - ui_obs::dp(3.0f), ty - ui_obs::dp(1.0f));
            hi_box_max = ImVec2(tx + hsz.x + ui_obs::dp(3.0f), ty + hsz.y + ui_obs::dp(1.0f));
            dl->AddRectFilled(hi_box_min, hi_box_max, IM_COL32(10, 14, 20, 210), ui_obs::dp(2.0f));
            dl->AddText(mf, mfs, ImVec2(tx, ty), hi_col, hi_lbl);
            has_hi_tag = true;
        }

        // Peak Diamond Marker + Collision-Free Flat Readout Tag
        const double pk_mhz = peak_freq_hz_ * 1e-6;
        if (!spectrum_data_.empty() && pk_mhz >= view_x_min && pk_mhz <= view_x_max && std::isfinite(peak_dbfs_)) {
            const float px = std::clamp(vp.map_x(pk_mhz), vp.plot_min.x, vp.plot_max.x);
            const float py = std::clamp(vp.map_y(peak_dbfs_), vp.plot_min.y, vp.plot_max.y);
            const ImU32 pk_col = IM_COL32(251, 113, 133, 255);

            const float r = ui_obs::dp(4.5f);
            dl->AddQuadFilled(ImVec2(px, py - r), ImVec2(px + r, py),
                              ImVec2(px, py + r), ImVec2(px - r, py), pk_col);

            char pk_buf[40];
            std::snprintf(pk_buf, sizeof(pk_buf), "PK %.1f dB | %.3f", peak_dbfs_, pk_mhz);
            ImFont* mf = fonts_.mono ? fonts_.mono : ImGui::GetFont();
            const float mfs = ui_obs::font_dp(11.0f);
            const ImVec2 psz = mf->CalcTextSizeA(mfs, FLT_MAX, 0.0f, pk_buf);
            float tx = px + ui_obs::dp(7.0f);
            if (tx + psz.x + ui_obs::dp(6.0f) > vp.plot_max.x) {
                tx = px - psz.x - ui_obs::dp(7.0f);
            }
            const float pk_lo_x = vp.plot_min.x + ui_obs::dp(4.0f);
            const float pk_lo_y = vp.plot_min.y + ui_obs::dp(4.0f);
            tx = std::floor(std::clamp(tx, pk_lo_x, std::max(pk_lo_x, vp.plot_max.x - psz.x - ui_obs::dp(4.0f))));
            float ty = std::floor(std::clamp(py - psz.y * 0.5f, pk_lo_y, std::max(pk_lo_y, vp.plot_max.y - psz.y - ui_obs::dp(4.0f))));

            if (has_hi_tag) {
                const bool x_ov = tx < (hi_box_max.x + ui_obs::dp(6.0f)) && (tx + psz.x) > (hi_box_min.x - ui_obs::dp(6.0f));
                const bool y_ov = ty < (hi_box_max.y + ui_obs::dp(4.0f)) && (ty + psz.y) > (hi_box_min.y - ui_obs::dp(4.0f));
                if (x_ov && y_ov) {
                    ty = std::floor(std::min(std::max(pk_lo_y, vp.plot_max.y - psz.y - ui_obs::dp(4.0f)), hi_box_max.y + ui_obs::dp(5.0f)));
                }
            }
            dl->AddRectFilled(ImVec2(tx - ui_obs::dp(3.0f), ty - ui_obs::dp(1.0f)),
                              ImVec2(tx + psz.x + ui_obs::dp(3.0f), ty + psz.y + ui_obs::dp(1.0f)),
                              IM_COL32(10, 14, 20, 215), ui_obs::dp(2.0f));
            dl->AddText(mf, mfs, ImVec2(tx, ty), pk_col, pk_buf);
        }

        // Right-gutter live dBFS Signal Level Meter (directly aligned above Waterfall colorbar)
        const ImVec2 meter_min(vp.plot_max.x + ui_obs::dp(kSharedBarOffsetXBase), vp.plot_min.y + ui_obs::dp(12.0f));
        const ImVec2 meter_max(meter_min.x + ui_obs::dp(kSharedBarWidthBase), vp.plot_max.y);
        ui_obs::draw_signal_level_bar(
            dl, meter_min, meter_max, fft_min_db_, fft_max_db_,
            peak_dbfs_, dsp_.get_noise_floor_db(), peak_hold_max_db_,
            show_peak_hold_, waterfall_.get_palette_lut(), fonts_.mono
        );

        // Interactive 2D Zoom/Pan, Floating Toolbar, Context Menu & Ctrl+Click LO re-tune
        ui_obs::PlotInteractConfig icfg;
        icfg.def_xmin = full_x_min;
        icfg.def_xmax = full_x_max;
        icfg.bound_xmin = full_x_min;
        icfg.bound_xmax = full_x_max;
        // Use current fft_min_db_/fft_max_db_ as def_ymin/ymax so Shift+Wheel/box-zoom
        // always maps against the active display range rather than jumping to [-95, -15].
        icfg.def_ymin = fft_min_db_;
        icfg.def_ymax = fft_max_db_;
        icfg.bound_ymin = -140.0;
        icfg.bound_ymax = 10.0;
        icfg.clamp_x = true;
        icfg.clamp_y = true;
        icfg.x_unit = "MHz";
        icfg.y_unit = "dBFS";
        icfg.auto_y_label = "Auto dB";
        icfg.ignore_ctrl_left_drag = true;
        icfg.has_hi_focus = (hi_mhz >= full_x_min && hi_mhz <= full_x_max);
        icfg.hi_xmin = std::max(full_x_min, hi_mhz - 0.22);
        icfg.hi_xmax = std::min(full_x_max, hi_mhz + 0.22);

        spec_interact_zoom_.x_active = live_zoom_.x_active;
        spec_interact_zoom_.x_min = view_x_min;
        spec_interact_zoom_.x_max = view_x_max;
        spec_interact_zoom_.y_min = fft_min_db_;
        spec_interact_zoom_.y_max = fft_max_db_;
        spec_interact_zoom_.sync_active();

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            ImGui::IsMouseHoveringRect(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), true)) {
            context_tune_freq_mhz_ = std::clamp(vp.unmap_x(ImGui::GetMousePos().x), full_x_min, full_x_max);
        }

        const auto ires = ui_obs::handle_plot_interaction(
            "##SpecCanvas", origin, avail, vp.plot_min, vp.plot_max,
            icfg, spec_interact_zoom_, dl, fonts_.semibold, fonts_.mono,
            [&]() {
                if (context_tune_freq_mhz_ > 0.0) {
                    char tune_lbl[64];
                    std::snprintf(tune_lbl, sizeof(tune_lbl), "Tune LO to %.4f MHz", context_tune_freq_mhz_);
                    if (ImGui::MenuItem(tune_lbl, "Ctrl+Click")) {
                        apply_tuned_frequency_hz(context_tune_freq_mhz_ * 1.0e6);
                    }
                }
                if (peak_freq_hz_ > 0.0) {
                    char pk_lbl[64];
                    std::snprintf(pk_lbl, sizeof(pk_lbl), "Tune LO to Live Peak (%.4f MHz)", peak_freq_hz_ * 1e-6);
                    if (ImGui::MenuItem(pk_lbl)) {
                        apply_tuned_frequency_hz(peak_freq_hz_);
                    }
                }
                if (ImGui::MenuItem("Tune LO for 21cm H I (1420.5000 MHz, +94 kHz LO)")) {
                    apply_tuned_frequency_hz(1420.5e6);
                }
                if (ImGui::MenuItem(show_peak_hold_ ? "Disable Peak Hold" : "Enable Peak Hold", "P")) {
                    show_peak_hold_ = !show_peak_hold_;
                }
            },
            true
        );

        if (ires.x_changed) {
            if (spec_interact_zoom_.x_active) {
                const double u0 = std::clamp((spec_interact_zoom_.x_min - full_x_min) / bw_mhz, 0.0, 1.0);
                const double u1 = std::clamp((spec_interact_zoom_.x_max - full_x_min) / bw_mhz, 0.0, 1.0);
                if (u1 - u0 > 1e-4 && (u1 - u0) < 0.9995) {
                    live_zoom_.x_min = u0;
                    live_zoom_.x_max = u1;
                    live_zoom_.x_active = true;
                } else {
                    live_zoom_.x_active = false;
                    live_zoom_.x_min = 0.0;
                    live_zoom_.x_max = 1.0;
                }
            } else {
                live_zoom_.x_active = false;
                live_zoom_.x_min = 0.0;
                live_zoom_.x_max = 1.0;
            }
            live_zoom_.sync_active();
        }
        if (ires.requested_auto_y) {
            auto_fit_db_range();
        } else if (ires.requested_reset) {
            fft_min_db_ = SdrConstants::DEFAULT_FFT_MIN_DB;
            fft_max_db_ = SdrConstants::DEFAULT_FFT_MAX_DB;
            waterfall_.recolor(fft_min_db_, fft_max_db_);
        } else if (ires.y_changed) {
            // Apply Y change even when y_manual resets to false on full zoom-out
            const double next_ymin = spec_interact_zoom_.y_manual ? spec_interact_zoom_.y_min : icfg.bound_ymin;
            const double next_ymax = spec_interact_zoom_.y_manual ? spec_interact_zoom_.y_max : icfg.bound_ymax;
            fft_min_db_ = static_cast<float>(std::clamp(next_ymin, -140.0, 5.0));
            fft_max_db_ = static_cast<float>(std::clamp(next_ymax, static_cast<double>(fft_min_db_ + 5.0f), 10.0));
            waterfall_.recolor(fft_min_db_, fft_max_db_);
        }

        hover_spec_active_ = false;
        if (ires.hovered_plot) {
            const ImVec2 mp = ImGui::GetMousePos();
            const double hover_mhz = std::clamp(vp.unmap_x(mp.x), view_x_min, view_x_max);
            if (ImGui::GetIO().KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                apply_tuned_frequency_hz(hover_mhz * 1.0e6);
            }

            if (!ires.box_active) {
                hover_spec_active_ = true;
                hover_spec_freq_mhz_ = hover_mhz;

                ui_obs::widgets::draw_dashed_hline(dl, vp.plot_min.x, vp.plot_max.x, mp.y,
                                                   ui_obs::palette::PlotCrosshair, 1.0f, 4.0f, 4.0f);
                ui_obs::widgets::draw_dashed_vline(dl, mp.x, vp.plot_min.y, vp.plot_max.y,
                                                   ui_obs::palette::PlotCrosshair, 1.0f, 4.0f, 4.0f);

                float bin_db = static_cast<float>(vp.unmap_y(mp.y));
                float hold_db = bin_db;
                if (!spectrum_data_.empty()) {
                    const double norm_x = std::clamp((hover_mhz - full_x_min) / std::max(1e-9, bw_mhz), 0.0, 1.0);
                    const size_t idx = std::clamp(static_cast<size_t>(std::llround(norm_x * (spectrum_data_.size() - 1))),
                                                  size_t{0}, spectrum_data_.size() - 1);
                    bin_db = spectrum_data_[idx];
                    const float sy = std::clamp(vp.map_y(bin_db), vp.plot_min.y, vp.plot_max.y);
                    dl->AddCircleFilled(ImVec2(mp.x, sy), ui_obs::dp(3.5f), ui_obs::palette::U32(ui_obs::palette::Cyan), 12);
                    if (show_peak_hold_ && peak_hold_data_.size() == spectrum_data_.size()) {
                        hold_db = peak_hold_data_[idx];
                    }
                }

                char b_freq[32], b_off[32], b_pwr[32], b_vrad[32], b_hold[32];
                std::snprintf(b_freq, sizeof(b_freq), "%.6f MHz", hover_mhz);
                std::snprintf(b_off,  sizeof(b_off),  "%+.1f kHz", (hover_mhz - fc_mhz) * 1000.0);
                std::snprintf(b_pwr,  sizeof(b_pwr),  "%.2f dBFS", bin_db);
                std::snprintf(b_vrad, sizeof(b_vrad), "%+.1f km/s", SdrConstants::CalculateDopplerVelocityKmS(hover_mhz * 1.0e6));
                std::snprintf(b_hold, sizeof(b_hold), "%.2f dBFS", hold_db);

                ui_obs::widgets::TelemetryRow rows[5] = {
                    {"FREQ",  b_freq, ui_obs::palette::TextPrimary32},
                    {"dFC",   b_off,  ui_obs::palette::TextMuted32},
                    {"PSD",   b_pwr,  ui_obs::palette::U32(ui_obs::palette::Cyan)},
                    {"HOLD",  b_hold, ui_obs::palette::U32(ui_obs::palette::Amber)},
                    {"V_HI",  b_vrad, IM_COL32(142, 236, 158, 255)},
                };
                int row_cnt = 4;
                if (show_peak_hold_ && !peak_hold_data_.empty()) {
                    row_cnt = 5;
                } else {
                    rows[3] = rows[4];
                }
                ui_obs::widgets::draw_crosshair_card(
                    dl, mp, vp.plot_min, vp.plot_max,
                    "SPECTRUM CURSOR", ui_obs::palette::U32(ui_obs::palette::Cyan),
                    rows, row_cnt, fonts_.semibold, fonts_.mono
                );
            }
        } else if (hover_wf_active_ && hover_wf_freq_mhz_ >= view_x_min && hover_wf_freq_mhz_ <= view_x_max) {
            const float gx = std::floor(vp.map_x(hover_wf_freq_mhz_)) + 0.5f;
            ui_obs::widgets::draw_dashed_vline(dl, gx, vp.plot_min.y, vp.plot_max.y,
                                               ui_obs::palette::U32(ui_obs::palette::Cyan, 0.45f),
                                               1.0f, 4.0f, 4.0f);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void UiRecordPanel::render_waterfall_card(ID3D11DeviceContext* context, float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, ui_obs::palette::BorderSubtle);
    ImGui::BeginChild("##RecWaterfallCard", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();

    {
        ui_obs::ScopedFont hf(fonts_.semibold, 13.0f);
        ImGui::TextColored(ui_obs::palette::TextPrimary, "LIVE WATERFALL HISTORY");
    }
    ImGui::SameLine(0.0f, ui_obs::dp(12.0f));
    {
        ui_obs::ScopedFont mf(fonts_.mono, 12.0f);
        const auto& p_names = Waterfall::get_palette_names();
        const char* p_lbl = (selected_palette_idx_ >= 0 && selected_palette_idx_ < static_cast<int>(p_names.size()))
            ? p_names[static_cast<size_t>(selected_palette_idx_)].c_str()
            : "Viridis";
        const float hist_s = static_cast<float>(waterfall_.get_height()) * wf_row_dt_s_;
        ImGui::TextColored(ui_obs::palette::TextMuted, "%s | %dx%d px | %.1fs span",
                           p_lbl, waterfall_.get_width(), waterfall_.get_height(), hist_s);
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float left_m = ui_obs::dp(kAlignedLeftMarginBase);
    const float right_m = ui_obs::dp(kAlignedRightMarginBase);
    if (avail.x > left_m + right_m + 60.0f && avail.y > 48.0f) {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float wf_left  = std::floor(origin.x + left_m);
        const float wf_right = std::floor(origin.x + avail.x - right_m);
        const float wf_w     = std::max(64.0f, wf_right - wf_left);
        const float wf_top   = std::floor(origin.y + ui_obs::dp(16.0f));
        const float wf_h     = std::max(40.0f, std::floor(avail.y - ui_obs::dp(22.0f)));
        const ImVec2 wf_min(wf_left, wf_top);
        const ImVec2 wf_max(wf_left + wf_w, wf_top + wf_h);

        const int target_w = std::clamp(static_cast<int>(std::lround(wf_w)), 256, 2048);
        const int target_h = std::clamp(static_cast<int>(std::lround(wf_h)), 128, 512);
        if (std::abs(waterfall_.get_width() - target_w) >= 4 ||
            std::abs(waterfall_.get_height() - target_h) >= 4) {
            waterfall_.resize_preserving_history(target_w, target_h);
        }

        const auto now = std::chrono::steady_clock::now();
        if (sdr_.is_streaming() || pending_wf_sample_) {
            if (last_wf_update_time_.time_since_epoch().count() == 0) {
                if (!spectrum_data_.empty()) {
                    waterfall_.add_line(spectrum_data_, fft_min_db_, fft_max_db_);
                    pending_wf_sample_ = false;
                }
                last_wf_update_time_ = now;
            } else {
                const float elapsed_s = std::chrono::duration<float>(now - last_wf_update_time_).count();
                if (pending_wf_sample_ || elapsed_s >= 0.032f) {
                    if (!spectrum_data_.empty()) {
                        waterfall_.add_line(spectrum_data_, fft_min_db_, fft_max_db_);
                        pending_wf_sample_ = false;
                    }
                    wf_row_dt_s_ = 0.85f * wf_row_dt_s_ + 0.15f * std::clamp(elapsed_s, 0.015f, 0.25f);
                    last_wf_update_time_ = now;
                }
            }
        } else {
            last_wf_update_time_ = now;
        }

        update_waterfall_texture(context);

        const double fc_mhz = static_cast<double>(sdr_.get_current_frequency()) / 1e6;
        const double bw_mhz = std::max(1e-6, static_cast<double>(sdr_.get_current_sample_rate()) / 1e6);
        const double full_x_min = fc_mhz - 0.5 * bw_mhz;
        const double full_x_max = fc_mhz + 0.5 * bw_mhz;
        const double u0 = live_zoom_.x_active ? std::clamp(live_zoom_.x_min, 0.0, 1.0) : 0.0;
        const double u1 = live_zoom_.x_active ? std::clamp(live_zoom_.x_max, 0.0, 1.0) : 1.0;
        const double view_x_min = full_x_min + u0 * bw_mhz;
        const double view_x_max = full_x_min + u1 * bw_mhz;
        const double view_bw_mhz = std::max(1e-9, view_x_max - view_x_min);

        if (waterfall_srv_) {
            dl->AddImage(reinterpret_cast<ImTextureID>(waterfall_srv_), wf_min, wf_max,
                         ImVec2(static_cast<float>(u0), 0.0f), ImVec2(static_cast<float>(u1), 1.0f));
        } else {
            dl->AddRectFilled(wf_min, wf_max, ui_obs::palette::PlotBg32);
        }

        // Left Time-Axis Ticks inside aligned 60px gutter
        const float hist_s = static_cast<float>(waterfall_.get_height()) * wf_row_dt_s_;
        ImFont* mf = fonts_.mono ? fonts_.mono : ImGui::GetFont();
        const float mfs = ui_obs::font_dp(11.0f);
        constexpr int kTimeTicks = 4;
        for (int t = 0; t < kTimeTicks; ++t) {
            const float frac = static_cast<float>(t) / static_cast<float>(kTimeTicks - 1);
            const float ty = std::floor(wf_min.y + frac * wf_h);
            dl->AddLineH(wf_min.x - ui_obs::dp(4.0f), wf_min.x, ty, ui_obs::palette::PlotBorder32, 1.0f);
            char tbuf[16];
            if (t == 0) std::snprintf(tbuf, sizeof(tbuf), "now");
            else        std::snprintf(tbuf, sizeof(tbuf), "-%.0fs", frac * hist_s);
            const ImVec2 tsz = mf->CalcTextSizeA(mfs, FLT_MAX, 0.0f, tbuf);
            const float t_lo_y = wf_min.y - 2.0f;
            dl->AddText(mf, mfs, ImVec2(std::floor(wf_min.x - tsz.x - ui_obs::dp(6.0f)),
                                        std::floor(std::clamp(ty - tsz.y * 0.5f, t_lo_y, std::max(t_lo_y, wf_max.y - tsz.y)))),
                        ui_obs::palette::TextMuted32, tbuf);
        }

        const double hi_mhz = SdrConstants::HI_REST_FREQ_HZ * 1e-6;
        if (hi_mhz >= view_x_min && hi_mhz <= view_x_max) {
            const float hx = std::floor(wf_min.x + static_cast<float>((hi_mhz - view_x_min) / view_bw_mhz) * wf_w) + 0.5f;
            const ImU32 hi_col = IM_COL32(63, 185, 80, 220);
            dl->AddLineV(hx, wf_min.y, wf_min.y + ui_obs::dp(5.0f), hi_col, 1.5f);
            dl->AddLineV(hx, wf_max.y - ui_obs::dp(5.0f), wf_max.y, hi_col, 1.5f);
        }

        dl->AddRect(wf_min, wf_max, ui_obs::palette::PlotBorder32, 0.0f, 1.0f, ImDrawFlags_None);

        // Shared 1:1 right-gutter colorbar aligned with spectrum level meter above
        ui_obs::draw_vertical_colorbar(
            dl,
            ImVec2(wf_max.x + ui_obs::dp(kSharedBarOffsetXBase), wf_min.y),
            ImVec2(wf_max.x + ui_obs::dp(kSharedBarOffsetXBase + kSharedBarWidthBase), wf_max.y),
            fft_min_db_, fft_max_db_, "%.0f",
            waterfall_.get_palette_lut(), fonts_.mono, "dBFS"
        );

        ui_obs::PlotInteractConfig wcfg;
        wcfg.def_xmin = full_x_min;
        wcfg.def_xmax = full_x_max;
        wcfg.bound_xmin = full_x_min;
        wcfg.bound_xmax = full_x_max;
        wcfg.def_ymin = 0.0;
        wcfg.def_ymax = std::max(1.0, static_cast<double>(hist_s));
        wcfg.bound_ymin = 0.0;
        wcfg.bound_ymax = std::max(1.0, static_cast<double>(hist_s));
        wcfg.clamp_x = true;
        wcfg.allow_y_zoom = false;
        wcfg.y_inverted = true;
        wcfg.x_unit = "MHz";
        wcfg.y_unit = "s";
        wcfg.ignore_ctrl_left_drag = true;
        wcfg.has_hi_focus = (hi_mhz >= full_x_min && hi_mhz <= full_x_max);
        wcfg.hi_xmin = std::max(full_x_min, hi_mhz - 0.22);
        wcfg.hi_xmax = std::min(full_x_max, hi_mhz + 0.22);
        wcfg.is_compact = true;

        wf_interact_zoom_.x_active = live_zoom_.x_active;
        wf_interact_zoom_.x_min = view_x_min;
        wf_interact_zoom_.x_max = view_x_max;
        wf_interact_zoom_.y_manual = false;
        wf_interact_zoom_.sync_active();

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            ImGui::IsMouseHoveringRect(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), true)) {
            const double u_rc = std::clamp(static_cast<double>(ImGui::GetMousePos().x - wf_min.x) / static_cast<double>(wf_w), 0.0, 1.0);
            context_tune_freq_mhz_ = view_x_min + u_rc * view_bw_mhz;
        }

        const auto wres = ui_obs::handle_plot_interaction(
            "##WfCanvas", origin, avail, wf_min, wf_max,
            wcfg, wf_interact_zoom_, dl, fonts_.semibold, fonts_.mono,
            [&]() {
                if (context_tune_freq_mhz_ > 0.0) {
                    char tune_lbl[64];
                    std::snprintf(tune_lbl, sizeof(tune_lbl), "Tune LO to %.4f MHz", context_tune_freq_mhz_);
                    if (ImGui::MenuItem(tune_lbl, "Ctrl+Click")) {
                        apply_tuned_frequency_hz(context_tune_freq_mhz_ * 1.0e6);
                    }
                }
                if (peak_freq_hz_ > 0.0) {
                    char pk_lbl[64];
                    std::snprintf(pk_lbl, sizeof(pk_lbl), "Tune LO to Live Peak (%.4f MHz)", peak_freq_hz_ * 1e-6);
                    if (ImGui::MenuItem(pk_lbl)) {
                        apply_tuned_frequency_hz(peak_freq_hz_);
                    }
                }
                if (ImGui::MenuItem("Tune LO for 21cm H I (1420.5000 MHz, +94 kHz LO)")) {
                    apply_tuned_frequency_hz(1420.5e6);
                }
            },
            true
        );

        if (wres.x_changed) {
            if (wf_interact_zoom_.x_active) {
                const double nu0 = std::clamp((wf_interact_zoom_.x_min - full_x_min) / bw_mhz, 0.0, 1.0);
                const double nu1 = std::clamp((wf_interact_zoom_.x_max - full_x_min) / bw_mhz, 0.0, 1.0);
                if (nu1 - nu0 > 1e-4 && (nu1 - nu0) < 0.9995) {
                    live_zoom_.x_min = nu0;
                    live_zoom_.x_max = nu1;
                    live_zoom_.x_active = true;
                } else {
                    live_zoom_.x_active = false;
                    live_zoom_.x_min = 0.0;
                    live_zoom_.x_max = 1.0;
                }
            } else {
                live_zoom_.x_active = false;
                live_zoom_.x_min = 0.0;
                live_zoom_.x_max = 1.0;
            }
            live_zoom_.sync_active();
        }

        hover_wf_active_ = false;
        if (wres.hovered_plot) {
            const ImVec2 mp = ImGui::GetMousePos();
            const double u = std::clamp(static_cast<double>(mp.x - wf_min.x) / static_cast<double>(wf_w), 0.0, 1.0);
            const double freq = view_x_min + u * view_bw_mhz;
            if (ImGui::GetIO().KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                apply_tuned_frequency_hz(freq * 1.0e6);
            }

            if (!wres.box_active) {
                const float age_s = std::clamp((mp.y - wf_min.y) / wf_h, 0.0f, 1.0f) * hist_s;
                hover_wf_active_ = true;
                hover_wf_freq_mhz_ = freq;

                ui_obs::widgets::draw_dashed_vline(dl, mp.x, wf_min.y, wf_max.y,
                                                   ui_obs::palette::PlotCrosshair, 1.0f, 4.0f, 4.0f);
                ui_obs::widgets::draw_dashed_hline(dl, wf_min.x, wf_max.x, mp.y,
                                                   ui_obs::palette::PlotCrosshair, 1.0f, 4.0f, 4.0f);

                char w_freq[32], w_off[32], w_age[24], w_vrad[32];
                std::snprintf(w_freq, sizeof(w_freq), "%.6f MHz", freq);
                std::snprintf(w_off,  sizeof(w_off),  "%+.1f kHz", (freq - fc_mhz) * 1000.0);
                std::snprintf(w_age,  sizeof(w_age),  "-%.1f s", age_s);
                std::snprintf(w_vrad, sizeof(w_vrad), "%+.1f km/s", SdrConstants::CalculateDopplerVelocityKmS(freq * 1.0e6));
                ui_obs::widgets::TelemetryRow rows[4] = {
                    {"FREQ", w_freq, ui_obs::palette::TextPrimary32},
                    {"dFC",  w_off,  ui_obs::palette::TextMuted32},
                    {"AGE",  w_age,  ui_obs::palette::U32(ui_obs::palette::Cyan)},
                    {"V_HI", w_vrad, IM_COL32(142, 236, 158, 255)},
                };
                ui_obs::widgets::draw_crosshair_card(
                    dl, mp, wf_min, wf_max,
                    "WATERFALL CURSOR", ui_obs::palette::U32(ui_obs::palette::Cyan),
                    rows, 4, fonts_.semibold, fonts_.mono
                );
            }
        } else if (hover_spec_active_ && hover_spec_freq_mhz_ >= view_x_min && hover_spec_freq_mhz_ <= view_x_max) {
            const float gx = std::floor(wf_min.x + static_cast<float>((hover_spec_freq_mhz_ - view_x_min) / view_bw_mhz) * wf_w) + 0.5f;
            ui_obs::widgets::draw_dashed_vline(dl, gx, wf_min.y, wf_max.y,
                                               ui_obs::palette::U32(ui_obs::palette::Cyan, 0.45f),
                                               1.0f, 4.0f, 4.0f);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void UiRecordPanel::update_waterfall_texture(ID3D11DeviceContext* context) {
    if (!d3d_device_ || !context) return;

    const int w = waterfall_.get_width();
    const int h = waterfall_.get_height();
    if (w <= 0 || h <= 0) return;

    const uint64_t rev = waterfall_.get_revision();
    if (waterfall_texture_ && w == tex_width_ && h == tex_height_ && rev == wf_uploaded_rev_) {
        return;
    }

    if (!waterfall_texture_ || w != tex_width_ || h != tex_height_) {
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
            if (FAILED(d3d_device_->CreateShaderResourceView(waterfall_texture_, &srv_desc, &waterfall_srv_)) || !waterfall_srv_) {
                release_d3d_resources();
                return;
            }
            tex_width_ = w;
            tex_height_ = h;
        }
    }

    if (waterfall_texture_) {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(context->Map(waterfall_texture_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            const size_t row_bytes = static_cast<size_t>(w) * sizeof(uint32_t);
            if (mapped.RowPitch == row_bytes) {
                waterfall_.copy_linear_pixels(static_cast<uint32_t*>(mapped.pData));
            } else {
                const uint32_t* src = waterfall_.get_pixel_data();
                uint8_t* dst = static_cast<uint8_t*>(mapped.pData);
                for (int y = 0; y < h; ++y) {
                    std::memcpy(dst + static_cast<size_t>(y) * mapped.RowPitch,
                                src + static_cast<size_t>(y) * static_cast<size_t>(w),
                                row_bytes);
                }
            }
            context->Unmap(waterfall_texture_, 0);
            wf_uploaded_rev_ = rev;
        }
    }
}
