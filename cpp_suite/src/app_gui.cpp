#include "app_gui.h"
#include "constants.h"
#include "simd_dsp.h"

#include <psapi.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

AppGui::AppGui()
    : writer_(ui_obs::win_fs::get_default_recordings_dir()),
      waterfall_(SdrConstants::DEFAULT_WATERFALL_WIDTH, SdrConstants::DEFAULT_WATERFALL_HEIGHT),
      record_panel_(sdr_, dsp_, writer_, waterfall_, state_),
      analyze_panel_(state_)
{
    cached_simd_name_ = SimdDsp::get_simd_level_name();

    // Route live SDR packets into real-time FFT + async CS16 recorder
    sdr_.add_sample_callback([this](const uint8_t* data, size_t len) {
        dsp_.feed_samples(data, len);
        writer_.feed_samples(data, len);
    });
    sdr_.set_freq_changed_callback([this](uint64_t) {
        dsp_.flush();
    });

    sdr_.initialize(false, 0);
    sdr_.start_streaming();
    record_panel_.sync_device_selection();

    setup_observatory_theme();
    load_fonts();
}

AppGui::~AppGui() {
    sdr_.clear_sample_callbacks();
    sdr_.set_freq_changed_callback(nullptr);
    sdr_.shutdown();
    writer_.stop_recording();
    release_d3d_resources();
}

void AppGui::set_owner_hwnd(HWND hwnd) {
    analyze_panel_.set_owner_hwnd(hwnd);
}

void AppGui::setup_observatory_theme() {
    ui_obs::apply_observatory_slate_style(ImGui::GetStyle().FontScaleDpi);
}

void AppGui::load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    wchar_t win_dir_w[MAX_PATH] = {};
    UINT len = GetWindowsDirectoryW(win_dir_w, MAX_PATH);
    std::filesystem::path fonts_dir = (len > 0 && len < MAX_PATH)
        ? (std::filesystem::path(win_dir_w) / "Fonts")
        : std::filesystem::path("C:/Windows/Fonts");

    auto try_load = [&](const char* filename, float size_px, float raster_mult = 1.12f) -> ImFont* {
        std::filesystem::path p = fonts_dir / filename;
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            std::string utf8_p = ui_obs::win_fs::wide_to_utf8(p.c_str());
            if (!utf8_p.empty()) {
                ImFontConfig cfg;
                cfg.OversampleH = 2;
                cfg.OversampleV = 1;
                cfg.PixelSnapH = true;
                cfg.RasterizerMultiply = raster_mult;
                return io.Fonts->AddFontFromFileTTF(utf8_p.c_str(), std::round(size_px), &cfg);
            }
        }
        return nullptr;
    };

    // High-legibility technical UI & telemetry font stack (Calibri / Segoe UI Semibold + Bold + Consolas)
    fonts_.regular  = try_load("calibri.ttf", 16.0f, 1.16f);
    if (!fonts_.regular) {
        fonts_.regular = try_load("seguisb.ttf", 15.0f, 1.12f);
    }
    if (!fonts_.regular) {
        fonts_.regular = try_load("segoeui.ttf", 15.0f, 1.18f);
    }

    fonts_.semibold = try_load("calibrib.ttf", 16.0f, 1.08f);
    if (!fonts_.semibold) {
        fonts_.semibold = try_load("segoeuib.ttf", 15.0f, 1.08f);
    }
    if (!fonts_.semibold) {
        fonts_.semibold = try_load("seguisb.ttf", 16.0f, 1.12f);
    }

    // Crisp static TrueType monospace for numeric telemetry, axes, and KPI readouts
    fonts_.mono     = try_load("consola.ttf", 14.0f, 1.15f);
    if (!fonts_.mono) {
        fonts_.mono = try_load("CascadiaMono.ttf", 14.0f, 1.12f);
    }

    if (!fonts_.regular) {
        fonts_.regular = io.Fonts->AddFontDefault();
    }
    if (!fonts_.semibold) fonts_.semibold = fonts_.regular;
    if (!fonts_.mono)     fonts_.mono     = fonts_.regular;

    io.FontDefault = fonts_.regular;

    record_panel_.set_fonts(fonts_);
    analyze_panel_.set_fonts(fonts_);
}

void AppGui::init_d3d_resources(ID3D11Device* device) {
    d3d_device_ = device;
    record_panel_.init_d3d_resources(device);
    analyze_panel_.init_d3d_resources(device);
}

void AppGui::release_d3d_resources() {
    record_panel_.release_d3d_resources();
    analyze_panel_.release_d3d_resources();
}

size_t AppGui::poll_process_memory_mb() {
    auto now = std::chrono::steady_clock::now();
    if (last_ram_poll_.time_since_epoch().count() != 0) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ram_poll_).count();
        if (elapsed_ms < 500) {
            return cached_ram_mb_;
        }
    }
    last_ram_poll_ = now;
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        cached_ram_mb_ = static_cast<size_t>(pmc.WorkingSetSize / (1024 * 1024));
    }
    return cached_ram_mb_;
}

void AppGui::render_top_navigation_bar(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::HeaderBg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(14.0f), 0.0f));
    ImGui::BeginChild("##ObsTopNav", ImVec2(width, height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 win_pos = ImGui::GetWindowPos();
    dl->AddLineH(win_pos.x, win_pos.x + width, win_pos.y + height - 1.0f,
                 ui_obs::palette::U32(ui_obs::palette::BorderSubtle), 1.0f);

    const float mid_y = win_pos.y + height * 0.5f;

    // 1. Left Brand & Concise Instrument Band Label (no neon capsule pills)
    ui_obs::icons::draw_dish(dl, ImVec2(win_pos.x + ui_obs::dp(22.0f), mid_y),
                             ui_obs::dp(9.0f), ui_obs::palette::U32(ui_obs::palette::Cyan));
    {
        ui_obs::ScopedFont title_f(fonts_.semibold, 15.0f);
        ImGui::SetCursorPos(ImVec2(ui_obs::dp(38.0f), (height - ImGui::GetTextLineHeight()) * 0.5f));
        ImGui::TextColored(ui_obs::palette::TextPrimary, "HYDROGEN");
    }

    ImGui::SameLine(0.0f, ui_obs::dp(10.0f));
    float brand_right_local = ImGui::GetItemRectMax().x - win_pos.x;
    {
        ui_obs::ScopedFont tag_f(fonts_.mono, 12.0f);
        ImGui::SetCursorPosY((height - ImGui::GetTextLineHeight()) * 0.5f);
        ImGui::TextColored(ui_obs::palette::TextMuted, "21cm H I");
        brand_right_local = ImGui::GetItemRectMax().x - win_pos.x;
    }

    const float div_x = win_pos.x + brand_right_local + ui_obs::dp(14.0f);
    dl->AddLineV(div_x, win_pos.y + ui_obs::dp(11.0f), win_pos.y + height - ui_obs::dp(11.0f),
                 ui_obs::palette::U32(ui_obs::palette::BorderSubtle), 1.0f);

    // 2. Right Global Status Indicator (computed first & always drawn so it never disappears)
    const bool is_rec = state_.is_recording_active();
    const bool is_ana = analyze_panel_.is_analyzing();
    const bool is_live = sdr_.is_streaming();
    const bool compact_status = (width < ui_obs::dp(760.0f));

    const char* status_label = "IDLE";
    ImVec4 status_col = ui_obs::palette::TextMuted;
    if (is_rec) {
        status_label = "REC ACTIVE";
        status_col = ui_obs::palette::Red;
    } else if (is_ana) {
        status_label = "ANALYZING";
        status_col = ui_obs::palette::Cyan;
    } else if (is_live) {
        status_label = sdr_.is_simulated()
            ? (compact_status ? "SIM RX" : "SIMULATED RX")
            : (compact_status ? "LIVE"   : "RTL-SDR LIVE");
        status_col = sdr_.is_simulated() ? ui_obs::palette::Amber : ui_obs::palette::Emerald;
    }

    float right_limit = win_pos.x + width - ui_obs::dp(14.0f);
    {
        ui_obs::ScopedFont st_f(fonts_.mono, 12.0f);
        const ImVec2 tsz = ImGui::CalcTextSize(status_label);
        const float dot_r = ui_obs::dp(3.5f);
        const float right_x = win_pos.x + width - ui_obs::dp(14.0f);
        const float text_x = right_x - tsz.x;
        const float dot_x = text_x - dot_r - ui_obs::dp(7.0f);
        right_limit = dot_x - dot_r - ui_obs::dp(12.0f);
        dl->AddCircleFilled(ImVec2(dot_x, mid_y), dot_r, ui_obs::palette::U32(status_col), 12);
        dl->AddText(ImVec2(text_x, mid_y - tsz.y * 0.5f - 0.5f),
                    ui_obs::palette::U32(ui_obs::palette::TextPrimary), status_label);
    }

    // 3. Center Animated Segmented Mode Switcher
    const AppSection active_sec = state_.get_section();
    const int active_tab = (active_sec == AppSection::Record) ? 0 : 1;
    const bool compact_tabs = (width < ui_obs::dp(880.0f));

    const float seg_h = ui_obs::dp(34.0f);
    const float icon_slot = ui_obs::dp(22.0f);
    const float pad_side = compact_tabs ? ui_obs::dp(10.0f) : ui_obs::dp(14.0f);
    const char* label_rec = compact_tabs ? "1. Live SDR"     : "1. Live SDR & Record";
    const char* label_ana = compact_tabs ? "2. H I Analysis" : "2. H I Scientific Analysis";
    float w_rec = 0.0f, w_ana = 0.0f;
    {
        ui_obs::ScopedFont seg_f(fonts_.semibold, 14.5f);
        w_rec = icon_slot + ImGui::CalcTextSize(label_rec).x + pad_side * 2.0f;
        w_ana = icon_slot + ImGui::CalcTextSize(label_ana).x + pad_side * 2.0f;
    }
    const float gap = ui_obs::dp(4.0f);
    const float inner_pad = ui_obs::dp(3.0f);
    const float trough_w = w_rec + w_ana + gap + inner_pad * 2.0f;
    const float min_tabs_x = brand_right_local + ui_obs::dp(18.0f);
    const float max_tabs_x = std::max(min_tabs_x, (right_limit - win_pos.x) - trough_w);
    const float tabs_x = std::clamp((width - trough_w) * 0.5f, min_tabs_x, max_tabs_x);

    const ImVec2 tr_min(win_pos.x + tabs_x, win_pos.y + (height - seg_h) * 0.5f);
    const ImVec2 tr_max(tr_min.x + trough_w, tr_min.y + seg_h);
    dl->AddRectFilled(tr_min, tr_max, ui_obs::palette::U32(ui_obs::palette::BgWell), ui_obs::dp(8.0f), ImDrawFlags_RoundAll);
    dl->AddRect(tr_min, tr_max, ui_obs::palette::U32(ui_obs::palette::BorderSubtle), ui_obs::dp(8.0f), 1.0f, ImDrawFlags_RoundAll);

    const float slot0_x = tr_min.x + inner_pad;
    const float slot1_x = slot0_x + w_rec + gap;
    const float target_x = (active_tab == 0) ? slot0_x : slot1_x;
    const float target_w = (active_tab == 0) ? w_rec : w_ana;

    ImGuiStorage* st = ImGui::GetStateStorage();
    float* anim_x = st->GetFloatRef(ImGui::GetID("##SegAnimX"), target_x);
    float* anim_w = st->GetFloatRef(ImGui::GetID("##SegAnimW"), target_w);
    // Re-anchor smoothly if window resized dramatically
    if (std::abs(*anim_x - target_x) > trough_w) {
        *anim_x = target_x;
        *anim_w = target_w;
    } else {
        const float k = std::clamp(1.0f - std::exp(-16.0f * ImGui::GetIO().DeltaTime), 0.0f, 1.0f);
        *anim_x += (target_x - *anim_x) * k;
        *anim_w += (target_w - *anim_w) * k;
    }
    *anim_x = std::clamp(*anim_x, slot0_x, slot1_x);

    const ImVec2 pill_min(*anim_x, tr_min.y + inner_pad);
    const ImVec2 pill_max(*anim_x + *anim_w, tr_max.y - inner_pad);
    dl->AddRectFilled(pill_min, pill_max, ui_obs::palette::U32(ui_obs::palette::SurfaceAlt), ui_obs::dp(6.0f), ImDrawFlags_RoundAll);
    dl->AddRect(pill_min, pill_max, ui_obs::palette::U32(ui_obs::palette::Border), ui_obs::dp(6.0f), 1.0f, ImDrawFlags_RoundAll);
    dl->AddRectFilled(ImVec2(pill_min.x + ui_obs::dp(12.0f), pill_max.y - ui_obs::dp(2.5f)),
                      ImVec2(pill_max.x - ui_obs::dp(12.0f), pill_max.y),
                      ui_obs::palette::U32(ui_obs::palette::AccentBlue), ui_obs::dp(1.5f), ImDrawFlags_RoundAll);

    auto draw_seg_slot = [&](const char* btn_id, const char* label, float abs_x, float slot_w,
                             bool selected, bool is_rec_tab) -> bool {
        const float slot_h = seg_h - inner_pad * 2.0f;
        ImGui::SetCursorScreenPos(ImVec2(abs_x, tr_min.y + inner_pad));
        const bool clicked = ImGui::InvisibleButton(btn_id, ImVec2(slot_w, slot_h));
        const bool hov = ImGui::IsItemHovered();
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

        if (hov && !selected) {
            dl->AddRectFilled(ImVec2(abs_x, tr_min.y + inner_pad),
                              ImVec2(abs_x + slot_w, tr_max.y - inner_pad),
                              ui_obs::palette::U32(ui_obs::palette::SurfaceHover, 0.45f),
                              ui_obs::dp(6.0f), ImDrawFlags_RoundAll);
        }

        ui_obs::ScopedFont seg_f(fonts_.semibold, 14.5f);
        const ImU32 fg = selected ? ui_obs::palette::U32(ui_obs::palette::TextPrimary)
                                  : ui_obs::palette::U32(hov ? ui_obs::palette::TextPrimary : ui_obs::palette::TextMuted);
        const ImVec2 ts = ImGui::CalcTextSize(label);
        const float content_w = icon_slot + ts.x;
        const float cx = abs_x + (slot_w - content_w) * 0.5f;
        const float cy = tr_min.y + seg_h * 0.5f;

        if (is_rec_tab) {
            ui_obs::icons::draw_dish(dl, ImVec2(cx + ui_obs::dp(8.0f), cy), ui_obs::dp(7.0f), fg);
        } else {
            ui_obs::icons::draw_spectrum_peak(dl, ImVec2(cx + ui_obs::dp(8.0f), cy), ui_obs::dp(7.5f), fg);
        }
        dl->AddText(ImVec2(cx + icon_slot, cy - ts.y * 0.5f - 0.5f), fg, label);
        return clicked;
    };

    if (draw_seg_slot("##TabRec", label_rec, slot0_x, w_rec, active_tab == 0, true)) {
        state_.switch_to_record();
    }
    if (draw_seg_slot("##TabAna", label_ana, slot1_x, w_ana, active_tab == 1, false)) {
        state_.set_section(AppSection::Analyze);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void AppGui::render_status_bar(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::HeaderBg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ui_obs::dp(12.0f), 0.0f));
    ImGui::BeginChild("##ObsStatusBar", ImVec2(width, height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 win_pos = ImGui::GetWindowPos();
    dl->AddLineH(win_pos.x, win_pos.x + width, win_pos.y,
                 ui_obs::palette::U32(ui_obs::palette::BorderSubtle), 1.0f);

    const AppSection sec = state_.get_section();
    const bool is_rec = state_.is_recording_active();
    const bool is_ana = analyze_panel_.is_analyzing();

    const float mid_y = win_pos.y + height * 0.5f;
    auto draw_vdiv = [&](float x) {
        dl->AddLineV(x, win_pos.y + ui_obs::dp(5.0f), win_pos.y + height - ui_obs::dp(5.0f),
                     ui_obs::palette::U32(ui_obs::palette::BorderSubtle), 1.0f);
    };
    auto draw_seg_right = [&](float& rx, const char* txt, const ImVec4& col, bool divider = true) {
        const ImVec2 tsz = ImGui::CalcTextSize(txt);
        rx -= tsz.x;
        dl->AddText(ImVec2(rx, mid_y - tsz.y * 0.5f - 0.5f), ui_obs::palette::U32(col), txt);
        rx -= ui_obs::dp(10.0f);
        if (divider) {
            draw_vdiv(rx);
            rx -= ui_obs::dp(10.0f);
        }
    };

    float right_cursor = win_pos.x + width - ui_obs::dp(12.0f);
    {
        ui_obs::ScopedFont rf(fonts_.mono, 11.5f);

        auto draw_shortcut = [&](const char* btn_id, const char* label, bool active) -> bool {
            const ImVec2 tsz = ImGui::CalcTextSize(label);
            right_cursor -= tsz.x;
            const ImVec2 pos(right_cursor, mid_y - tsz.y * 0.5f - 0.5f);
            ImGui::SetCursorScreenPos(ImVec2(pos.x - ui_obs::dp(3.0f), win_pos.y));
            const bool clicked = ImGui::InvisibleButton(btn_id, ImVec2(tsz.x + ui_obs::dp(6.0f), height));
            const bool hov = ImGui::IsItemHovered();
            if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

            const ImVec4 fg = active ? ui_obs::palette::TextPrimary
                                     : (hov ? ui_obs::palette::Cyan : ui_obs::palette::TextDim);
            dl->AddText(pos, ui_obs::palette::U32(fg), label);
            right_cursor -= ui_obs::dp(10.0f);
            return clicked;
        };

        if (draw_shortcut("##KeyF2", "F2: Analysis", sec == AppSection::Analyze)) {
            state_.set_section(AppSection::Analyze);
        }
        draw_vdiv(right_cursor + ui_obs::dp(4.0f));
        right_cursor -= ui_obs::dp(4.0f);
        if (draw_shortcut("##KeyF1", "F1: Live SDR", sec == AppSection::Record)) {
            state_.switch_to_record();
        }

        if (width > ui_obs::dp(680.0f)) {
            draw_vdiv(right_cursor + ui_obs::dp(4.0f));
            right_cursor -= ui_obs::dp(6.0f);
            char ram_buf[32];
            std::snprintf(ram_buf, sizeof(ram_buf), "RAM %zu MB", poll_process_memory_mb());
            draw_seg_right(right_cursor, ram_buf, ui_obs::palette::TextMuted, width > ui_obs::dp(820.0f));
        }

        if (width > ui_obs::dp(820.0f)) {
            char simd_buf[48];
            std::snprintf(simd_buf, sizeof(simd_buf), "SIMD %s", cached_simd_name_.c_str());
            draw_seg_right(right_cursor, simd_buf, ui_obs::palette::TextMuted, false);
        }
    }

    // Left cluster flat monospace readouts separated by crisp 1px hairlines
    {
        ui_obs::ScopedFont lf(fonts_.mono, 11.5f);
        float cur_x = win_pos.x + ui_obs::dp(12.0f);

        auto draw_seg_left = [&](const char* txt, const ImVec4& col, bool with_dot = false,
                                 const ImVec4& dot_col = ui_obs::palette::TextMuted) {
            const ImVec2 tsz = ImGui::CalcTextSize(txt);
            const float dot_space = with_dot ? ui_obs::dp(12.0f) : 0.0f;
            if (cur_x + dot_space + tsz.x >= right_cursor - ui_obs::dp(12.0f)) return false;
            if (cur_x > win_pos.x + ui_obs::dp(14.0f)) {
                draw_vdiv(cur_x);
                cur_x += ui_obs::dp(10.0f);
            }
            if (with_dot) {
                dl->AddCircleFilled(ImVec2(cur_x + ui_obs::dp(3.0f), mid_y), ui_obs::dp(3.0f),
                                    ui_obs::palette::U32(dot_col), 10);
                cur_x += dot_space;
            }
            dl->AddText(ImVec2(cur_x, mid_y - tsz.y * 0.5f - 0.5f), ui_obs::palette::U32(col), txt);
            cur_x += tsz.x + ui_obs::dp(10.0f);
            return true;
        };

        if (sec == AppSection::Record) {
            const char* mode_lbl = is_rec ? "RECORDING" : (sdr_.is_simulated() ? "SIMULATED" : "HARDWARE");
            const ImVec4 mode_col = is_rec ? ui_obs::palette::Red
                                           : (sdr_.is_simulated() ? ui_obs::palette::Amber : ui_obs::palette::Emerald);
            draw_seg_left(mode_lbl, ui_obs::palette::TextPrimary, true, mode_col);

            char dsp_buf[48];
            std::snprintf(dsp_buf, sizeof(dsp_buf), "DSP %.0f us | %.1f MSPS",
                          dsp_.get_last_latency_us(), dsp_.get_dsp_throughput_msps());
            draw_seg_left(dsp_buf, ui_obs::palette::TextMuted);

            char ring_buf[32];
            const float ring_pct = sdr_.get_ring_buffer_fill_pct();
            std::snprintf(ring_buf, sizeof(ring_buf), "RING %.1f%%", ring_pct);
            draw_seg_left(ring_buf, (ring_pct > 80.0f) ? ui_obs::palette::Amber : ui_obs::palette::TextMuted);

            const uint64_t drops = sdr_.get_dropped_samples_count();
            char drop_buf[32];
            std::snprintf(drop_buf, sizeof(drop_buf), "DROPS %llu", static_cast<unsigned long long>(drops));
            draw_seg_left(drop_buf, drops > 0 ? ui_obs::palette::Red : ui_obs::palette::TextMuted);
        } else {
            const char* mode_lbl = is_ana ? "PROCESSING" : "READY";
            const ImVec4 mode_col = is_ana ? ui_obs::palette::Cyan : ui_obs::palette::Emerald;
            draw_seg_left(mode_lbl, ui_obs::palette::TextPrimary, true, mode_col);
            draw_seg_left("Wheel: Zoom X | Shift+Wheel: Y | R-Drag: Box Zoom | L-Drag: Pan | Dbl-Click/R: Reset", ui_obs::palette::TextMuted);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void AppGui::render_frame(ID3D11DeviceContext* context, float display_w, float display_h) {
    if (display_w < 100.0f || display_h < 100.0f) return;

    // Keep global REC ACTIVE badge and collision guards synchronized across tabs
    record_panel_.sync_recording_state();

    // Keyboard shortcuts when not editing a text box
    if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_F1)) {
            state_.switch_to_record();
        } else if (ImGui::IsKeyPressed(ImGuiKey_F2)) {
            state_.set_section(AppSection::Analyze);
        }
    }

    // Consume pending [Analyze This Capture] bridge trigger
    std::string target_path;
    if (state_.check_and_clear_auto_analyze(target_path)) {
        analyze_panel_.select_and_run_file(target_path, true);
    }

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(display_w, display_h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGuiWindowFlags root_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                  ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar |
                                  ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::Begin("##ObsRootWindow", nullptr, root_flags);

    const float nav_h = ui_obs::dp(44.0f);
    const float status_h = ui_obs::dp(24.0f);
    const float body_h = std::max(80.0f, display_h - nav_h - status_h);

    render_top_navigation_bar(display_w, nav_h);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui_obs::palette::BgDeep);
    ImGui::BeginChild("##ObsWorkspace", ImVec2(display_w, body_h), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (state_.get_section() == AppSection::Record) {
        record_panel_.render_panel(context, display_w, body_h);
    } else {
        analyze_panel_.render_panel(context, display_w, body_h);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    render_status_bar(display_w, status_h);

    ImGui::End();
    ImGui::PopStyleVar(3);
}
