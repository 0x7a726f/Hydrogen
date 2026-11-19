#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>

#include "imgui.h"
#include "constants.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ui_obs {

// ======================================================================
// Observatory Slate Palette, DPI Helpers & Font Registry (ImGui 1.93.1+)
// ======================================================================
struct UiFonts {
    ImFont* regular  = nullptr;
    ImFont* semibold = nullptr;
    ImFont* mono     = nullptr;
};

inline float dp(float px) {
    const float s = ImGui::GetStyle().FontScaleDpi;
    return px * (s > 0.0f ? s : 1.0f);
}

// Rounds scaled pixel size to an exact integer so ImGui 1.93's ImFont::RenderText
// uses scale = 1.0f against the integer-baked glyph atlas (eliminating fractional quad resampling blur).
inline float font_dp(float px) {
    return std::round(dp(px));
}

// unscaled_size < 0 preserves font's AddFont* LegacySize; == 0 keeps current FontSizeBase; > 0 sets explicit size.
inline void PushFontSafe(ImFont* font, float unscaled_size = -1.0f) {
    ImFont* target = font ? font : ImGui::GetFont();
    float sz = unscaled_size;
    if (sz < 0.0f) {
        sz = (font && font->LegacySize > 0.0f) ? std::round(font->LegacySize) : 0.0f;
    } else if (sz > 0.0f) {
        sz = std::round(sz);
    }
    ImGui::PushFont(target, sz);
}

struct ScopedFont {
    explicit ScopedFont(ImFont* font, float unscaled_size = -1.0f) {
        PushFontSafe(font, unscaled_size);
    }
    ~ScopedFont() {
        ImGui::PopFont();
    }
    ScopedFont(const ScopedFont&) = delete;
    ScopedFont& operator=(const ScopedFont&) = delete;
};

namespace palette {
    // 4-tier luminance hierarchy
    constexpr ImVec4 BgWell        = ImVec4(0.035f, 0.047f, 0.067f, 1.0f);                                              // #090C11 (plot wells, troughs, inputs)
    constexpr ImVec4 BgDeep        = ImVec4(astro::theme::BG_R, astro::theme::BG_G, astro::theme::BG_B, 1.0f);          // #0D1117 (workspace gutter)
    constexpr ImVec4 HeaderBg      = ImVec4(0.059f, 0.078f, 0.106f, 1.0f);                                              // #0F141B (nav + status chrome)
    constexpr ImVec4 Surface       = ImVec4(astro::theme::FRAME_R, astro::theme::FRAME_G, astro::theme::FRAME_B, 1.0f); // #141A23 (elevated card surfaces)
    constexpr ImVec4 SurfaceAlt    = ImVec4(0.106f, 0.137f, 0.184f, 1.0f);                                              // #1B232F (buttons, active pills)
    constexpr ImVec4 SurfaceHover  = ImVec4(0.141f, 0.184f, 0.247f, 1.0f);                                              // #242F3F
    constexpr ImVec4 SurfaceActive = ImVec4(0.173f, 0.227f, 0.306f, 1.0f);                                              // #2C3A4E

    // Soft structural borders & inner specular highlight
    constexpr ImVec4 BorderSubtle  = ImVec4(0.55f, 0.64f, 0.78f, 0.14f);
    constexpr ImVec4 Border        = ImVec4(0.55f, 0.64f, 0.78f, 0.26f);
    constexpr ImVec4 HairlineLight = ImVec4(1.00f, 1.00f, 1.00f, 0.05f);

    constexpr ImVec4 TextPrimary   = ImVec4(0.902f, 0.929f, 0.953f, 1.0f); // #E6EDF3
    constexpr ImVec4 TextMuted     = ImVec4(astro::theme::MUTED_R, astro::theme::MUTED_G, astro::theme::MUTED_B, 1.0f); // #8B949E
    constexpr ImVec4 TextDim       = ImVec4(0.388f, 0.439f, 0.510f, 1.0f); // #637082

    constexpr ImVec4 AccentBlue    = ImVec4(astro::theme::ACCENT_R, astro::theme::ACCENT_G, astro::theme::ACCENT_B, 1.0f); // #2F81F7
    constexpr ImVec4 Cyan          = ImVec4(astro::theme::CYAN_R, astro::theme::CYAN_G, astro::theme::CYAN_B, 1.0f);       // #38BDF8
    constexpr ImVec4 Emerald       = ImVec4(astro::theme::GREEN_R, astro::theme::GREEN_G, astro::theme::GREEN_B, 1.0f);    // #10B981
    constexpr ImVec4 Red           = ImVec4(astro::theme::RED_R, astro::theme::RED_G, astro::theme::RED_B, 1.0f);          // #EF4444
    constexpr ImVec4 Amber         = ImVec4(astro::theme::AMBER_R, astro::theme::AMBER_G, astro::theme::AMBER_B, 1.0f);    // #F59E0B
    constexpr ImVec4 Orange        = ImVec4(astro::theme::ORANGE_R, astro::theme::ORANGE_G, astro::theme::ORANGE_B, 1.0f); // #FB923C
    constexpr ImVec4 Pink          = ImVec4(astro::theme::PINK_R, astro::theme::PINK_G, astro::theme::PINK_B, 1.0f);       // #F472B6
    constexpr ImVec4 Purple        = ImVec4(0.655f, 0.545f, 0.980f, 1.0f);                                                 // #A78BFA

    inline ImVec4 Tint(const ImVec4& c, float alpha) {
        return ImVec4(c.x, c.y, c.z, alpha);
    }

    // Alpha-aware U32 converter respecting ImGui::BeginDisabled() / style.Alpha
    inline ImU32 U32(const ImVec4& c, float alpha_mul = 1.0f) {
        ImVec4 out = c;
        out.w *= alpha_mul;
        return ImGui::GetColorU32(out);
    }

    inline constexpr ImU32 WithAlpha(ImU32 c, uint8_t a) {
        return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a) << IM_COL32_A_SHIFT);
    }

    inline constexpr ImU32 WithAlphaF(ImU32 c, float alpha01) {
        const float a = alpha01 < 0.0f ? 0.0f : (alpha01 > 1.0f ? 1.0f : alpha01);
        return WithAlpha(c, static_cast<uint8_t>(a * 255.0f + 0.5f));
    }

    inline constexpr ImU32 LerpU32(ImU32 c0, ImU32 c1, float t) {
        const float s = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        const auto lerp_ch = [s](uint32_t a, uint32_t b) -> uint8_t {
            return static_cast<uint8_t>(a + (static_cast<float>(b) - static_cast<float>(a)) * s + 0.5f);
        };
        return IM_COL32(
            lerp_ch((c0 >> 0) & 0xFFu,  (c1 >> 0) & 0xFFu),
            lerp_ch((c0 >> 8) & 0xFFu,  (c1 >> 8) & 0xFFu),
            lerp_ch((c0 >> 16) & 0xFFu, (c1 >> 16) & 0xFFu),
            lerp_ch((c0 >> 24) & 0xFFu, (c1 >> 24) & 0xFFu)
        );
    }

    constexpr ImU32 PlotBg32       = IM_COL32(9, 12, 17, 255);
    constexpr ImU32 PlotGrid32     = IM_COL32(28, 37, 50, 255);
    constexpr ImU32 PlotBorder32   = IM_COL32(43, 55, 73, 255);
    constexpr ImU32 PlotCrosshair  = IM_COL32(139, 148, 158, 145);
    constexpr ImU32 CardBg32       = IM_COL32(15, 20, 28, 238);
    constexpr ImU32 PillBg32       = IM_COL32(13, 18, 26, 230);
    constexpr ImU32 BezelBg32      = IM_COL32(9, 13, 19, 255);
    constexpr ImU32 BezelBorder32  = IM_COL32(48, 60, 78, 255);
    constexpr ImU32 TextPrimary32  = IM_COL32(230, 237, 243, 255);
    constexpr ImU32 TextMuted32    = IM_COL32(139, 148, 158, 255);
} // namespace palette

inline void apply_observatory_slate_style(float dpi_scale = 0.0f) {
    ImGuiStyle& style = ImGui::GetStyle();
    const float target_dpi = (dpi_scale > 0.0f) ? dpi_scale
                           : (style.FontScaleDpi > 0.0f ? style.FontScaleDpi : 1.0f);
    style = ImGuiStyle{};

    style.FontSizeBase        = 15.0f;
    style.FontScaleDpi        = target_dpi;

    style.WindowPadding       = ImVec2(9.0f, 7.0f);
    style.FramePadding        = ImVec2(8.0f, 4.0f);
    style.CellPadding         = ImVec2(5.0f, 3.0f);
    style.ItemSpacing         = ImVec2(6.0f, 5.0f);
    style.ItemInnerSpacing    = ImVec2(5.0f, 4.0f);
    style.ScrollbarSize       = 9.0f;
    style.ScrollbarPadding    = 2.0f;
    style.GrabMinSize         = 10.0f;

    style.WindowBorderSize    = 1.0f;
    style.ChildBorderSize     = 1.0f;
    style.PopupBorderSize     = 1.0f;
    style.FrameBorderSize     = 1.0f;
    style.TabBorderSize       = 0.0f;
    style.TabBarBorderSize    = 1.0f;
    style.TabBarOverlineSize  = 2.0f;
    style.SeparatorSize       = 1.0f;

    style.WindowRounding      = 0.0f;
    style.ChildRounding       = 8.0f;
    style.FrameRounding       = 5.0f;
    style.PopupRounding       = 7.0f;
    style.ScrollbarRounding   = 6.0f;
    style.GrabRounding        = 5.0f;
    style.TabRounding         = 6.0f;
    style.SelectableRounding  = 5.0f;
    style.MenuItemRounding    = 5.0f;
    style.HoverDelayShort     = 0.25f;

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text]                 = palette::TextPrimary;
    c[ImGuiCol_TextDisabled]         = palette::TextDim;
    c[ImGuiCol_WindowBg]             = palette::BgDeep;
    c[ImGuiCol_ChildBg]              = palette::Surface;
    c[ImGuiCol_PopupBg]              = palette::HeaderBg;
    c[ImGuiCol_Border]               = palette::BorderSubtle;
    c[ImGuiCol_BorderShadow]         = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    c[ImGuiCol_FrameBg]              = palette::BgWell;
    c[ImGuiCol_FrameBgHovered]       = palette::SurfaceAlt;
    c[ImGuiCol_FrameBgActive]        = palette::SurfaceHover;

    c[ImGuiCol_TitleBg]              = palette::HeaderBg;
    c[ImGuiCol_TitleBgActive]        = palette::HeaderBg;
    c[ImGuiCol_MenuBarBg]            = palette::HeaderBg;

    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_ScrollbarGrab]        = palette::SurfaceAlt;
    c[ImGuiCol_ScrollbarGrabHovered] = palette::SurfaceHover;
    c[ImGuiCol_ScrollbarGrabActive]  = palette::Border;

    c[ImGuiCol_CheckMark]            = palette::Cyan;
    c[ImGuiCol_CheckboxSelectedBg]   = palette::SurfaceActive;
    c[ImGuiCol_SliderGrab]           = palette::AccentBlue;
    c[ImGuiCol_SliderGrabActive]     = palette::Cyan;

    c[ImGuiCol_Button]               = palette::SurfaceAlt;
    c[ImGuiCol_ButtonHovered]        = palette::SurfaceHover;
    c[ImGuiCol_ButtonActive]         = palette::SurfaceActive;

    c[ImGuiCol_Header]               = palette::Tint(palette::AccentBlue, 0.22f);
    c[ImGuiCol_HeaderHovered]        = palette::Tint(palette::AccentBlue, 0.34f);
    c[ImGuiCol_HeaderActive]         = palette::Tint(palette::AccentBlue, 0.48f);

    c[ImGuiCol_Separator]            = palette::BorderSubtle;
    c[ImGuiCol_SeparatorHovered]     = palette::AccentBlue;
    c[ImGuiCol_SeparatorActive]      = palette::Cyan;

    c[ImGuiCol_Tab]                  = palette::BgWell;
    c[ImGuiCol_TabHovered]           = palette::SurfaceHover;
    c[ImGuiCol_TabSelected]          = palette::SurfaceAlt;
    c[ImGuiCol_TabSelectedOverline]  = palette::AccentBlue;

    c[ImGuiCol_PlotLines]            = palette::Cyan;
    c[ImGuiCol_PlotHistogram]        = palette::AccentBlue;
    c[ImGuiCol_TableHeaderBg]        = palette::SurfaceAlt;
    c[ImGuiCol_TableBorderStrong]    = palette::Border;
    c[ImGuiCol_TableBorderLight]     = palette::BorderSubtle;
    c[ImGuiCol_TableRowBg]           = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TableRowBgAlt]        = ImVec4(1.0f, 1.0f, 1.0f, 0.025f);
    c[ImGuiCol_NavCursor]            = palette::Cyan;
    c[ImGuiCol_ModalWindowDimBg]     = ImVec4(0.02f, 0.03f, 0.05f, 0.72f);

    if (target_dpi != 1.0f) {
        style.ScaleAllSizes(target_dpi);
    }
}

// ======================================================================
// Vector Icon Primitives (zero Unicode/emoji glyph fallback artifacts)
// ======================================================================
namespace icons {

inline void draw_dish(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
    dl->PathClear();
    for (int i = 0; i <= 10; ++i) {
        float a = 3.65f + (static_cast<float>(i) / 10.0f) * 1.9f;
        dl->PathLineTo(ImVec2(c.x + std::cos(a) * r * 0.95f, c.y + std::sin(a) * r * 0.95f));
    }
    dl->PathStroke(col, 1.8f, ImDrawFlags_None);
    dl->AddLine(ImVec2(c.x - r * 0.1f, c.y + r * 0.1f), ImVec2(c.x + r * 0.65f, c.y - r * 0.65f), col, 1.5f);
    dl->AddCircleFilled(ImVec2(c.x + r * 0.65f, c.y - r * 0.65f), 2.2f, col);
    dl->AddLine(ImVec2(c.x - r * 0.15f, c.y + r * 0.25f), ImVec2(c.x - r * 0.5f, c.y + r * 0.85f), col, 1.5f);
    dl->AddLine(ImVec2(c.x - r * 0.15f, c.y + r * 0.25f), ImVec2(c.x + r * 0.25f, c.y + r * 0.85f), col, 1.5f);
}

inline void draw_spectrum_peak(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
    ImVec2 pts[7] = {
        ImVec2(c.x - r,         c.y + r * 0.55f),
        ImVec2(c.x - r * 0.55f, c.y + r * 0.45f),
        ImVec2(c.x - r * 0.25f, c.y - r * 0.15f),
        ImVec2(c.x,             c.y - r * 0.80f),
        ImVec2(c.x + r * 0.25f, c.y - r * 0.10f),
        ImVec2(c.x + r * 0.58f, c.y + r * 0.45f),
        ImVec2(c.x + r,         c.y + r * 0.55f)
    };
    dl->AddPolyline(pts, 7, col, 1.8f, ImDrawFlags_None);
    dl->AddLineH(c.x - r, c.x + r, c.y + r * 0.72f, palette::U32(palette::TextMuted), 1.0f);
}

inline void draw_stop_square(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
    dl->AddRectFilled(ImVec2(c.x - r * 0.58f, c.y - r * 0.58f), ImVec2(c.x + r * 0.58f, c.y + r * 0.58f), col, 1.5f);
}

inline void draw_play_triangle(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
    dl->AddTriangleFilled(
        ImVec2(c.x - r * 0.45f, c.y - r * 0.65f),
        ImVec2(c.x + r * 0.68f, c.y),
        ImVec2(c.x - r * 0.45f, c.y + r * 0.65f),
        col
    );
}

inline void draw_bolt(ImDrawList* dl, ImVec2 c, float r, ImU32 col) {
    ImVec2 pts[4] = {
        ImVec2(c.x + r * 0.15f, c.y - r * 0.85f),
        ImVec2(c.x - r * 0.45f, c.y + r * 0.05f),
        ImVec2(c.x + r * 0.10f, c.y + r * 0.05f),
        ImVec2(c.x - r * 0.25f, c.y + r * 0.85f)
    };
    dl->AddPolyline(pts, 4, col, 1.8f, ImDrawFlags_None);
}

} // namespace icons

// ======================================================================
// Observatory Controls, Badges, Cards & Overlay Helpers
// ======================================================================
namespace widgets {

inline bool ToggleSwitch(const char* label, bool* v, const ImVec4& accent = palette::AccentBlue) {
    ImGuiStyle& style = ImGui::GetStyle();
    const float frame_h = ImGui::GetFrameHeight();
    const float track_h = std::round(ImGui::GetTextLineHeight() * 1.18f);
    const float track_w = std::round(track_h * 1.88f);

    const char* disp_end = nullptr;
    for (const char* p = label; *p; ++p) {
        if (p[0] == '#' && p[1] == '#') {
            disp_end = p;
            break;
        }
    }
    const ImVec2 label_sz = (disp_end != label) ? ImGui::CalcTextSize(label, disp_end) : ImVec2(0.0f, 0.0f);
    const float total_w = track_w + (label_sz.x > 0.0f ? style.ItemInnerSpacing.x + label_sz.x : 0.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();

    const bool pressed = ImGui::InvisibleButton(label, ImVec2(total_w, frame_h));
    if (pressed) *v = !(*v);

    const bool hovered = ImGui::IsItemHovered();
    const bool focused = ImGui::IsItemFocused();
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

    const ImGuiID id = ImGui::GetItemID();
    float* t_ptr = ImGui::GetStateStorage()->GetFloatRef(id, *v ? 1.0f : 0.0f);
    const float alpha_step = 1.0f - std::exp(-18.0f * ImGui::GetIO().DeltaTime);
    *t_ptr += ((*v ? 1.0f : 0.0f) - *t_ptr) * std::clamp(alpha_step, 0.0f, 1.0f);
    const float t = *t_ptr;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 t_min(pos.x, pos.y + (frame_h - track_h) * 0.5f);
    const ImVec2 t_max(t_min.x + track_w, t_min.y + track_h);
    const float radius = track_h * 0.5f;

    auto lerp_v4 = [](const ImVec4& a, const ImVec4& b, float u) {
        return ImVec4(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u, a.w + (b.w - a.w) * u);
    };
    const ImVec4 off_bg = hovered ? palette::SurfaceHover : palette::BgWell;
    const ImVec4 on_bg  = hovered ? lerp_v4(accent, ImVec4(1, 1, 1, 1), 0.14f) : accent;
    const ImVec4 tr_col = lerp_v4(off_bg, on_bg, t);

    dl->AddRectFilled(t_min, t_max, palette::U32(tr_col), radius, ImDrawFlags_RoundAll);
    dl->AddRect(t_min, t_max,
                palette::U32(focused ? palette::Cyan : lerp_v4(palette::Border, accent, t * 0.6f)),
                radius, focused ? 1.5f : 1.0f, ImDrawFlags_RoundAll);

    const float pad = dp(2.5f);
    const float knob_r = std::max(2.0f, radius - pad);
    const float knob_x = t_min.x + radius + t * (track_w - track_h);
    const ImVec2 knob_c(knob_x, t_min.y + radius);
    dl->AddCircleFilled(ImVec2(knob_c.x, knob_c.y + 0.5f), knob_r + 0.5f, ImGui::GetColorU32(IM_COL32(0, 0, 0, 75)), 20);
    dl->AddCircleFilled(knob_c, knob_r, palette::U32(palette::TextPrimary), 20);

    if (label_sz.x > 0.0f) {
        dl->AddText(ImVec2(t_max.x + style.ItemInnerSpacing.x, pos.y + (frame_h - label_sz.y) * 0.5f),
                    ImGui::GetColorU32(ImGuiCol_Text), label, disp_end);
    }
    return pressed;
}

inline void draw_slider_rail(const char* label, float u, const ImVec4& accent) {
    const ImVec2 rmin = ImGui::GetItemRectMin();
    const ImVec2 rmax = ImGui::GetItemRectMax();
    const float box_w = (label[0] == '#' && label[1] == '#') ? (rmax.x - rmin.x) : ImGui::CalcItemWidth();
    if (box_w > 4.0f) {
        const float uc = std::clamp(u, 0.0f, 1.0f);
        const float rail_h = dp(2.0f);
        const float pad_x = dp(4.0f);
        const float fill_x = rmin.x + pad_x + uc * std::max(0.0f, box_w - pad_x * 2.0f);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(rmin.x + pad_x, rmax.y - rail_h - 1.0f),
                          ImVec2(fill_x, rmax.y - 1.0f),
                          palette::U32(accent, 0.78f), rail_h * 0.5f, ImDrawFlags_RoundAll);
    }
}

inline bool SliderFloatAccent(const char* label, float* v, float v_min, float v_max,
                              const char* format = "%.2f", const ImVec4& accent = palette::Cyan,
                              ImGuiSliderFlags flags = 0,
                              bool has_default = false, float default_val = 0.0f) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, palette::BgWell);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, palette::Tint(accent, 0.48f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, palette::Tint(accent, 0.78f));
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, dp(6.0f));
    bool changed = ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    if (has_default && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        const float clamped = std::clamp(default_val, v_min, v_max);
        if (std::abs(*v - clamped) > 1e-6f) {
            *v = clamped;
            changed = true;
        }
    }

    if (v_max > v_min) {
        float u = (*v - v_min) / (v_max - v_min);
        if ((flags & ImGuiSliderFlags_Logarithmic) != 0 && v_min > 0.0f && v_max > v_min && *v > 0.0f) {
            u = std::log(*v / v_min) / std::log(v_max / v_min);
        }
        draw_slider_rail(label, u, accent);
    }
    return changed;
}

inline bool SliderIntAccent(const char* label, int* v, int v_min, int v_max,
                            const char* format = "%d", const ImVec4& accent = palette::Cyan,
                            ImGuiSliderFlags flags = 0,
                            bool has_default = false, int default_val = 0) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, palette::BgWell);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, palette::Tint(accent, 0.48f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, palette::Tint(accent, 0.78f));
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, dp(6.0f));
    bool changed = ImGui::SliderInt(label, v, v_min, v_max, format, flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    if (has_default && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        const int clamped = std::clamp(default_val, v_min, v_max);
        if (*v != clamped) {
            *v = clamped;
            changed = true;
        }
    }

    if (v_max > v_min) {
        draw_slider_rail(label, static_cast<float>(*v - v_min) / static_cast<float>(v_max - v_min), accent);
    }
    return changed;
}

inline void SectionHeader(ImFont* semibold, const char* title, const ImVec4& accent,
                          const char* badge_text = nullptr, const ImVec4& badge_col = palette::TextMuted) {
    (void)accent;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float line_h = ImGui::GetTextLineHeight();

    {
        ScopedFont f(semibold, 12.5f);
        ImGui::TextColored(palette::TextPrimary, "%s", title);
    }

    if (badge_text && badge_text[0] != '\0') {
        ScopedFont bf(semibold, 11.0f);
        const ImVec2 tsz = ImGui::CalcTextSize(badge_text);
        const float tx = start.x + avail_w - tsz.x;
        if (tx > start.x + dp(90.0f)) {
            dl->AddText(ImVec2(tx, start.y + (line_h - tsz.y) * 0.5f), palette::U32(badge_col), badge_text);
        }
    }

    const float y = start.y + line_h + dp(3.0f);
    dl->AddLineH(start.x, start.x + avail_w, y, palette::U32(palette::BorderSubtle), 1.0f);
    ImGui::SetCursorScreenPos(ImVec2(start.x, y + dp(4.0f)));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

inline bool BeginCard(const char* child_id, ImFont* semibold, const char* title,
                      const ImVec4& accent, const char* badge_text = nullptr,
                      const ImVec4& badge_col = palette::TextMuted, ImVec2 size = ImVec2(-1.0f, 0.0f)) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, palette::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, palette::BorderSubtle);
    const ImGuiChildFlags flags = ImGuiChildFlags_Borders | (size.y == 0.0f ? ImGuiChildFlags_AutoResizeY : ImGuiChildFlags_None);
    const bool open = ImGui::BeginChild(child_id, size, flags);
    if (open) {
        if (title && title[0] != '\0') {
            SectionHeader(semibold, title, accent, badge_text, badge_col);
        }
    }
    return open;
}

inline void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

inline void draw_dashed_vline(ImDrawList* dl, float x, float y0, float y1,
                              ImU32 col, float thickness = 1.0f,
                              float dash = 5.0f, float gap = 4.0f) {
    if (y1 <= y0) return;
    const float period = dash + gap;
    for (float y = y0; y < y1; y += period) {
        const float y_end = std::min(y + dash, y1);
        dl->AddLineV(x, y, y_end, col, thickness);
    }
}

inline void draw_dashed_hline(ImDrawList* dl, float x0, float x1, float y,
                              ImU32 col, float thickness = 1.0f,
                              float dash = 5.0f, float gap = 4.0f) {
    if (x1 <= x0) return;
    const float period = dash + gap;
    for (float x = x0; x < x1; x += period) {
        const float x_end = std::min(x + dash, x1);
        dl->AddLineH(x, x_end, y, col, thickness);
    }
}

struct PillPlacement {
    ImVec2 min{0.0f, 0.0f};
    ImVec2 max{0.0f, 0.0f};
};

inline PillPlacement compute_clamped_pill_rect(
    const char* text, ImFont* font, float font_size_unscaled,
    float anchor_x, float top_y,
    ImVec2 bounds_min, ImVec2 bounds_max,
    float pad_x = 7.0f, float pad_y = 2.5f, float dot_space = 0.0f) {
    ImFont* f = font ? font : ImGui::GetFont();
    const float sz = font_dp(font_size_unscaled > 0.0f ? font_size_unscaled : 12.0f);
    const ImVec2 tsz = f->CalcTextSizeA(sz, FLT_MAX, 0.0f, text);
    const float px = dp(pad_x);
    const float py = dp(pad_y);
    const float ds = dp(dot_space);
    const float w = tsz.x + px * 2.0f + ds;
    const float h = tsz.y + py * 2.0f;

    float x0 = std::floor(anchor_x - w * 0.5f);
    x0 = std::clamp(x0, bounds_min.x + dp(4.0f), std::max(bounds_min.x + dp(4.0f), bounds_max.x - dp(4.0f) - w));
    float y0 = std::floor(top_y);
    y0 = std::clamp(y0, bounds_min.y + dp(3.0f), std::max(bounds_min.y + dp(3.0f), bounds_max.y - dp(3.0f) - h));
    return { ImVec2(x0, y0), ImVec2(x0 + w, y0 + h) };
}

inline void draw_pill_badge(
    ImDrawList* dl, const PillPlacement& rect, const char* text,
    ImFont* font, float font_size_unscaled,
    ImU32 bg_col, ImU32 border_col, ImU32 text_col,
    ImU32 dot_col = 0, float pad_x = 7.0f, float pad_y = 2.5f) {
    ImFont* f = font ? font : ImGui::GetFont();
    const float sz = font_dp(font_size_unscaled > 0.0f ? font_size_unscaled : 12.0f);
    dl->AddRectFilled(rect.min, rect.max, bg_col, dp(4.0f), ImDrawFlags_RoundAll);
    dl->AddRect(rect.min, rect.max, border_col, dp(4.0f), 1.0f, ImDrawFlags_RoundAll);

    float tx = rect.min.x + dp(pad_x);
    if (dot_col != 0) {
        const float cy = (rect.min.y + rect.max.y) * 0.5f;
        dl->AddCircleFilled(ImVec2(tx + dp(3.0f), cy), dp(2.8f), dot_col, 12);
        tx += dp(9.0f);
    }
    dl->AddText(f, sz, ImVec2(std::floor(tx), std::floor(rect.min.y + dp(pad_y))), text_col, text);
}

struct TelemetryRow {
    const char* label;
    const char* value;
    ImU32 value_col;
};

inline void draw_crosshair_card(
    ImDrawList* dl, ImVec2 cursor_px, ImVec2 bounds_min, ImVec2 bounds_max,
    const char* header_title, ImU32 accent_col,
    const TelemetryRow* rows, int num_rows,
    ImFont* ui_font, ImFont* mono_font) {
    if (!rows || num_rows <= 0) return;

    ImFont* uf = ui_font ? ui_font : ImGui::GetFont();
    ImFont* mf = mono_font ? mono_font : uf;
    const float ufs = font_dp(12.0f);
    const float mfs = font_dp(13.0f);
    const float hfs = font_dp(12.0f);

    float max_lbl_w = 0.0f;
    float max_val_w = 0.0f;
    for (int i = 0; i < num_rows; ++i) {
        max_lbl_w = std::max(max_lbl_w, uf->CalcTextSizeA(ufs, FLT_MAX, 0.0f, rows[i].label).x);
        max_val_w = std::max(max_val_w, mf->CalcTextSizeA(mfs, FLT_MAX, 0.0f, rows[i].value).x);
    }
    const ImVec2 hsz = (header_title && header_title[0]) ? uf->CalcTextSizeA(hfs, FLT_MAX, 0.0f, header_title) : ImVec2(0, 0);

    const float pad_x = dp(10.0f);
    const float pad_y = dp(7.0f);
    const float row_h = dp(16.5f);
    const float header_h = (header_title && header_title[0]) ? dp(18.0f) : 0.0f;
    const float card_w = std::max(hsz.x + pad_x * 2.0f + dp(8.0f), max_lbl_w + max_val_w + pad_x * 2.0f + dp(14.0f));
    const float card_h = pad_y * 2.0f + header_h + static_cast<float>(num_rows) * row_h;

    const float offset = dp(14.0f);
    float cx = cursor_px.x + offset;
    float cy = cursor_px.y + offset;
    if (cx + card_w > bounds_max.x - dp(6.0f)) cx = cursor_px.x - offset - card_w;
    if (cy + card_h > bounds_max.y - dp(6.0f)) cy = cursor_px.y - offset - card_h;
    cx = std::clamp(cx, bounds_min.x + dp(4.0f), std::max(bounds_min.x + dp(4.0f), bounds_max.x - dp(4.0f) - card_w));
    cy = std::clamp(cy, bounds_min.y + dp(4.0f), std::max(bounds_min.y + dp(4.0f), bounds_max.y - dp(4.0f) - card_h));

    const ImVec2 cmin(std::floor(cx), std::floor(cy));
    const ImVec2 cmax(std::floor(cx + card_w), std::floor(cy + card_h));

    dl->AddRectFilled(ImVec2(cmin.x + dp(2.0f), cmin.y + dp(3.0f)),
                      ImVec2(cmax.x + dp(2.0f), cmax.y + dp(3.0f)),
                      IM_COL32(0, 0, 0, 95), dp(4.0f), ImDrawFlags_RoundAll);
    dl->AddRectFilled(cmin, cmax, palette::CardBg32, dp(4.0f), ImDrawFlags_RoundAll);
    dl->AddRect(cmin, cmax, palette::BezelBorder32, dp(4.0f), 1.0f, ImDrawFlags_RoundAll);

    float y = cmin.y + pad_y;
    if (header_h > 0.0f) {
        dl->AddText(uf, hfs, ImVec2(cmin.x + pad_x, y), accent_col, header_title);
        dl->AddLineH(cmin.x + pad_x, cmax.x - pad_x, y + dp(14.0f),
                     palette::WithAlphaF(palette::PlotBorder32, 0.9f), 1.0f);
        y += header_h;
    }

    for (int i = 0; i < num_rows; ++i) {
        const ImVec2 vsz = mf->CalcTextSizeA(mfs, FLT_MAX, 0.0f, rows[i].value);
        dl->AddText(uf, ufs, ImVec2(cmin.x + pad_x, y + 1.0f), palette::TextMuted32, rows[i].label);
        dl->AddText(mf, mfs, ImVec2(cmax.x - pad_x - vsz.x, y), rows[i].value_col, rows[i].value);
        y += row_h;
    }
}

} // namespace widgets

// ======================================================================
// Win32 Path & Native File Dialog Helpers
// ======================================================================
namespace win_fs {

inline std::string wide_to_utf8(const wchar_t* wstr) {
    if (!wstr || !*wstr) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return "";
    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &out[0], len, nullptr, nullptr);
    return out;
}

inline std::wstring utf8_to_wide(const std::string& str) {
    if (str.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (len <= 1) return L"";
    std::wstring out(static_cast<size_t>(len - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &out[0], len);
    return out;
}

inline std::filesystem::path get_exe_dir() {
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::filesystem::current_path();
    return std::filesystem::path(buf).parent_path();
}

inline std::string get_default_recordings_dir() {
    std::filesystem::path p = get_exe_dir() / "recordings";
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p.string();
}

inline bool browse_for_cs16_file(HWND owner, const std::string& initial_dir, std::string& out_path) {
    wchar_t file_buf[1024] = {};
    std::wstring init_w = utf8_to_wide(initial_dir);

    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = L"IQ Recordings (*.cs16)\0*.cs16\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = file_buf;
    ofn.nMaxFile = sizeof(file_buf) / sizeof(wchar_t);
    ofn.lpstrInitialDir = init_w.empty() ? nullptr : init_w.c_str();
    ofn.lpstrTitle = L"Select .cs16 I/Q Recording";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameW(&ofn)) {
        out_path = wide_to_utf8(file_buf);
        return !out_path.empty();
    }
    return false;
}

inline void open_folder_in_explorer(const std::string& dir) {
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::wstring wdir = utf8_to_wide(dir);
    ShellExecuteW(nullptr, L"open", wdir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

} // namespace win_fs

// ======================================================================
// Shared Scientific 2D Plot Viewport, Zoom/Pan, Axis Ticks & Decimator
// ======================================================================
struct PlotMargin {
    float left   = 58.0f;
    float right  = 18.0f;
    float top    = 22.0f;
    float bottom = 36.0f;
};

struct PlotZoomState {
    bool   x_active = false;
    bool   y_manual = false;
    bool   active = false; // Compatibility alias kept in sync with is_zoomed()
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    // Rubber-band 2D box-zoom transient state
    bool   box_dragging = false;
    ImVec2 box_start_px{0.0f, 0.0f};
    ImVec2 box_cur_px{0.0f, 0.0f};

    // Internal drag state: 0=none, 1=pan_plot, 2=pan_x, 3=pan_y, 4=box_right, 5=box_shift_left, 6=rclick_axis
    int    drag_mode = 0;
    ImVec2 rclick_press_px{0.0f, 0.0f};

    bool is_zoomed() const { return x_active || y_manual; }

    void sync_active() { active = (x_active || y_manual); }

    void reset() {
        x_active = false;
        y_manual = false;
        active = false;
        box_dragging = false;
        drag_mode = 0;
    }

    void auto_y() {
        y_manual = false;
        sync_active();
    }
};

inline double calc_nice_step(double span, int target_ticks) {
    if (span <= 0.0 || target_ticks < 2) return 1.0;
    const double raw = span / static_cast<double>(target_ticks - 1);
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double nice = (norm < 1.5) ? 1.0 : (norm < 3.0) ? 2.0 : (norm < 7.0) ? 5.0 : 10.0;
    return nice * mag;
}

inline int calc_tick_decimals(double step, int min_decimals = 0, int max_decimals = 5) {
    if (!(step > 0.0) || !std::isfinite(step)) return std::clamp(min_decimals, 0, max_decimals);
    const double lg = std::log10(step);
    int dec = (lg < 0.0) ? static_cast<int>(std::ceil(-lg - 1e-9)) : 0;
    const double scaled = step * std::pow(10.0, static_cast<double>(dec));
    if (dec < max_decimals && std::abs(scaled - std::round(scaled)) > 1e-3) {
        ++dec;
    }
    return std::clamp(dec, min_decimals, max_decimals);
}

inline void format_adaptive_tick(double val, double step, char* buf, size_t sz,
                                 bool force_sign = false, int min_decimals = 0) {
    if (!buf || sz == 0) return;
    if (!std::isfinite(val)) {
        std::snprintf(buf, sz, "0");
        return;
    }
    const double abs_step = std::abs(step);
    if (std::abs(val) < std::max(abs_step * 1e-6, 1e-12)) val = 0.0;
    const int dec = calc_tick_decimals(abs_step, min_decimals, 5);
    if (force_sign) {
        std::snprintf(buf, sz, "%+.*f", dec, val);
    } else {
        std::snprintf(buf, sz, "%.*f", dec, val);
    }
}

inline void parse_tick_fmt_hint(const char* fmt, int& out_min_dec, bool& out_sign, const char*& out_suffix) {
    out_min_dec = 0;
    out_sign = false;
    out_suffix = "";
    if (!fmt) return;
    const char* p = std::strchr(fmt, '%');
    if (!p) return;
    ++p;
    while (*p == '+' || *p == '-' || *p == ' ' || *p == '#' || *p == '0') {
        if (*p == '+') out_sign = true;
        ++p;
    }
    while (*p >= '0' && *p <= '9') ++p;
    if (*p == '.') {
        ++p;
        int d = 0;
        while (*p >= '0' && *p <= '9') {
            d = d * 10 + (*p - '0');
            ++p;
        }
        out_min_dec = std::clamp(d, 0, 5);
    }
    if (*p == 'f' || *p == 'F' || *p == 'g' || *p == 'G') {
        ++p;
        out_suffix = p;
    }
}

struct PlotInteractConfig {
    // Default viewport domain (e.g. [-200, +200] km/s)
    double def_xmin = 0.0, def_xmax = 1.0;
    double def_ymin = 0.0, def_ymax = 1.0;

    // Full data domain clamp (e.g. full passband)
    double bound_xmin = 0.0, bound_xmax = 1.0;
    double bound_ymin = -1e9, bound_ymax = 1e9;

    bool   clamp_x = true;
    bool   clamp_y = false;
    bool   y_inverted = false;       // true for top-down waterfalls (0 at top, max at bottom)
    bool   allow_y_zoom = true;
    double min_x_span_frac = 0.0005; // up to 2000x zoom on X
    double min_y_span_frac = 0.005;  // up to 200x zoom on Y

    const char* x_unit = "";
    const char* y_unit = "";

    bool   has_hi_focus = false;
    double hi_xmin = 0.0, hi_xmax = 0.0;

    bool   show_full_btn = true;
    bool   is_compact = false;
    bool   ignore_ctrl_left_drag = false; // allows Ctrl+LeftClick passthrough (e.g. Live SDR tune)
    const char* auto_y_label = "Auto Y";
    const char* popup_id = nullptr;
};

struct PlotInteractResult {
    bool hovered_plot = false;
    bool hovered_any = false;
    bool over_toolbar = false;
    bool changed = false;
    bool x_changed = false;
    bool y_changed = false;
    bool box_active = false;
    bool context_open = false;
    bool context_opened_this_frame = false;
    bool requested_auto_y = false;
    bool requested_reset = false;
    bool requested_hi_focus = false;
    bool requested_full = false;
    ImVec2 toolbar_min{0.0f, 0.0f};
    ImVec2 toolbar_max{0.0f, 0.0f};
};

namespace detail {

inline void plot_apply_zoom_x(PlotZoomState& zoom, const PlotInteractConfig& cfg,
                              double factor, double anchor_u) {
    const double cur_xmin = zoom.x_active ? zoom.x_min : cfg.def_xmin;
    const double cur_xmax = zoom.x_active ? zoom.x_max : cfg.def_xmax;
    const double cur_span = std::max(1e-12, cur_xmax - cur_xmin);
    const double u = std::clamp(anchor_u, 0.0, 1.0);
    const double ax = cur_xmin + u * cur_span;

    const double b_span = std::max(1e-9, cfg.bound_xmax - cfg.bound_xmin);
    const double min_span = b_span * std::clamp(cfg.min_x_span_frac, 1e-6, 0.5);
    const double max_span = cfg.clamp_x ? b_span : (b_span * 4.0);
    const double new_span = std::clamp(cur_span * factor, min_span, max_span);

    double nx0 = ax - u * new_span;
    double nx1 = nx0 + new_span;
    if (cfg.clamp_x) {
        if (nx0 < cfg.bound_xmin) { nx1 += (cfg.bound_xmin - nx0); nx0 = cfg.bound_xmin; }
        if (nx1 > cfg.bound_xmax) { nx0 -= (nx1 - cfg.bound_xmax); nx1 = cfg.bound_xmax; }
        nx0 = std::max(cfg.bound_xmin, nx0);
        nx1 = std::min(cfg.bound_xmax, nx1);
    }

    const bool def_is_bound = (std::abs(cfg.def_xmin - cfg.bound_xmin) <= b_span * 1e-5) &&
                              (std::abs(cfg.def_xmax - cfg.bound_xmax) <= b_span * 1e-5);
    if (cfg.clamp_x && def_is_bound && (nx1 - nx0) >= b_span * 0.9995) {
        zoom.x_active = false;
        zoom.x_min = cfg.def_xmin;
        zoom.x_max = cfg.def_xmax;
    } else {
        zoom.x_min = nx0;
        zoom.x_max = nx1;
        zoom.x_active = true;
    }
    zoom.sync_active();
}

inline void plot_apply_zoom_y(PlotZoomState& zoom, const PlotInteractConfig& cfg,
                              double factor, double anchor_v) {
    if (!cfg.allow_y_zoom) return;
    const double cur_ymin = zoom.y_manual ? zoom.y_min : cfg.def_ymin;
    const double cur_ymax = zoom.y_manual ? zoom.y_max : cfg.def_ymax;
    const double cur_span = std::max(1e-12, cur_ymax - cur_ymin);
    const double v = std::clamp(anchor_v, 0.0, 1.0);
    const double ay = cur_ymin + v * cur_span;

    const double ref_span = cfg.clamp_y
        ? std::max(1e-9, cfg.bound_ymax - cfg.bound_ymin)
        : std::max(1e-9, cfg.def_ymax - cfg.def_ymin);
    const double min_span = ref_span * std::clamp(cfg.min_y_span_frac, 1e-6, 0.5);
    const double max_span = cfg.clamp_y ? ref_span : (ref_span * 25.0);
    const double new_span = std::clamp(cur_span * factor, min_span, max_span);

    double ny0 = ay - v * new_span;
    double ny1 = ny0 + new_span;
    if (cfg.clamp_y) {
        if (ny0 < cfg.bound_ymin) { ny1 += (cfg.bound_ymin - ny0); ny0 = cfg.bound_ymin; }
        if (ny1 > cfg.bound_ymax) { ny0 -= (ny1 - cfg.bound_ymax); ny1 = cfg.bound_ymax; }
        ny0 = std::max(cfg.bound_ymin, ny0);
        ny1 = std::min(cfg.bound_ymax, ny1);
    }

    if (cfg.clamp_y && (ny1 - ny0) >= ref_span * 0.9995) {
        zoom.y_manual = false;
        zoom.y_min = cfg.def_ymin;
        zoom.y_max = cfg.def_ymax;
    } else {
        zoom.y_min = ny0;
        zoom.y_max = ny1;
        zoom.y_manual = true;
    }
    zoom.sync_active();
}

inline void plot_apply_pan_x(PlotZoomState& zoom, const PlotInteractConfig& cfg, double dx) {
    if (!zoom.x_active) {
        zoom.x_min = cfg.def_xmin;
        zoom.x_max = cfg.def_xmax;
        zoom.x_active = true;
    }
    const double span = std::max(1e-12, zoom.x_max - zoom.x_min);
    double nx0 = zoom.x_min + dx;
    double nx1 = nx0 + span;
    if (cfg.clamp_x) {
        if (nx0 < cfg.bound_xmin) { nx0 = cfg.bound_xmin; nx1 = std::min(cfg.bound_xmax, nx0 + span); }
        if (nx1 > cfg.bound_xmax) { nx1 = cfg.bound_xmax; nx0 = std::max(cfg.bound_xmin, nx1 - span); }
    }
    zoom.x_min = nx0;
    zoom.x_max = nx1;
    zoom.sync_active();
}

inline void plot_apply_pan_y(PlotZoomState& zoom, const PlotInteractConfig& cfg, double dy) {
    if (!cfg.allow_y_zoom) return;
    if (!zoom.y_manual) {
        zoom.y_min = cfg.def_ymin;
        zoom.y_max = cfg.def_ymax;
        zoom.y_manual = true;
    }
    const double span = std::max(1e-12, zoom.y_max - zoom.y_min);
    double ny0 = zoom.y_min + dy;
    double ny1 = ny0 + span;
    if (cfg.clamp_y) {
        if (ny0 < cfg.bound_ymin) { ny0 = cfg.bound_ymin; ny1 = std::min(cfg.bound_ymax, ny0 + span); }
        if (ny1 > cfg.bound_ymax) { ny1 = cfg.bound_ymax; ny0 = std::max(cfg.bound_ymin, ny1 - span); }
    }
    zoom.y_min = ny0;
    zoom.y_max = ny1;
    zoom.sync_active();
}

inline void plot_apply_hi_focus(PlotZoomState& zoom, const PlotInteractConfig& cfg) {
    if (!cfg.has_hi_focus || !(cfg.hi_xmax > cfg.hi_xmin)) return;
    double x0 = cfg.hi_xmin;
    double x1 = cfg.hi_xmax;
    if (cfg.clamp_x) {
        x0 = std::clamp(x0, cfg.bound_xmin, cfg.bound_xmax);
        x1 = std::clamp(x1, cfg.bound_xmin, cfg.bound_xmax);
    }
    if (x1 > x0) {
        zoom.x_min = x0;
        zoom.x_max = x1;
        zoom.x_active = true;
        zoom.y_manual = false;
        zoom.sync_active();
    }
}

inline void plot_apply_full_view(PlotZoomState& zoom, const PlotInteractConfig& cfg) {
    const double b_span = std::max(1e-9, cfg.bound_xmax - cfg.bound_xmin);
    const bool def_is_bound = (std::abs(cfg.def_xmin - cfg.bound_xmin) <= b_span * 1e-5) &&
                              (std::abs(cfg.def_xmax - cfg.bound_xmax) <= b_span * 1e-5);
    if (def_is_bound) {
        zoom.reset();
    } else {
        zoom.x_min = cfg.bound_xmin;
        zoom.x_max = cfg.bound_xmax;
        zoom.x_active = true;
        zoom.y_manual = false;
        zoom.sync_active();
    }
}

} // namespace detail

namespace widgets {

inline bool draw_plot_zoom_toolbar(
    const char* id_prefix,
    ImDrawList* dl,
    const ImVec2& plot_min, const ImVec2& plot_max,
    const PlotInteractConfig& cfg,
    PlotZoomState& zoom,
    PlotInteractResult& io_res,
    ImFont* ui_font = nullptr,
    ImFont* mono_font = nullptr,
    bool force_visible = true) {
    const float avail_w = plot_max.x - plot_min.x;
    const float avail_h = plot_max.y - plot_min.y;
    if (avail_w < dp(84.0f) || avail_h < dp(36.0f)) return false;

    const float bar_h = dp(22.0f);
    const float pad_h = dp(6.0f);

    const double ref_span = std::max(1e-9, cfg.def_xmax - cfg.def_xmin);
    const double cur_span = zoom.x_active ? std::max(1e-9, zoom.x_max - zoom.x_min) : ref_span;
    const double zoom_mag = ref_span / cur_span;
    const bool show_mag = zoom.x_active && (zoom_mag >= 1.05 || zoom_mag <= 0.95);

    char mag_buf[16] = "";
    if (show_mag) {
        if (zoom_mag >= 100.0) std::snprintf(mag_buf, sizeof(mag_buf), "%.0fx", zoom_mag);
        else if (zoom_mag >= 10.0) std::snprintf(mag_buf, sizeof(mag_buf), "%.0fx", zoom_mag);
        else std::snprintf(mag_buf, sizeof(mag_buf), "%.1fx", zoom_mag);
    }

    struct TbItem {
        const char* id;
        const char* label;
        const char* tip;
        int action; // 0=zoom_in, 1=zoom_out, 2=auto_y, 3=hi, 4=full, 5=reset, 6=readout
        bool highlight;
        bool is_mono;
        float w;
    };

    TbItem items[8];
    int n_items = 0;

    ImFont* uf = ui_font ? ui_font : ImGui::GetFont();
    ImFont* mf = mono_font ? mono_font : uf;
    const float ufs = font_dp(11.0f);
    const float mfs = font_dp(11.5f);

    auto calc_w = [&](const char* txt, bool mono, float min_w) {
        ImFont* f = mono ? mf : uf;
        const float fs = mono ? mfs : ufs;
        return std::max(dp(min_w), std::ceil(f->CalcTextSizeA(fs, FLT_MAX, 0.0f, txt).x) + dp(10.0f));
    };

    auto push_item = [&](const char* id, const char* lbl, const char* tip, int act, bool hi, bool mono, float min_w) {
        if (n_items < 8) {
            items[n_items++] = {id, lbl, tip, act, hi, mono, calc_w(lbl, mono, min_w)};
        }
    };

    if (cfg.is_compact) {
        push_item("##tb_zi", "+", "Zoom In (+)", 0, false, true, 20.0f);
        push_item("##tb_zo", "-", "Zoom Out (-)", 1, false, true, 20.0f);
        if (zoom.y_manual && cfg.allow_y_zoom) {
            push_item("##tb_ay", "Auto Y", "Auto-Fit Y (A)", 2, true, false, 38.0f);
        }
        if (show_mag) {
            push_item("##tb_mg", mag_buf, "Current X zoom factor", 6, true, true, 28.0f);
        }
        if (zoom.is_zoomed()) {
            push_item("##tb_rs", "Reset", "Reset View (R / Dbl-Click)", 5, false, false, 34.0f);
        }
    } else {
        push_item("##tb_zi", "+", "Zoom In (+)", 0, false, true, 22.0f);
        push_item("##tb_zo", "-", "Zoom Out (-)", 1, false, true, 22.0f);
        if (cfg.allow_y_zoom) {
            const char* ay_lbl = (cfg.auto_y_label && cfg.auto_y_label[0]) ? cfg.auto_y_label : "Auto Y";
            push_item("##tb_ay", ay_lbl, "Auto-Fit Y Axis (A)", 2, zoom.y_manual, false, 44.0f);
        }
        if (cfg.has_hi_focus && avail_w > dp(260.0f)) {
            push_item("##tb_hi", "H I", "Focus 21-cm H I Line (H)", 3, false, false, 28.0f);
        }
        if (cfg.show_full_btn && avail_w > dp(310.0f)) {
            push_item("##tb_fl", "Full", "Full Passband Span (F)", 4, false, false, 32.0f);
        }
        if (show_mag) {
            push_item("##tb_mg", mag_buf, "Current X magnification", 6, true, true, 30.0f);
        }
        push_item("##tb_rs", "Reset", "Reset View (R / 0 / Dbl-Click)", 5, zoom.is_zoomed(), false, 38.0f);
    }

    float total_w = 0.0f;
    for (int i = 0; i < n_items; ++i) total_w += items[i].w;
    if (total_w > avail_w - pad_h * 2.0f) return false;

    const ImVec2 tb_min(std::floor(plot_max.x - pad_h - total_w), std::floor(plot_min.y + pad_h));
    const ImVec2 tb_max(tb_min.x + total_w, tb_min.y + bar_h);
    io_res.toolbar_min = tb_min;
    io_res.toolbar_max = tb_max;

    const bool mouse_in_tb = ImGui::IsMouseHoveringRect(tb_min, tb_max, true);
    if (!force_visible && !mouse_in_tb && !zoom.is_zoomed()) {
        return false;
    }

    // Background bezel pill
    dl->AddRectFilled(tb_min, tb_max, palette::PillBg32, dp(4.0f), ImDrawFlags_RoundAll);
    dl->AddRect(tb_min, tb_max, palette::PlotBorder32, dp(4.0f), 1.0f, ImDrawFlags_RoundAll);

    ImGui::PushID(id_prefix);
    float cx = tb_min.x;
    bool any_over = mouse_in_tb;

    for (int i = 0; i < n_items; ++i) {
        const TbItem& it = items[i];
        const ImVec2 bmin(cx, tb_min.y);
        const ImVec2 bmax(cx + it.w, tb_max.y);

        if (i > 0) {
            dl->AddLineV(cx, tb_min.y + dp(3.0f), tb_max.y - dp(3.0f),
                         palette::WithAlphaF(palette::PlotBorder32, 0.75f), 1.0f);
        }

        bool hov = false;
        bool clicked = false;
        if (it.action != 6) {
            ImGui::SetCursorScreenPos(bmin);
            clicked = ImGui::InvisibleButton(it.id, ImVec2(it.w, bar_h));
            hov = ImGui::IsItemHovered();
            if (hov) {
                any_over = true;
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (it.tip && it.tip[0]) {
                    ImGui::SetTooltip("%s", it.tip);
                }
            }
        }

        if (hov) {
            dl->AddRectFilled(ImVec2(bmin.x + 1.0f, bmin.y + 1.0f),
                              ImVec2(bmax.x - 1.0f, bmax.y - 1.0f),
                              palette::U32(palette::SurfaceHover, 0.85f), dp(3.0f), ImDrawFlags_RoundAll);
        } else if (it.highlight && it.action != 6) {
            dl->AddRectFilled(ImVec2(bmin.x + 1.0f, bmin.y + 1.0f),
                              ImVec2(bmax.x - 1.0f, bmax.y - 1.0f),
                              palette::U32(palette::Cyan, 0.14f), dp(3.0f), ImDrawFlags_RoundAll);
        }

        ImFont* f = it.is_mono ? mf : uf;
        const float fs = it.is_mono ? mfs : ufs;
        const ImVec2 tsz = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, it.label);
        const ImU32 fg = (it.highlight || hov)
            ? palette::U32(palette::Cyan)
            : (it.action == 5 && !zoom.is_zoomed() ? palette::U32(palette::TextDim) : palette::TextPrimary32);
        dl->AddText(f, fs,
                    ImVec2(std::floor(bmin.x + (it.w - tsz.x) * 0.5f),
                           std::floor(bmin.y + (bar_h - tsz.y) * 0.5f)),
                    fg, it.label);

        if (clicked) {
            switch (it.action) {
                case 0:
                    detail::plot_apply_zoom_x(zoom, cfg, 0.72, 0.5);
                    io_res.changed = true;
                    io_res.x_changed = true;
                    break;
                case 1:
                    detail::plot_apply_zoom_x(zoom, cfg, 1.0 / 0.72, 0.5);
                    io_res.changed = true;
                    io_res.x_changed = true;
                    break;
                case 2:
                    zoom.auto_y();
                    io_res.requested_auto_y = true;
                    io_res.y_changed = true;
                    io_res.changed = true;
                    break;
                case 3:
                    detail::plot_apply_hi_focus(zoom, cfg);
                    io_res.requested_hi_focus = true;
                    io_res.x_changed = true;
                    io_res.changed = true;
                    break;
                case 4:
                    detail::plot_apply_full_view(zoom, cfg);
                    io_res.requested_full = true;
                    io_res.x_changed = true;
                    io_res.changed = true;
                    break;
                case 5:
                    zoom.reset();
                    io_res.requested_reset = true;
                    io_res.x_changed = true;
                    io_res.y_changed = true;
                    io_res.changed = true;
                    break;
                default:
                    break;
            }
        }
        cx += it.w;
    }

    ImGui::PopID();
    io_res.over_toolbar = any_over;
    return any_over;
}

} // namespace widgets

inline PlotInteractResult handle_plot_interaction(
    const char* hit_id,
    const ImVec2& canvas_min, const ImVec2& canvas_size,
    const ImVec2& plot_min, const ImVec2& plot_max,
    const PlotInteractConfig& cfg,
    PlotZoomState& zoom,
    ImDrawList* dl,
    ImFont* ui_font = nullptr,
    ImFont* mono_font = nullptr,
    const std::function<void()>& extra_context_menu = nullptr,
    bool draw_toolbar = true) {
    PlotInteractResult res;
    const ImVec2 canvas_max(canvas_min.x + canvas_size.x, canvas_min.y + canvas_size.y);
    const float pw = std::max(1.0f, plot_max.x - plot_min.x);
    const float ph = std::max(1.0f, plot_max.y - plot_min.y);
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mp = io.MousePos;

    // 1. Overlapping interactive hit area
    ImGui::SetCursorScreenPos(canvas_min);
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton(hit_id, canvas_size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool item_hovered = ImGui::IsItemHovered();
    const bool item_focused = ImGui::IsItemFocused();

    char popup_buf[96];
    const char* popup_name = cfg.popup_id;
    if (!popup_name || !popup_name[0]) {
        std::snprintf(popup_buf, sizeof(popup_buf), "##PlotCtx_%s", hit_id ? hit_id : "default");
        popup_name = popup_buf;
    }

    // 2. Floating top-right zoom toolbar (sits above canvas via AllowOverlap)
    if (draw_toolbar && dl) {
        const bool show_tb = !cfg.is_compact || item_hovered || zoom.is_zoomed();
        widgets::draw_plot_zoom_toolbar(hit_id ? hit_id : "##tb", dl, plot_min, plot_max,
                                        cfg, zoom, res, ui_font, mono_font, show_tb);
    }

    const bool popup_open = ImGui::IsPopupOpen(popup_name);
    const bool any_popup_open = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup);
    const bool in_plot   = (mp.x >= plot_min.x && mp.x <= plot_max.x && mp.y >= plot_min.y && mp.y <= plot_max.y);
    const bool in_y_axis = (mp.x >= canvas_min.x && mp.x < plot_min.x && mp.y >= plot_min.y && mp.y <= plot_max.y);
    const bool in_x_axis = (mp.x >= plot_min.x && mp.x <= plot_max.x && mp.y > plot_max.y && mp.y <= canvas_max.y);
    const bool can_interact = item_hovered && !res.over_toolbar && !popup_open;

    auto map_px_to_x = [&](float px) -> double {
        const double x0 = zoom.x_active ? zoom.x_min : cfg.def_xmin;
        const double x1 = zoom.x_active ? zoom.x_max : cfg.def_xmax;
        const double u = std::clamp(static_cast<double>(px - plot_min.x) / static_cast<double>(pw), 0.0, 1.0);
        return x0 + u * (x1 - x0);
    };
    auto map_py_to_y = [&](float py) -> double {
        const double y0 = zoom.y_manual ? zoom.y_min : cfg.def_ymin;
        const double y1 = zoom.y_manual ? zoom.y_max : cfg.def_ymax;
        const double v = cfg.y_inverted
            ? std::clamp(static_cast<double>(py - plot_min.y) / static_cast<double>(ph), 0.0, 1.0)
            : std::clamp(static_cast<double>(plot_max.y - py) / static_cast<double>(ph), 0.0, 1.0);
        return y0 + v * (y1 - y0);
    };

    // 3. Scroll Wheel & Horizontal Wheel
    if (can_interact) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelX);

        if (io.MouseWheel != 0.0f) {
            const double factor = std::pow(0.85, static_cast<double>(io.MouseWheel));
            const double u = std::clamp(static_cast<double>(mp.x - plot_min.x) / static_cast<double>(pw), 0.0, 1.0);
            const double v = cfg.y_inverted
                ? std::clamp(static_cast<double>(mp.y - plot_min.y) / static_cast<double>(ph), 0.0, 1.0)
                : std::clamp(static_cast<double>(plot_max.y - mp.y) / static_cast<double>(ph), 0.0, 1.0);

            bool do_x = false;
            bool do_y = false;
            if (in_y_axis) {
                do_y = cfg.allow_y_zoom;
            } else if (in_x_axis) {
                do_x = true;
            } else if (in_plot) {
                if (io.KeyCtrl) {
                    do_x = true;
                    do_y = cfg.allow_y_zoom;
                } else if (io.KeyShift) {
                    do_y = cfg.allow_y_zoom;
                } else {
                    do_x = true;
                }
            }
            if (do_x) {
                detail::plot_apply_zoom_x(zoom, cfg, factor, u);
                res.x_changed = true;
                res.changed = true;
            }
            if (do_y) {
                detail::plot_apply_zoom_y(zoom, cfg, factor, v);
                res.y_changed = true;
                res.changed = true;
            }
        }

        if (io.MouseWheelH != 0.0f && (in_plot || in_x_axis)) {
            const double cur_span = zoom.x_active ? (zoom.x_max - zoom.x_min) : (cfg.def_xmax - cfg.def_xmin);
            detail::plot_apply_pan_x(zoom, cfg, -static_cast<double>(io.MouseWheelH) * 0.12 * cur_span);
            res.x_changed = true;
            res.changed = true;
        }

        // 4. Double-Click Left
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            zoom.drag_mode = 0;
            zoom.box_dragging = false;
            if (in_y_axis && cfg.allow_y_zoom) {
                zoom.auto_y();
                res.requested_auto_y = true;
                res.y_changed = true;
                res.changed = true;
            } else if (in_x_axis) {
                zoom.x_active = false;
                zoom.sync_active();
                res.x_changed = true;
                res.changed = true;
            } else if (in_plot) {
                zoom.reset();
                res.requested_reset = true;
                res.x_changed = true;
                res.y_changed = true;
                res.changed = true;
            }
        }
    }

    // 5. Commit helper for rubber-band box zoom
    auto commit_box_zoom = [&]() {
        const float bx0 = std::min(zoom.box_start_px.x, zoom.box_cur_px.x);
        const float bx1 = std::max(zoom.box_start_px.x, zoom.box_cur_px.x);
        const float by0 = std::min(zoom.box_start_px.y, zoom.box_cur_px.y);
        const float by1 = std::max(zoom.box_start_px.y, zoom.box_cur_px.y);
        const float bw = bx1 - bx0;
        const float bh = by1 - by0;

        if (bw < dp(8.0f)) return; // Reject accidental micro-drags

        double nx0 = map_px_to_x(bx0);
        double nx1 = map_px_to_x(bx1);
        if (nx0 > nx1) std::swap(nx0, nx1);

        const double b_xspan = std::max(1e-9, cfg.bound_xmax - cfg.bound_xmin);
        const double min_xspan = b_xspan * std::clamp(cfg.min_x_span_frac, 1e-6, 0.5);
        if (nx1 - nx0 < min_xspan) {
            const double mid = 0.5 * (nx0 + nx1);
            nx0 = mid - 0.5 * min_xspan;
            nx1 = mid + 0.5 * min_xspan;
        }
        if (cfg.clamp_x) {
            nx0 = std::clamp(nx0, cfg.bound_xmin, cfg.bound_xmax);
            nx1 = std::clamp(nx1, cfg.bound_xmin, cfg.bound_xmax);
        }
        if (nx1 > nx0) {
            zoom.x_min = nx0;
            zoom.x_max = nx1;
            zoom.x_active = true;
            res.x_changed = true;
            res.changed = true;
        }

        // Smart 1D X-band zoom when height < 12 dp; full 2D box zoom otherwise
        if (bh >= dp(12.0f) && cfg.allow_y_zoom) {
            double ny0 = map_py_to_y(by0);
            double ny1 = map_py_to_y(by1);
            if (ny0 > ny1) std::swap(ny0, ny1);
            const double ref_yspan = cfg.clamp_y
                ? std::max(1e-9, cfg.bound_ymax - cfg.bound_ymin)
                : std::max(1e-9, cfg.def_ymax - cfg.def_ymin);
            const double min_yspan = ref_yspan * std::clamp(cfg.min_y_span_frac, 1e-6, 0.5);
            if (ny1 - ny0 < min_yspan) {
                const double mid = 0.5 * (ny0 + ny1);
                ny0 = mid - 0.5 * min_yspan;
                ny1 = mid + 0.5 * min_yspan;
            }
            if (cfg.clamp_y) {
                ny0 = std::clamp(ny0, cfg.bound_ymin, cfg.bound_ymax);
                ny1 = std::clamp(ny1, cfg.bound_ymin, cfg.bound_ymax);
            }
            if (ny1 > ny0) {
                zoom.y_min = ny0;
                zoom.y_max = ny1;
                zoom.y_manual = true;
                res.y_changed = true;
                res.changed = true;
            }
        }
        zoom.sync_active();
    };

    // 6. Left-Drag Pan & Shift+Left-Drag Box Zoom
    if (can_interact && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        if (!(cfg.ignore_ctrl_left_drag && io.KeyCtrl)) {
            if (io.KeyShift && in_plot) {
                zoom.drag_mode = 5;
                zoom.box_dragging = true;
                zoom.box_start_px = mp;
                zoom.box_cur_px = mp;
            } else if (in_y_axis && cfg.allow_y_zoom) {
                zoom.drag_mode = 3;
            } else if (in_x_axis) {
                zoom.drag_mode = 2;
            } else if (in_plot) {
                zoom.drag_mode = 1;
            }
        }
    }

    if (zoom.drag_mode == 1 || zoom.drag_mode == 2 || zoom.drag_mode == 3 || zoom.drag_mode == 5) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (zoom.drag_mode == 5) {
                zoom.box_cur_px = ImVec2(std::clamp(mp.x, plot_min.x, plot_max.x),
                                         std::clamp(mp.y, plot_min.y, plot_max.y));
            } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
                const double x_span = zoom.x_active ? (zoom.x_max - zoom.x_min) : (cfg.def_xmax - cfg.def_xmin);
                const double y_span = zoom.y_manual ? (zoom.y_max - zoom.y_min) : (cfg.def_ymax - cfg.def_ymin);
                if (zoom.drag_mode == 1 || zoom.drag_mode == 2) {
                    if (io.MouseDelta.x != 0.0f) {
                        detail::plot_apply_pan_x(zoom, cfg, -(static_cast<double>(io.MouseDelta.x) / static_cast<double>(pw)) * x_span);
                        res.x_changed = true;
                        res.changed = true;
                    }
                }
                const bool pan_y = (zoom.drag_mode == 3) ||
                                   (zoom.drag_mode == 1 && (zoom.y_manual || cfg.y_inverted));
                if (pan_y && cfg.allow_y_zoom && io.MouseDelta.y != 0.0f) {
                    const double sign = cfg.y_inverted ? -1.0 : 1.0;
                    detail::plot_apply_pan_y(zoom, cfg, sign * (static_cast<double>(io.MouseDelta.y) / static_cast<double>(ph)) * y_span);
                    res.y_changed = true;
                    res.changed = true;
                }
                ImGui::SetMouseCursor(zoom.drag_mode == 3 ? ImGuiMouseCursor_ResizeNS
                                    : (zoom.drag_mode == 2 ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeAll));
            }
        } else {
            if (zoom.drag_mode == 5 && zoom.box_dragging) {
                commit_box_zoom();
            }
            zoom.drag_mode = 0;
            zoom.box_dragging = false;
        }
    }

    // 7. Right-Drag 2D Box Zoom & Short Right-Click Context Menu
    if (can_interact && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        zoom.rclick_press_px = mp;
        if (in_plot) {
            zoom.drag_mode = 4;
            zoom.box_start_px = mp;
            zoom.box_cur_px = mp;
            zoom.box_dragging = false;
        } else if (in_y_axis || in_x_axis) {
            zoom.drag_mode = 6;
            zoom.box_dragging = false;
        }
    }

    if (zoom.drag_mode == 4 || zoom.drag_mode == 6) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            if (zoom.drag_mode == 4) {
                zoom.box_cur_px = ImVec2(std::clamp(mp.x, plot_min.x, plot_max.x),
                                         std::clamp(mp.y, plot_min.y, plot_max.y));
                const float dist = std::hypot(mp.x - zoom.rclick_press_px.x, mp.y - zoom.rclick_press_px.y);
                if (dist >= dp(4.0f)) {
                    zoom.box_dragging = true;
                }
            }
        } else {
            const float dist = std::hypot(mp.x - zoom.rclick_press_px.x, mp.y - zoom.rclick_press_px.y);
            if (zoom.drag_mode == 4 && zoom.box_dragging) {
                commit_box_zoom();
            } else if (dist < dp(4.0f)) {
                ImGui::OpenPopup(popup_name);
                res.context_opened_this_frame = true;
            }
            zoom.drag_mode = 0;
            zoom.box_dragging = false;
        }
    }

    // 8. Esc cancels active rubber-band drag
    if (zoom.box_dragging && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        zoom.box_dragging = false;
        zoom.drag_mode = 0;
    }

    // 9. Draw Rubber-Band Box Zoom overlay + live delta readout pill
    if (zoom.box_dragging && dl) {
        const float bx0 = std::min(zoom.box_start_px.x, zoom.box_cur_px.x);
        const float bx1 = std::max(zoom.box_start_px.x, zoom.box_cur_px.x);
        const float by0 = std::min(zoom.box_start_px.y, zoom.box_cur_px.y);
        const float by1 = std::max(zoom.box_start_px.y, zoom.box_cur_px.y);
        const float bw = bx1 - bx0;
        const float bh = by1 - by0;
        const bool is_1d_xband = (bh < dp(12.0f) || !cfg.allow_y_zoom);

        dl->PushClipRect(plot_min, plot_max, true);
        if (is_1d_xband) {
            dl->AddRectFilled(ImVec2(bx0, plot_min.y), ImVec2(bx1, plot_max.y),
                              palette::U32(palette::Cyan, 0.14f));
            dl->AddLineV(bx0, plot_min.y, plot_max.y, palette::U32(palette::Cyan, 0.90f), 1.5f);
            dl->AddLineV(bx1, plot_min.y, plot_max.y, palette::U32(palette::Cyan, 0.90f), 1.5f);
        } else {
            dl->AddRectFilled(ImVec2(bx0, by0), ImVec2(bx1, by1),
                              palette::U32(palette::Cyan, 0.14f));
            dl->AddRect(ImVec2(bx0, by0), ImVec2(bx1, by1),
                        palette::U32(palette::Cyan, 0.92f), 0.0f, 1.5f, ImDrawFlags_None);
        }

        if (bw >= dp(8.0f)) {
            const double dx_val = std::abs(map_px_to_x(bx1) - map_px_to_x(bx0));
            char dx_buf[32], dy_buf[32], pill_buf[96];
            format_adaptive_tick(dx_val, dx_val * 0.1, dx_buf, sizeof(dx_buf), false, 1);
            const char* xu = (cfg.x_unit && cfg.x_unit[0]) ? cfg.x_unit : "";
            const char* yu = (cfg.y_unit && cfg.y_unit[0]) ? cfg.y_unit : "";
            if (is_1d_xband) {
                std::snprintf(pill_buf, sizeof(pill_buf), "dX %s%s%s (X-Band)",
                              dx_buf, xu[0] ? " " : "", xu);
            } else {
                const double dy_val = std::abs(map_py_to_y(by1) - map_py_to_y(by0));
                format_adaptive_tick(dy_val, dy_val * 0.1, dy_buf, sizeof(dy_buf), false, 1);
                std::snprintf(pill_buf, sizeof(pill_buf), "dX %s%s%s | dY %s%s%s",
                              dx_buf, xu[0] ? " " : "", xu,
                              dy_buf, yu[0] ? " " : "", yu);
            }
            const float anchor_x = 0.5f * (bx0 + bx1);
            const float anchor_y = is_1d_xband ? (plot_min.y + dp(32.0f)) : (by0 - dp(24.0f));
            const auto pr = widgets::compute_clamped_pill_rect(
                pill_buf, mono_font ? mono_font : ui_font, 11.0f,
                anchor_x, anchor_y, plot_min, plot_max);
            widgets::draw_pill_badge(dl, pr, pill_buf, mono_font ? mono_font : ui_font, 11.0f,
                                     palette::CardBg32, palette::U32(palette::Cyan, 0.80f),
                                     palette::U32(palette::Cyan));
        }
        dl->PopClipRect();
    }

    // 10. Plot-local Keyboard Shortcuts (when hovered or focused & not typing)
    if ((can_interact || item_focused) && !io.WantTextInput && !any_popup_open && !io.KeyCtrl && !io.KeyAlt) {
        const double anchor_u = in_plot
            ? std::clamp(static_cast<double>(mp.x - plot_min.x) / static_cast<double>(pw), 0.0, 1.0)
            : 0.5;
        if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) {
            detail::plot_apply_zoom_x(zoom, cfg, 0.75, anchor_u);
            res.x_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) {
            detail::plot_apply_zoom_x(zoom, cfg, 1.0 / 0.75, anchor_u);
            res.x_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_0) || ImGui::IsKeyPressed(ImGuiKey_Keypad0) ||
                   ImGui::IsKeyPressed(ImGuiKey_R) || ImGui::IsKeyPressed(ImGuiKey_Home)) {
            zoom.reset();
            res.requested_reset = true; res.x_changed = true; res.y_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_A) && cfg.allow_y_zoom) {
            zoom.auto_y();
            res.requested_auto_y = true; res.y_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_H) && cfg.has_hi_focus) {
            detail::plot_apply_hi_focus(zoom, cfg);
            res.requested_hi_focus = true; res.x_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_F)) {
            detail::plot_apply_full_view(zoom, cfg);
            res.requested_full = true; res.x_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
            const double cur_span = zoom.x_active ? (zoom.x_max - zoom.x_min) : (cfg.def_xmax - cfg.def_xmin);
            detail::plot_apply_pan_x(zoom, cfg, -0.15 * cur_span);
            res.x_changed = true; res.changed = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
            const double cur_span = zoom.x_active ? (zoom.x_max - zoom.x_min) : (cfg.def_xmax - cfg.def_xmin);
            detail::plot_apply_pan_x(zoom, cfg, +0.15 * cur_span);
            res.x_changed = true; res.changed = true;
        }
    }

    // 11. Right-Click Context Menu
    if (ImGui::BeginPopup(popup_name)) {
        if (cfg.has_hi_focus) {
            if (ImGui::MenuItem("Focus H I Line", "H")) {
                detail::plot_apply_hi_focus(zoom, cfg);
                res.requested_hi_focus = true; res.x_changed = true; res.changed = true;
            }
        }
        if (cfg.allow_y_zoom) {
            const char* ay_lbl = (cfg.auto_y_label && cfg.auto_y_label[0]) ? cfg.auto_y_label : "Auto-Fit Y";
            if (ImGui::MenuItem(ay_lbl, "A")) {
                zoom.auto_y();
                res.requested_auto_y = true; res.y_changed = true; res.changed = true;
            }
        }
        if (cfg.show_full_btn) {
            if (ImGui::MenuItem("Full Passband", "F")) {
                detail::plot_apply_full_view(zoom, cfg);
                res.requested_full = true; res.x_changed = true; res.changed = true;
            }
        }
        if (ImGui::MenuItem("Reset View", "R")) {
            zoom.reset();
            res.requested_reset = true; res.x_changed = true; res.y_changed = true; res.changed = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Zoom In", "+")) {
            detail::plot_apply_zoom_x(zoom, cfg, 0.72, 0.5);
            res.x_changed = true; res.changed = true;
        }
        if (ImGui::MenuItem("Zoom Out", "-")) {
            detail::plot_apply_zoom_x(zoom, cfg, 1.0 / 0.72, 0.5);
            res.x_changed = true; res.changed = true;
        }
        if (extra_context_menu) {
            ImGui::Separator();
            extra_context_menu();
        }
        ImGui::EndPopup();
    }

    // 12. Axis Hover Cursor Affordances
    if (can_interact && zoom.drag_mode == 0) {
        if (in_y_axis && cfg.allow_y_zoom) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        } else if (in_x_axis) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
    }

    zoom.sync_active();
    res.hovered_plot = can_interact && in_plot;
    res.hovered_any = can_interact;
    res.box_active = zoom.box_dragging;
    res.context_open = popup_open;
    return res;
}

// Legacy compatibility wrapper
inline void handle_plot_zoom_pan(const ImVec2& plot_min, const ImVec2& plot_max,
                                 double def_xmin, double def_xmax,
                                 double def_ymin, double def_ymax,
                                 PlotZoomState& zoom,
                                 bool clamp_x_to_default = true,
                                 bool clamp_y_to_default = false) {
    const float pw = std::max(1.0f, plot_max.x - plot_min.x);
    const float ph = std::max(1.0f, plot_max.y - plot_min.y);
    const ImGuiIO& io = ImGui::GetIO();
    PlotInteractConfig cfg;
    cfg.def_xmin = cfg.bound_xmin = def_xmin;
    cfg.def_xmax = cfg.bound_xmax = def_xmax;
    cfg.def_ymin = cfg.bound_ymin = def_ymin;
    cfg.def_ymax = cfg.bound_ymax = def_ymax;
    cfg.clamp_x = clamp_x_to_default;
    cfg.clamp_y = clamp_y_to_default;

    if (ImGui::IsItemHovered()) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            zoom.reset();
            return;
        }
        if (io.MouseWheel != 0.0f &&
            io.MousePos.x >= plot_min.x && io.MousePos.x <= plot_max.x &&
            io.MousePos.y >= plot_min.y && io.MousePos.y <= plot_max.y) {
            const double factor = std::pow(0.85, static_cast<double>(io.MouseWheel));
            const double u = static_cast<double>(io.MousePos.x - plot_min.x) / pw;
            const double v = static_cast<double>(plot_max.y - io.MousePos.y) / ph;
            if (!io.KeyShift) detail::plot_apply_zoom_x(zoom, cfg, factor, u);
            if (io.KeyShift || io.KeyCtrl) detail::plot_apply_zoom_y(zoom, cfg, factor, v);
        }
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
        const double xs = zoom.x_active ? (zoom.x_max - zoom.x_min) : (def_xmax - def_xmin);
        detail::plot_apply_pan_x(zoom, cfg, -(static_cast<double>(io.MouseDelta.x) / pw) * xs);
        if (zoom.y_manual) {
            const double ys = zoom.y_max - zoom.y_min;
            detail::plot_apply_pan_y(zoom, cfg, (static_cast<double>(io.MouseDelta.y) / ph) * ys);
        }
    }
}

struct PlotViewport2D {
    ImVec2 origin{0, 0};
    ImVec2 total_size{0, 0};
    ImVec2 plot_min{0, 0};
    ImVec2 plot_max{0, 0};
    float  plot_w = 1.0f;
    float  plot_h = 1.0f;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double inv_x_span = 1.0;
    double inv_y_span = 1.0;

    static PlotViewport2D create(ImVec2 origin_px, ImVec2 size_px,
                                 double xmin, double xmax,
                                 double ymin, double ymax,
                                 const PlotMargin& m = PlotMargin{}) {
        PlotViewport2D vp;
        vp.origin = origin_px;
        vp.total_size = size_px;
        vp.plot_min = ImVec2(origin_px.x + m.left, origin_px.y + m.top);
        vp.plot_max = ImVec2(
            std::max(vp.plot_min.x + 24.0f, origin_px.x + size_px.x - m.right),
            std::max(vp.plot_min.y + 24.0f, origin_px.y + size_px.y - m.bottom)
        );
        vp.plot_w = vp.plot_max.x - vp.plot_min.x;
        vp.plot_h = vp.plot_max.y - vp.plot_min.y;
        vp.x_min = xmin;
        vp.x_max = (std::abs(xmax - xmin) < 1e-12) ? (xmin + 1.0) : xmax;
        vp.y_min = ymin;
        vp.y_max = (std::abs(ymax - ymin) < 1e-12) ? (ymin + 1.0) : ymax;
        vp.inv_x_span = 1.0 / (vp.x_max - vp.x_min);
        vp.inv_y_span = 1.0 / (vp.y_max - vp.y_min);
        return vp;
    }

    float map_x(double x) const {
        return plot_min.x + static_cast<float>((x - x_min) * inv_x_span) * plot_w;
    }

    float map_y(double y) const {
        return plot_max.y - static_cast<float>((y - y_min) * inv_y_span) * plot_h;
    }

    double unmap_x(float px) const {
        return x_min + (static_cast<double>(px - plot_min.x) / plot_w) * (x_max - x_min);
    }

    double unmap_y(float py) const {
        return y_max - (static_cast<double>(py - plot_min.y) / plot_h) * (y_max - y_min);
    }

    bool contains(ImVec2 p) const {
        return p.x >= plot_min.x && p.x <= plot_max.x && p.y >= plot_min.y && p.y <= plot_max.y;
    }

    void draw_background_and_grid(ImDrawList* dl,
                                  int x_ticks, int y_ticks,
                                  const char* x_fmt, const char* y_fmt,
                                  const char* x_label = nullptr,
                                  const char* y_label = nullptr,
                                  ImFont* mono_font = nullptr) const {
        dl->AddRectFilled(plot_min, plot_max, palette::PlotBg32, dp(3.0f), ImDrawFlags_RoundAll);

        int x_min_dec = 0, y_min_dec = 0;
        bool x_sign = false, y_sign = false;
        const char* x_sfx = "";
        const char* y_sfx = "";
        parse_tick_fmt_hint(x_fmt, x_min_dec, x_sign, x_sfx);
        parse_tick_fmt_hint(y_fmt, y_min_dec, y_sign, y_sfx);

        float tick_h = dp(12.0f);
        {
            ScopedFont tick_f(mono_font, 12.0f);
            tick_h = ImGui::GetFontSize();
            const ImU32 text_col = palette::U32(palette::TextMuted);

            // Horizontal gridlines & adaptive nice Y ticks
            if (y_ticks > 1) {
                const double y_span = y_max - y_min;
                const double step = calc_nice_step(y_span, y_ticks);
                const double first = std::ceil((y_min - step * 1e-6) / step) * step;
                for (int k = 0; k < 24; ++k) {
                    double val = first + static_cast<double>(k) * step;
                    if (val > y_max + step * 1e-4) break;
                    if (std::abs(val) < step * 1e-4) val = 0.0;
                    const float py = map_y(val);
                    if (py < plot_min.y - 1.0f || py > plot_max.y + 1.0f) continue;
                    dl->AddLineH(plot_min.x, plot_max.x, py, palette::PlotGrid32, 1.0f);

                    char num_buf[32], buf[40];
                    format_adaptive_tick(val, step, num_buf, sizeof(num_buf), y_sign, y_min_dec);
                    if (y_sfx && *y_sfx) std::snprintf(buf, sizeof(buf), "%s%s", num_buf, y_sfx);
                    else                 std::snprintf(buf, sizeof(buf), "%s", num_buf);
                    const ImVec2 ts = ImGui::CalcTextSize(buf);
                    dl->AddText(ImVec2(std::floor(plot_min.x - ts.x - dp(6.0f)), std::floor(py - tick_h * 0.5f)), text_col, buf);
                }
            }

            // Vertical gridlines & adaptive nice X ticks
            if (x_ticks > 1) {
                const double x_span = x_max - x_min;
                const double step = calc_nice_step(x_span, x_ticks);
                const double first = std::ceil((x_min - step * 1e-6) / step) * step;
                float last_label_right = -1e9f;
                for (int k = 0; k < 24; ++k) {
                    double val = first + static_cast<double>(k) * step;
                    if (val > x_max + step * 1e-4) break;
                    if (std::abs(val) < step * 1e-4) val = 0.0;
                    const float px = map_x(val);
                    if (px < plot_min.x - 1.0f || px > plot_max.x + 1.0f) continue;
                    dl->AddLineV(px, plot_min.y, plot_max.y, palette::PlotGrid32, 1.0f);

                    char num_buf[32], buf[40];
                    format_adaptive_tick(val, step, num_buf, sizeof(num_buf), x_sign, x_min_dec);
                    if (x_sfx && *x_sfx) std::snprintf(buf, sizeof(buf), "%s%s", num_buf, x_sfx);
                    else                 std::snprintf(buf, sizeof(buf), "%s", num_buf);
                    const ImVec2 ts = ImGui::CalcTextSize(buf);
                    const float lx = std::clamp(px - ts.x * 0.5f, plot_min.x, plot_max.x - ts.x);
                    if (lx >= last_label_right + dp(6.0f)) {
                        dl->AddText(ImVec2(std::floor(lx), std::floor(plot_max.y + dp(3.5f))), text_col, buf);
                        last_label_right = lx + ts.x;
                    }
                }
            }
        }

        if ((x_label && *x_label) || (y_label && *y_label)) {
            ScopedFont lbl_f(nullptr, 12.0f);
            if (x_label && *x_label) {
                const ImVec2 ts = ImGui::CalcTextSize(x_label);
                dl->AddText(ImVec2(std::floor(plot_min.x + (plot_w - ts.x) * 0.5f), std::floor(plot_max.y + tick_h + dp(4.5f))),
                            palette::U32(palette::TextDim), x_label);
            }
            if (y_label && *y_label) {
                const ImVec2 ys = ImGui::CalcTextSize(y_label);
                const float y_pos = std::floor(std::max(origin.y + dp(1.0f), plot_min.y - ys.y - dp(4.0f)));
                const float x_pos = std::floor(std::max(origin.x + dp(2.0f), plot_min.x - dp(8.0f) - ys.x));
                dl->AddText(ImVec2(x_pos, y_pos),
                            palette::U32(palette::TextMuted), y_label);
            }
        }

        dl->AddRect(plot_min, plot_max, palette::PlotBorder32, dp(3.0f), 1.0f, ImDrawFlags_RoundAll);
    }

    void draw_top_axis(ImDrawList* dl, double top_min, double top_max,
                       int ticks, const char* fmt, const char* suffix = nullptr,
                       ImFont* mono_font = nullptr) const {
        if (ticks <= 1) return;
        int min_dec = 0;
        bool sign = false;
        const char* sfx = "";
        parse_tick_fmt_hint(fmt, min_dec, sign, sfx);
        const double step = std::abs(top_max - top_min) / static_cast<double>(ticks - 1);

        float tick_h = dp(12.0f);
        {
            ScopedFont tick_f(mono_font, 12.0f);
            tick_h = ImGui::GetFontSize();
            const ImU32 text_col = palette::U32(palette::Cyan, 0.88f);
            const ImU32 tick_col = palette::U32(palette::Cyan, 0.55f);
            float last_label_right = plot_min.x + dp(10.0f);
            for (int i = 0; i < ticks; ++i) {
                const double frac = static_cast<double>(i) / (ticks - 1);
                const double val = top_min + frac * (top_max - top_min);
                const float px = plot_min.x + static_cast<float>(frac) * plot_w;
                dl->AddLineV(px, plot_min.y - dp(3.5f), plot_min.y, tick_col, 1.0f);

                char num_buf[36], buf[44];
                format_adaptive_tick(val, step, num_buf, sizeof(num_buf), sign, min_dec);
                if (sfx && *sfx) std::snprintf(buf, sizeof(buf), "%s%s", num_buf, sfx);
                else             std::snprintf(buf, sizeof(buf), "%s", num_buf);
                const ImVec2 ts = ImGui::CalcTextSize(buf);
                const float lx = px - ts.x * 0.5f;
                if (lx >= last_label_right + dp(6.0f) && lx + ts.x <= plot_max.x - dp(2.0f)) {
                    dl->AddText(ImVec2(std::floor(lx), std::floor(plot_min.y - tick_h - dp(4.0f))), text_col, buf);
                    last_label_right = lx + ts.x;
                }
            }
        }
        if (suffix && *suffix) {
            ScopedFont lbl_f(nullptr, 12.0f);
            const ImVec2 ts = ImGui::CalcTextSize(suffix);
            const float sy = std::floor(std::max(origin.y + dp(1.0f), plot_min.y - tick_h - ts.y - dp(6.5f)));
            dl->AddText(ImVec2(std::floor(plot_min.x + (plot_w - ts.x) * 0.5f), sy),
                        palette::U32(palette::TextDim), suffix);
        }
    }

    template <typename XGetter>
    bool find_visible_range_xy(size_t n_points, XGetter&& x_at, size_t& out_lo, size_t& out_hi) const {
        if (n_points < 2) return false;
        const double x_first = static_cast<double>(x_at(0));
        const double x_last  = static_cast<double>(x_at(n_points - 1));
        const bool ascending = (x_last >= x_first);

        if (ascending) {
            if (x_last < x_min || x_first > x_max) return false;
            size_t lo = 0, hi = n_points;
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (static_cast<double>(x_at(mid)) < x_min) lo = mid + 1;
                else hi = mid;
            }
            const size_t first_ge = lo;
            lo = 0; hi = n_points;
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (static_cast<double>(x_at(mid)) <= x_max) lo = mid + 1;
                else hi = mid;
            }
            const size_t first_gt = lo;
            out_lo = (first_ge > 0) ? (first_ge - 1) : 0;
            out_hi = (first_gt < n_points) ? first_gt : (n_points - 1);
            return out_hi > out_lo;
        } else {
            if (x_first < x_min || x_last > x_max) return false;
            size_t lo = 0, hi = n_points;
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (static_cast<double>(x_at(mid)) > x_max) lo = mid + 1;
                else hi = mid;
            }
            const size_t first_le = lo;
            lo = 0; hi = n_points;
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (static_cast<double>(x_at(mid)) >= x_min) lo = mid + 1;
                else hi = mid;
            }
            const size_t first_lt = lo;
            out_lo = (first_le > 0) ? (first_le - 1) : 0;
            out_hi = (first_lt < n_points) ? first_lt : (n_points - 1);
            return out_hi > out_lo;
        }
    }

    template <typename YGetter>
    void decimate_trace(size_t n_points, double trace_x_min, double trace_x_max,
                        YGetter&& y_at, std::vector<ImVec2>& out_pts) const {
        out_pts.clear();
        if (n_points < 2 || plot_w <= 1.0f) return;

        const double span = trace_x_max - trace_x_min;
        if (!(span > 0.0)) return;
        if (trace_x_max < x_min || trace_x_min > x_max) return;

        const double inv_step_x = static_cast<double>(n_points - 1) / span;
        const double step_x = span / static_cast<double>(n_points - 1);
        int idx_lo = static_cast<int>(std::floor((x_min - trace_x_min) * inv_step_x)) - 1;
        int idx_hi = static_cast<int>(std::ceil((x_max - trace_x_min) * inv_step_x)) + 1;
        idx_lo = std::clamp(idx_lo, 0, static_cast<int>(n_points) - 1);
        idx_hi = std::clamp(idx_hi, 0, static_cast<int>(n_points) - 1);
        if (idx_hi <= idx_lo) return;

        const int num_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const int visible_bins = idx_hi - idx_lo + 1;

        if (visible_bins <= num_cols * 2) {
            out_pts.reserve(static_cast<size_t>(visible_bins));
            for (int i = idx_lo; i <= idx_hi; ++i) {
                const double y = static_cast<double>(y_at(static_cast<size_t>(i)));
                if (!std::isfinite(y)) continue;
                const double x = trace_x_min + static_cast<double>(i) * step_x;
                out_pts.emplace_back(map_x(x), map_y(y));
            }
            return;
        }

        const double view_span = x_max - x_min;
        const double px_step = static_cast<double>(plot_w) / static_cast<double>(num_cols - 1);
        const double inv_cols = 1.0 / static_cast<double>(num_cols - 1);

        out_pts.reserve(static_cast<size_t>(num_cols));
        for (int col = 0; col < num_cols; ++col) {
            const double cx0 = x_min + (static_cast<double>(col) * inv_cols) * view_span;
            const double cx1 = x_min + (static_cast<double>(col + 1) * inv_cols) * view_span;
            if (cx1 < trace_x_min || cx0 > trace_x_max) continue;

            const double sx0 = std::max(cx0, trace_x_min);
            const double sx1 = std::min(cx1, trace_x_max);
            int i0 = static_cast<int>(std::floor((sx0 - trace_x_min) * inv_step_x));
            int i1 = static_cast<int>(std::ceil((sx1 - trace_x_min) * inv_step_x));
            i0 = std::clamp(i0, 0, static_cast<int>(n_points) - 1);
            i1 = std::clamp(i1, i0, static_cast<int>(n_points) - 1);

            double best_y = static_cast<double>(y_at(static_cast<size_t>(i0)));
            for (int i = i0 + 1; i <= i1; ++i) {
                const double v = static_cast<double>(y_at(static_cast<size_t>(i)));
                if (v > best_y) best_y = v;
            }
            if (!std::isfinite(best_y)) continue;
            const float px = static_cast<float>(plot_min.x + static_cast<double>(col) * px_step);
            const float py = map_y(best_y);
            out_pts.emplace_back(px, py);
        }
    }

    void draw_vertical_gradient_fill(ImDrawList* dl, const std::vector<ImVec2>& pts,
                                     ImU32 col_peak_top, ImU32 col_base_top, ImU32 col_floor) const {
        if (pts.size() < 2 || plot_h <= 1.0f) return;

        const ImVec2 white_uv = ImGui::GetFontTexUvWhitePixel();
        const float ceil_y = plot_min.y;
        const float floor_y = plot_max.y;
        const float inv_h = 1.0f / std::max(1.0f, floor_y - ceil_y);

        dl->PushClipRect(plot_min, plot_max, true);

        constexpr size_t kMaxColsPerChunk = 4096;
        size_t start = 0;
        while (start + 1 < pts.size()) {
            const size_t count = std::min(kMaxColsPerChunk, pts.size() - start);
            const int idx_count = static_cast<int>((count - 1) * 6);
            const int vtx_count = static_cast<int>(count * 2);

            dl->PrimReserve(idx_count, vtx_count);
            const unsigned int base_vtx = dl->_VtxCurrentIdx;

            for (size_t i = 0; i < count; ++i) {
                const ImVec2& p = pts[start + i];
                const float y_top = std::clamp(p.y, ceil_y, floor_y);
                const float h_norm = std::clamp((floor_y - y_top) * inv_h, 0.0f, 1.0f);
                const ImU32 top_col = palette::LerpU32(col_base_top, col_peak_top, h_norm);

                dl->PrimWriteVtx(ImVec2(p.x, y_top),   white_uv, top_col);
                dl->PrimWriteVtx(ImVec2(p.x, floor_y), white_uv, col_floor);
            }

            for (size_t i = 0; i + 1 < count; ++i) {
                const ImDrawIdx t0 = static_cast<ImDrawIdx>(base_vtx + i * 2);
                const ImDrawIdx b0 = static_cast<ImDrawIdx>(base_vtx + i * 2 + 1);
                const ImDrawIdx t1 = static_cast<ImDrawIdx>(base_vtx + (i + 1) * 2);
                const ImDrawIdx b1 = static_cast<ImDrawIdx>(base_vtx + (i + 1) * 2 + 1);
                dl->PrimWriteIdx(t0); dl->PrimWriteIdx(t1); dl->PrimWriteIdx(b1);
                dl->PrimWriteIdx(t0); dl->PrimWriteIdx(b1); dl->PrimWriteIdx(b0);
            }
            start += count - 1;
        }

        dl->PopClipRect();
    }

    template <typename YGetter>
    void draw_decimated_trace_with_fill(ImDrawList* dl, size_t n_points,
                                        double trace_x_min, double trace_x_max,
                                        YGetter&& y_at,
                                        ImU32 stroke_col, float stroke_thickness,
                                        bool draw_fill, ImU32 col_peak_top, ImU32 col_base_top, ImU32 col_floor,
                                        std::vector<ImVec2>& scratch) const {
        decimate_trace(n_points, trace_x_min, trace_x_max, std::forward<YGetter>(y_at), scratch);
        if (scratch.size() < 2) return;

        if (draw_fill) {
            draw_vertical_gradient_fill(dl, scratch, col_peak_top, col_base_top, col_floor);
        }
        dl->PushClipRect(plot_min, plot_max, true);
        dl->AddPolyline(scratch.data(), static_cast<int>(scratch.size()), stroke_col, stroke_thickness, ImDrawFlags_None);
        dl->PopClipRect();
    }

    template <typename YGetter>
    void draw_decimated_trace(ImDrawList* dl, size_t n_points,
                              double trace_x_min, double trace_x_max,
                              YGetter&& y_at,
                              ImU32 col, float thickness,
                              std::vector<ImVec2>& scratch) const {
        if (n_points < 2 || plot_w < 2.0f) return;

        scratch.clear();
        const double span_x = trace_x_max - trace_x_min;
        if (!(span_x > 0.0) || trace_x_max < x_min || trace_x_min > x_max) return;

        const double inv_step_x = static_cast<double>(n_points - 1) / span_x;
        const double step_x = span_x / static_cast<double>(n_points - 1);
        int idx_lo = static_cast<int>(std::floor((x_min - trace_x_min) * inv_step_x)) - 1;
        int idx_hi = static_cast<int>(std::ceil((x_max - trace_x_min) * inv_step_x)) + 1;
        idx_lo = std::clamp(idx_lo, 0, static_cast<int>(n_points) - 1);
        idx_hi = std::clamp(idx_hi, 0, static_cast<int>(n_points) - 1);
        if (idx_hi <= idx_lo) return;

        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const int visible_bins = idx_hi - idx_lo + 1;

        dl->PushClipRect(plot_min, plot_max, true);

        if (visible_bins <= max_cols * 2) {
            scratch.reserve(static_cast<size_t>(visible_bins));
            for (int i = idx_lo; i <= idx_hi; ++i) {
                const double y = static_cast<double>(y_at(static_cast<size_t>(i)));
                if (std::isfinite(y)) {
                    const double x = trace_x_min + static_cast<double>(i) * step_x;
                    scratch.emplace_back(map_x(x), map_y(y));
                }
            }
        } else {
            scratch.reserve(static_cast<size_t>(max_cols) * 2 + 4);
            int current_col = -1;
            float col_x_first = 0.0f;
            float col_y_min = 0.0f;
            float col_y_max = 0.0f;
            bool has_sample = false;

            auto flush_col = [&]() {
                if (!has_sample) return;
                if (std::abs(col_y_max - col_y_min) < 0.35f) {
                    scratch.emplace_back(col_x_first, col_y_min);
                } else {
                    scratch.emplace_back(col_x_first, col_y_min);
                    scratch.emplace_back(col_x_first, col_y_max);
                }
                has_sample = false;
            };

            for (int i = idx_lo; i <= idx_hi; ++i) {
                const double y = static_cast<double>(y_at(static_cast<size_t>(i)));
                if (!std::isfinite(y)) continue;
                const double x = trace_x_min + static_cast<double>(i) * step_x;
                const float px = map_x(x);
                const float py = map_y(y);
                const int col_idx = static_cast<int>(std::floor(px - plot_min.x));
                if (col_idx != current_col) {
                    flush_col();
                    current_col = col_idx;
                    col_x_first = px;
                    col_y_min = py;
                    col_y_max = py;
                    has_sample = true;
                } else {
                    if (py < col_y_min) col_y_min = py;
                    if (py > col_y_max) col_y_max = py;
                }
            }
            flush_col();
        }

        if (scratch.size() >= 2) {
            dl->AddPolyline(scratch.data(), static_cast<int>(scratch.size()), col, thickness, ImDrawFlags_None);
        }
        dl->PopClipRect();
    }

    template <typename XGetter, typename YGetter>
    void draw_area_fill_above_ref(ImDrawList* dl, size_t n_points,
                                  XGetter&& x_at, YGetter&& y_at,
                                  double y_ref, ImVec4 base_col,
                                  float alpha_bottom, float alpha_peak,
                                  std::vector<ImVec2>& fill_scratch) const {
        if (n_points < 2 || plot_w < 2.0f || plot_h < 2.0f) return;

        size_t i_lo = 0, i_hi = 0;
        if (!find_visible_range_xy(n_points, x_at, i_lo, i_hi)) return;

        const float y_zero = std::clamp(map_y(y_ref), plot_min.y, plot_max.y);
        fill_scratch.clear();
        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const size_t visible_bins = i_hi - i_lo + 1;

        if (visible_bins <= static_cast<size_t>(max_cols * 2)) {
            fill_scratch.reserve(visible_bins);
            for (size_t i = i_lo; i <= i_hi; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                fill_scratch.emplace_back(map_x(x), map_y(y));
            }
        } else {
            fill_scratch.reserve(static_cast<size_t>(max_cols) + 4);
            int cur_col = -1;
            float col_x = 0.0f, col_top_y = 0.0f;
            bool has = false;
            auto flush = [&]() {
                if (has) {
                    fill_scratch.emplace_back(col_x, col_top_y);
                    has = false;
                }
            };
            for (size_t i = i_lo; i <= i_hi; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                const float px = map_x(x);
                const float py = map_y(y);
                const int c = static_cast<int>(std::floor(px - plot_min.x));
                if (c != cur_col) {
                    flush();
                    cur_col = c;
                    col_x = px;
                    col_top_y = py;
                    has = true;
                } else if (py < col_top_y) {
                    col_top_y = py;
                }
            }
            flush();
        }

        if (fill_scratch.size() < 2) return;
        if (fill_scratch.front().x > fill_scratch.back().x) {
            std::reverse(fill_scratch.begin(), fill_scratch.end());
        }

        float min_top_y = y_zero;
        for (const ImVec2& p : fill_scratch) {
            if (p.y < min_top_y) min_top_y = p.y;
        }
        const float peak_span = std::max(8.0f, y_zero - min_top_y);

        int active_quads = 0;
        for (size_t i = 1; i < fill_scratch.size(); ++i) {
            if (fill_scratch[i - 1].y < y_zero || fill_scratch[i].y < y_zero) {
                ++active_quads;
            }
        }
        if (active_quads == 0) return;

        const ImVec2 uv = ImGui::GetFontTexUvWhitePixel();
        const ImU32 col_bot = palette::U32(base_col, alpha_bottom);
        auto top_col_at = [&](float py) -> ImU32 {
            const float t = std::clamp((y_zero - py) / peak_span, 0.0f, 1.0f);
            return palette::U32(base_col, alpha_bottom + t * (alpha_peak - alpha_bottom));
        };

        dl->PushClipRect(plot_min, plot_max, true);
        dl->PrimReserve(active_quads * 6, active_quads * 4);
        for (size_t i = 1; i < fill_scratch.size(); ++i) {
            ImVec2 a = fill_scratch[i - 1];
            ImVec2 b = fill_scratch[i];
            if (a.y >= y_zero && b.y >= y_zero) continue;

            const float dy = b.y - a.y;
            if (a.y > y_zero && std::abs(dy) > 1e-6f) {
                const float t = (y_zero - a.y) / dy;
                a.x = a.x + t * (b.x - a.x);
                a.y = y_zero;
            } else if (b.y > y_zero && std::abs(dy) > 1e-6f) {
                const float t = (y_zero - a.y) / dy;
                b.x = a.x + t * (b.x - a.x);
                b.y = y_zero;
            }

            const ImDrawIdx vtx_base = static_cast<ImDrawIdx>(dl->_VtxCurrentIdx);
            dl->PrimWriteVtx(ImVec2(a.x, a.y),    uv, top_col_at(a.y));
            dl->PrimWriteVtx(ImVec2(b.x, b.y),    uv, top_col_at(b.y));
            dl->PrimWriteVtx(ImVec2(b.x, y_zero), uv, col_bot);
            dl->PrimWriteVtx(ImVec2(a.x, y_zero), uv, col_bot);
            dl->PrimWriteIdx(vtx_base + 0); dl->PrimWriteIdx(vtx_base + 1); dl->PrimWriteIdx(vtx_base + 2);
            dl->PrimWriteIdx(vtx_base + 0); dl->PrimWriteIdx(vtx_base + 2); dl->PrimWriteIdx(vtx_base + 3);
        }
        dl->PopClipRect();
    }

    template <typename XGetter, typename YGetter>
    void draw_decimated_xy(ImDrawList* dl, size_t n_points,
                           XGetter&& x_at, YGetter&& y_at,
                           ImU32 col, float thickness,
                           std::vector<ImVec2>& scratch) const {
        if (n_points < 2 || plot_w < 2.0f) return;

        size_t i_lo = 0, i_hi = 0;
        if (!find_visible_range_xy(n_points, x_at, i_lo, i_hi)) return;

        scratch.clear();
        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const size_t visible_bins = i_hi - i_lo + 1;
        dl->PushClipRect(plot_min, plot_max, true);

        if (visible_bins <= static_cast<size_t>(max_cols * 2)) {
            scratch.reserve(visible_bins);
            for (size_t i = i_lo; i <= i_hi; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                scratch.emplace_back(map_x(x), map_y(y));
            }
        } else {
            scratch.reserve(static_cast<size_t>(max_cols) * 2 + 4);
            int current_col = -1;
            float col_x = 0.0f, col_ymin = 0.0f, col_ymax = 0.0f;
            bool has_sample = false;
            auto flush_col = [&]() {
                if (!has_sample) return;
                scratch.emplace_back(col_x, col_ymin);
                if (std::abs(col_ymax - col_ymin) >= 0.35f) {
                    scratch.emplace_back(col_x, col_ymax);
                }
                has_sample = false;
            };
            for (size_t i = i_lo; i <= i_hi; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                const float px = map_x(x);
                const float py = map_y(y);
                const int c = static_cast<int>(std::floor(px - plot_min.x));
                if (c != current_col) {
                    flush_col();
                    current_col = c;
                    col_x = px;
                    col_ymin = py;
                    col_ymax = py;
                    has_sample = true;
                } else {
                    if (py < col_ymin) col_ymin = py;
                    if (py > col_ymax) col_ymax = py;
                }
            }
            flush_col();
        }

        if (scratch.size() >= 2) {
            dl->AddPolyline(scratch.data(), static_cast<int>(scratch.size()), col, thickness, ImDrawFlags_None);
        }
        dl->PopClipRect();
    }
};

// Shared 256-entry RGBA colormap builders (for Waterfall DX11 upload & Colorbar)
inline std::vector<uint32_t> build_colormap_lut256(int mode) {
    // 0: Viridis, 1: Inferno, 2: Plasma, 3: Turbo
    std::vector<uint32_t> lut(256, 0xFF000000u);
    auto lerp_c = [](uint8_t a, uint8_t b, float f) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(a + (b - a) * f, 0.0f, 255.0f));
    };

    struct Stop { uint8_t r, g, b; };
    static constexpr Stop viridis[5] = {{68,1,84}, {59,82,139}, {33,145,140}, {94,201,98}, {253,231,37}};
    static constexpr Stop inferno[5] = {{0,0,4}, {87,16,110}, {188,55,84}, {249,142,9}, {252,255,164}};
    static constexpr Stop plasma[5]  = {{13,8,135}, {126,3,168}, {204,71,120}, {248,149,64}, {240,249,33}};
    static constexpr Stop turbo[5]   = {{48,18,59}, {40,188,235}, {164,252,60}, {251,126,33}, {122,4,3}};

    const Stop* stops = viridis;
    if (mode == 1) stops = inferno;
    else if (mode == 2) stops = plasma;
    else if (mode == 3) stops = turbo;

    for (int i = 0; i < 256; ++i) {
        const float t = static_cast<float>(i) / 255.0f;
        const float pos = t * 4.0f;
        const int idx = std::clamp(static_cast<int>(pos), 0, 3);
        const float frac = pos - static_cast<float>(idx);
        const uint8_t r = lerp_c(stops[idx].r, stops[idx + 1].r, frac);
        const uint8_t g = lerp_c(stops[idx].g, stops[idx + 1].g, frac);
        const uint8_t b = lerp_c(stops[idx].b, stops[idx + 1].b, frac);
        lut[static_cast<size_t>(i)] = IM_COL32(r, g, b, 255);
    }
    return lut;
}

inline void draw_vertical_colorbar(
    ImDrawList* dl, ImVec2 bar_min, ImVec2 bar_max,
    double vmin, double vmax, const char* fmt,
    const uint32_t* lut256, ImFont* mono_font = nullptr,
    const char* unit_label = "dBFS") {
    const float w = bar_max.x - bar_min.x;
    const float h = bar_max.y - bar_min.y;
    if (!lut256 || w < 2.0f || h < 12.0f) return;

    if (unit_label && unit_label[0] != '\0') {
        ImFont* uf = mono_font ? mono_font : ImGui::GetFont();
        const float ufs = font_dp(11.0f);
        const ImVec2 usz = uf->CalcTextSizeA(ufs, FLT_MAX, 0.0f, unit_label);
        dl->AddText(uf, ufs, ImVec2(std::floor(bar_min.x), std::floor(bar_min.y - usz.y - dp(3.0f))), palette::TextMuted32, unit_label);
    }

    constexpr int kBands = 64;
    for (int i = 0; i < kBands; ++i) {
        const float t_top = static_cast<float>(i) / static_cast<float>(kBands);
        const float t_bot = static_cast<float>(i + 1) / static_cast<float>(kBands);
        const int idx_top = std::clamp(static_cast<int>((1.0f - t_top) * 255.0f), 0, 255);
        const int idx_bot = std::clamp(static_cast<int>((1.0f - t_bot) * 255.0f), 0, 255);
        const float y0 = std::floor(bar_min.y + t_top * h);
        const float y1 = std::floor(bar_min.y + t_bot * h);
        if (y1 > y0) {
            dl->AddRectFilledMultiColor(
                ImVec2(bar_min.x, y0), ImVec2(bar_max.x, y1),
                lut256[idx_top], lut256[idx_top], lut256[idx_bot], lut256[idx_bot]
            );
        }
    }
    dl->AddRect(bar_min, bar_max, palette::PlotBorder32, 1.0f, 1.0f, ImDrawFlags_RoundAll);

    ImFont* f = mono_font ? mono_font : ImGui::GetFont();
    const float fs = font_dp(12.0f);
    char buf[24];
    constexpr int kTicks = 5;
    for (int i = 0; i < kTicks; ++i) {
        const float frac = static_cast<float>(i) / static_cast<float>(kTicks - 1);
        const double val = vmax - frac * (vmax - vmin);
        const float y = std::floor(bar_min.y + frac * h);
        dl->AddLineH(bar_max.x, bar_max.x + dp(4.0f), y, palette::PlotBorder32, 1.0f);
        std::snprintf(buf, sizeof(buf), fmt ? fmt : "%.0f", val);
        const ImVec2 tsz = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, buf);
        dl->AddText(f, fs, ImVec2(std::floor(bar_max.x + dp(6.0f)), std::floor(y - tsz.y * 0.5f)), palette::TextMuted32, buf);
    }
}

inline void draw_vertical_colorbar(
    ImDrawList* dl, ImVec2 bar_min, ImVec2 bar_max,
    double vmin, double vmax, const char* fmt,
    const std::vector<uint32_t>& lut256, ImFont* mono_font = nullptr) {
    draw_vertical_colorbar(dl, bar_min, bar_max, vmin, vmax, fmt,
                           lut256.size() >= 256 ? lut256.data() : nullptr, mono_font, nullptr);
}

inline void draw_signal_level_bar(
    ImDrawList* dl, ImVec2 bar_min, ImVec2 bar_max,
    double db_min, double db_max, float live_peak_db, float floor_db, float hold_peak_db,
    bool has_hold, const uint32_t* lut256, ImFont* mono_font = nullptr) {
    const float h = bar_max.y - bar_min.y;
    const double span = std::max(1e-3, db_max - db_min);
    if (!lut256 || h < 12.0f) return;

    dl->AddRectFilled(bar_min, bar_max, palette::BezelBg32, dp(2.0f), ImDrawFlags_RoundAll);

    const float live_norm = std::clamp(static_cast<float>((live_peak_db - db_min) / span), 0.0f, 1.0f);
    const float y_live = std::floor(bar_max.y - live_norm * h);
    constexpr int kSegs = 24;
    for (int i = 0; i < kSegs; ++i) {
        const float n0 = static_cast<float>(i) / static_cast<float>(kSegs);
        const float n1 = static_cast<float>(i + 1) / static_cast<float>(kSegs);
        if (n0 >= live_norm) break;
        const float top_n = std::min(n1, live_norm);
        const float sy0 = std::floor(bar_max.y - top_n * h);
        const float sy1 = std::floor(bar_max.y - n0 * h);
        const int c_top = std::clamp(static_cast<int>(top_n * 255.0f), 0, 255);
        const int c_bot = std::clamp(static_cast<int>(n0 * 255.0f), 0, 255);
        if (sy1 > sy0) {
            dl->AddRectFilledMultiColor(
                ImVec2(bar_min.x + 1.0f, sy0), ImVec2(bar_max.x - 1.0f, sy1),
                lut256[c_top], lut256[c_top], lut256[c_bot], lut256[c_bot]
            );
        }
    }

    const float floor_norm = std::clamp(static_cast<float>((floor_db - db_min) / span), 0.0f, 1.0f);
    const float y_floor = std::floor(bar_max.y - floor_norm * h);
    dl->AddLineH(bar_min.x - dp(2.0f), bar_max.x + dp(2.0f), y_floor,
                 palette::U32(palette::TextPrimary, 0.65f), 1.5f);

    if (has_hold) {
        const float hold_norm = std::clamp(static_cast<float>((hold_peak_db - db_min) / span), 0.0f, 1.0f);
        const float y_hold = std::floor(bar_max.y - hold_norm * h);
        dl->AddLineH(bar_min.x, bar_max.x, y_hold, palette::U32(palette::Amber), 2.0f);
    }

    dl->AddRect(bar_min, bar_max, palette::PlotBorder32, dp(2.0f), 1.0f, ImDrawFlags_RoundAll);

    ImFont* f = mono_font ? mono_font : ImGui::GetFont();
    const float fs = font_dp(11.0f);
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.1f", live_peak_db);
    dl->AddText(f, fs, ImVec2(std::floor(bar_min.x - 1.0f), std::floor(bar_min.y - dp(14.0f))), palette::U32(palette::Cyan), buf);
    std::snprintf(buf, sizeof(buf), "N %.0f", floor_db);
    dl->AddText(f, fs, ImVec2(std::floor(bar_max.x + dp(5.0f)), std::floor(std::clamp(y_floor - dp(6.0f), bar_min.y, bar_max.y - dp(12.0f)))),
                palette::TextMuted32, buf);
    if (live_norm > 0.02f) {
        dl->AddLineH(bar_min.x, bar_max.x, y_live, IM_COL32(255, 255, 255, 220), 1.5f);
    }
}

} // namespace ui_obs
