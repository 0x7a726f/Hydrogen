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
                              ImGuiSliderFlags flags = 0) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, palette::BgWell);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, palette::Tint(accent, 0.24f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, palette::Tint(accent, 0.42f));
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, dp(4.0f));
    const bool changed = ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    if (v_max > v_min) {
        draw_slider_rail(label, (*v - v_min) / (v_max - v_min), accent);
    }
    return changed;
}

inline bool SliderIntAccent(const char* label, int* v, int v_min, int v_max,
                            const char* format = "%d", const ImVec4& accent = palette::Cyan,
                            ImGuiSliderFlags flags = 0) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, palette::BgWell);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, palette::Tint(accent, 0.24f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, palette::Tint(accent, 0.42f));
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, dp(4.0f));
    const bool changed = ImGui::SliderInt(label, v, v_min, v_max, format, flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

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
    bool   active = false;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
};

inline double calc_nice_step(double span, int target_ticks) {
    if (span <= 0.0 || target_ticks < 2) return 1.0;
    const double raw = span / static_cast<double>(target_ticks - 1);
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double nice = (norm < 1.5) ? 1.0 : (norm < 3.0) ? 2.0 : (norm < 7.0) ? 5.0 : 10.0;
    return nice * mag;
}

inline void handle_plot_zoom_pan(const ImVec2& plot_min, const ImVec2& plot_max,
                                 double def_xmin, double def_xmax,
                                 double def_ymin, double def_ymax,
                                 PlotZoomState& zoom,
                                 bool clamp_x_to_default = true,
                                 bool clamp_y_to_default = false) {
    const float pw = std::max(1.0f, plot_max.x - plot_min.x);
    const float ph = std::max(1.0f, plot_max.y - plot_min.y);
    const ImGuiIO& io = ImGui::GetIO();

    if (ImGui::IsItemHovered()) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);

        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            zoom.active = false;
            return;
        }

        const float wheel = (io.MouseWheel != 0.0f) ? io.MouseWheel : io.MouseWheelH;
        if (wheel != 0.0f &&
            io.MousePos.x >= plot_min.x && io.MousePos.x <= plot_max.x &&
            io.MousePos.y >= plot_min.y && io.MousePos.y <= plot_max.y) {
            if (!zoom.active) {
                zoom.x_min = def_xmin; zoom.x_max = def_xmax;
                zoom.y_min = def_ymin; zoom.y_max = def_ymax;
                zoom.active = true;
            }
            const double factor = std::pow(0.86, static_cast<double>(wheel));
            const bool zoom_x = !io.KeyCtrl;
            const bool zoom_y = !io.KeyShift;

            const double u = static_cast<double>(io.MousePos.x - plot_min.x) / pw;
            const double v = static_cast<double>(plot_max.y - io.MousePos.y) / ph;
            const double ax = zoom.x_min + u * (zoom.x_max - zoom.x_min);
            const double ay = zoom.y_min + v * (zoom.y_max - zoom.y_min);

            if (zoom_x) {
                double span = (zoom.x_max - zoom.x_min) * factor;
                const double def_span = std::max(1e-9, def_xmax - def_xmin);
                span = std::clamp(span, def_span * 0.002, clamp_x_to_default ? def_span : def_span * 2.0);
                zoom.x_min = ax - u * span;
                zoom.x_max = zoom.x_min + span;
                if (clamp_x_to_default) {
                    if (zoom.x_min < def_xmin) { zoom.x_max += (def_xmin - zoom.x_min); zoom.x_min = def_xmin; }
                    if (zoom.x_max > def_xmax) { zoom.x_min -= (zoom.x_max - def_xmax); zoom.x_max = def_xmax; }
                    zoom.x_min = std::max(def_xmin, zoom.x_min);
                    zoom.x_max = std::min(def_xmax, zoom.x_max);
                }
            }
            if (zoom_y) {
                double span = (zoom.y_max - zoom.y_min) * factor;
                const double def_span = std::max(1e-9, def_ymax - def_ymin);
                span = std::clamp(span, def_span * 0.01, clamp_y_to_default ? def_span : def_span * 4.0);
                zoom.y_min = ay - v * span;
                zoom.y_max = zoom.y_min + span;
                if (clamp_y_to_default) {
                    if (zoom.y_min < def_ymin) { zoom.y_max += (def_ymin - zoom.y_min); zoom.y_min = def_ymin; }
                    if (zoom.y_max > def_ymax) { zoom.y_min -= (zoom.y_max - def_ymax); zoom.y_max = def_ymax; }
                }
            }
        }
    }

    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
        if (!zoom.active) {
            zoom.x_min = def_xmin; zoom.x_max = def_xmax;
            zoom.y_min = def_ymin; zoom.y_max = def_ymax;
            zoom.active = true;
        }
        const double dx = -(static_cast<double>(io.MouseDelta.x) / pw) * (zoom.x_max - zoom.x_min);
        const double dy =  (static_cast<double>(io.MouseDelta.y) / ph) * (zoom.y_max - zoom.y_min);
        zoom.x_min += dx; zoom.x_max += dx;
        zoom.y_min += dy; zoom.y_max += dy;
        if (clamp_x_to_default) {
            const double span = zoom.x_max - zoom.x_min;
            if (zoom.x_min < def_xmin) { zoom.x_min = def_xmin; zoom.x_max = def_xmin + span; }
            if (zoom.x_max > def_xmax) { zoom.x_max = def_xmax; zoom.x_min = def_xmax - span; }
        }
        if (clamp_y_to_default) {
            const double span = zoom.y_max - zoom.y_min;
            if (zoom.y_min < def_ymin) { zoom.y_min = def_ymin; zoom.y_max = def_ymin + span; }
            if (zoom.y_max > def_ymax) { zoom.y_max = def_ymax; zoom.y_min = def_ymax - span; }
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

        float tick_h = dp(12.0f);
        {
            ScopedFont tick_f(mono_font, 12.0f);
            tick_h = ImGui::GetFontSize();
            const ImU32 text_col = palette::U32(palette::TextMuted);

            // Horizontal gridlines & nice Y ticks
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

                    char buf[32];
                    std::snprintf(buf, sizeof(buf), y_fmt, val);
                    const ImVec2 ts = ImGui::CalcTextSize(buf);
                    dl->AddText(ImVec2(std::floor(plot_min.x - ts.x - dp(6.0f)), std::floor(py - tick_h * 0.5f)), text_col, buf);
                }
            }

            // Vertical gridlines & nice X ticks
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

                    char buf[32];
                    std::snprintf(buf, sizeof(buf), x_fmt, val);
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

                char buf[40];
                std::snprintf(buf, sizeof(buf), fmt, val);
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

    template <typename YGetter>
    void decimate_trace(size_t n_points, double trace_x_min, double trace_x_max,
                        YGetter&& y_at, std::vector<ImVec2>& out_pts) const {
        out_pts.clear();
        if (n_points < 2 || plot_w <= 1.0f) return;

        const int num_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const double span = trace_x_max - trace_x_min;
        if (span <= 0.0) return;

        const double px_step = static_cast<double>(plot_w) / static_cast<double>(num_cols - 1);
        const double inv_step_x = static_cast<double>(n_points - 1) / span;
        const double inv_cols = 1.0 / static_cast<double>(num_cols - 1);

        out_pts.reserve(static_cast<size_t>(num_cols));
        for (int col = 0; col < num_cols; ++col) {
            const double x0 = trace_x_min + (static_cast<double>(col) * inv_cols) * span;
            const double x1 = trace_x_min + (static_cast<double>(col + 1) * inv_cols) * span;
            if (x1 < x_min || x0 > x_max) continue;

            const double cx0 = std::max(x0, x_min);
            const double cx1 = std::min(x1, x_max);
            int i0 = static_cast<int>(std::floor((cx0 - trace_x_min) * inv_step_x));
            int i1 = static_cast<int>(std::ceil((cx1 - trace_x_min) * inv_step_x));
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
        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const double span_x = (std::abs(trace_x_max - trace_x_min) < 1e-12) ? 1.0 : (trace_x_max - trace_x_min);

        dl->PushClipRect(plot_min, plot_max, true);

        if (n_points <= static_cast<size_t>(max_cols * 2)) {
            scratch.reserve(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                double x = trace_x_min + (static_cast<double>(i) / static_cast<double>(n_points - 1)) * span_x;
                double y = static_cast<double>(y_at(i));
                if (std::isfinite(y)) {
                    scratch.emplace_back(map_x(x), map_y(y));
                }
            }
        } else {
            scratch.reserve(static_cast<size_t>(max_cols) * 2);
            const double inv_n = 1.0 / static_cast<double>(n_points - 1);
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

            for (size_t i = 0; i < n_points; ++i) {
                double y = static_cast<double>(y_at(i));
                if (!std::isfinite(y)) continue;
                double x = trace_x_min + (static_cast<double>(i) * inv_n) * span_x;
                float px = map_x(x);
                float py = map_y(y);
                int col_idx = static_cast<int>(std::floor(px - plot_min.x));
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

        const float y_zero = std::clamp(map_y(y_ref), plot_min.y, plot_max.y);
        fill_scratch.clear();
        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const double x_pad = 0.05 * (x_max - x_min);

        if (n_points <= static_cast<size_t>(max_cols * 2)) {
            fill_scratch.reserve(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                if (x < x_min - x_pad || x > x_max + x_pad) continue;
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
            for (size_t i = 0; i < n_points; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                if (x < x_min - x_pad || x > x_max + x_pad) continue;
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
        scratch.clear();
        const int max_cols = std::max(2, static_cast<int>(std::ceil(plot_w)));
        const double x_pad = 0.05 * (x_max - x_min);
        dl->PushClipRect(plot_min, plot_max, true);

        if (n_points <= static_cast<size_t>(max_cols * 2)) {
            scratch.reserve(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                if (x < x_min - x_pad || x > x_max + x_pad) continue;
                scratch.emplace_back(map_x(x), map_y(y));
            }
        } else {
            scratch.reserve(static_cast<size_t>(max_cols) * 2);
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
            for (size_t i = 0; i < n_points; ++i) {
                const double x = static_cast<double>(x_at(i));
                const double y = static_cast<double>(y_at(i));
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                if (x < x_min - x_pad || x > x_max + x_pad) continue;
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
