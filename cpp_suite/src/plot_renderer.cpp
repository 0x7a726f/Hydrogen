#include "plot_renderer.h"
#include "utf8_path.h"

#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <mutex>

namespace astro {

// Simple 8x12 readable bitmap font for ASCII 32..126
static const uint8_t FONT_8X12[95][12] = {
    // Space 32
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // ! 33
    {0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00},
    // " 34
    {0x00,0x66,0x66,0x66,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // # 35
    {0x00,0x66,0x66,0xFF,0x66,0x66,0xFF,0x66,0x66,0x00,0x00,0x00},
    // $ 36
    {0x18,0x7E,0x18,0x3C,0x66,0x3C,0x18,0x66,0x3C,0x18,0x7E,0x18},
    // % 37
    {0x00,0xC6,0xCC,0x18,0x30,0x60,0xC6,0x00,0x00,0x00,0x00,0x00},
    // & 38
    {0x00,0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00,0x00,0x00,0x00},
    // ' 39
    {0x00,0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // ( 40
    {0x00,0x0C,0x18,0x30,0x30,0x30,0x30,0x18,0x0C,0x00,0x00,0x00},
    // ) 41
    {0x00,0x30,0x18,0x0C,0x0C,0x0C,0x0C,0x18,0x30,0x00,0x00,0x00},
    // * 42
    {0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00,0x00,0x00,0x00},
    // + 43
    {0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00,0x00},
    // , 44
    {0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30,0x00,0x00,0x00},
    // - 45
    {0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // . 46
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00},
    // / 47
    {0x00,0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00,0x00,0x00,0x00},
    // 0 48
    {0x00,0x3C,0x66,0xC3,0xC3,0xC3,0x66,0x3C,0x00,0x00,0x00,0x00},
    // 1 49
    {0x00,0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,0x00,0x00},
    // 2 50
    {0x00,0x7C,0xC6,0x0C,0x18,0x30,0x60,0xFF,0x00,0x00,0x00,0x00},
    // 3 51
    {0x00,0x7E,0x0C,0x18,0x3C,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // 4 52
    {0x00,0x0C,0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x00,0x00,0x00,0x00},
    // 5 53
    {0x00,0xFE,0xC0,0xFC,0x06,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // 6 54
    {0x00,0x38,0x60,0xC0,0xFC,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // 7 55
    {0x00,0xFE,0x06,0x0C,0x18,0x30,0x30,0x30,0x00,0x00,0x00,0x00},
    // 8 56
    {0x00,0x7C,0xC6,0xC6,0x7C,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // 9 57
    {0x00,0x7C,0xC6,0xC6,0x7E,0x06,0x0C,0x78,0x00,0x00,0x00,0x00},
    // : 58
    {0x00,0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00},
    // ; 59
    {0x00,0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x30,0x00,0x00,0x00},
    // < 60
    {0x00,0x06,0x18,0x60,0xC0,0x60,0x18,0x06,0x00,0x00,0x00,0x00},
    // = 61
    {0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // > 62
    {0x00,0x60,0x18,0x06,0x03,0x06,0x18,0x60,0x00,0x00,0x00,0x00},
    // ? 63
    {0x00,0x7C,0xC6,0x0C,0x18,0x18,0x00,0x18,0x00,0x00,0x00,0x00},
    // @ 64
    {0x00,0x7C,0xC6,0xDE,0xD6,0xD6,0xC0,0x7C,0x00,0x00,0x00,0x00},
    // A 65
    {0x00,0x38,0x6C,0xC6,0xFE,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    // B 66
    {0x00,0xFC,0x66,0x66,0x7C,0x66,0x66,0xFC,0x00,0x00,0x00,0x00},
    // C 67
    {0x00,0x3C,0x66,0xC0,0xC0,0xC0,0x66,0x3C,0x00,0x00,0x00,0x00},
    // D 68
    {0x00,0xF8,0x6C,0x66,0x66,0x66,0x6C,0xF8,0x00,0x00,0x00,0x00},
    // E 69
    {0x00,0xFE,0x62,0x68,0x78,0x68,0x62,0xFE,0x00,0x00,0x00,0x00},
    // F 70
    {0x00,0xFE,0x62,0x68,0x78,0x68,0x60,0xF0,0x00,0x00,0x00,0x00},
    // G 71
    {0x00,0x3C,0x66,0xC0,0xCE,0xC6,0x66,0x3A,0x00,0x00,0x00,0x00},
    // H 72
    {0x00,0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    // I 73
    {0x00,0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,0x00,0x00},
    // J 74
    {0x00,0x1E,0x0C,0x0C,0x0C,0xCC,0xCC,0x78,0x00,0x00,0x00,0x00},
    // K 75
    {0x00,0xE6,0x66,0x6C,0x78,0x6C,0x66,0xE6,0x00,0x00,0x00,0x00},
    // L 76
    {0x00,0xF0,0x60,0x60,0x60,0x62,0x66,0xFE,0x00,0x00,0x00,0x00},
    // M 77
    {0x00,0xC3,0xE7,0xFF,0xDB,0xC3,0xC3,0xC3,0x00,0x00,0x00,0x00},
    // N 78
    {0x00,0xC6,0xE6,0xF6,0xFE,0xDE,0xCE,0xC6,0x00,0x00,0x00,0x00},
    // O 79
    {0x00,0x3C,0x66,0xC3,0xC3,0xC3,0x66,0x3C,0x00,0x00,0x00,0x00},
    // P 80
    {0x00,0xFC,0x66,0x66,0x7C,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    // Q 81
    {0x00,0x3C,0x66,0xC3,0xC3,0xDB,0x6E,0x3C,0x06,0x00,0x00,0x00},
    // R 82
    {0x00,0xFC,0x66,0x66,0x7C,0x6C,0x66,0xE6,0x00,0x00,0x00,0x00},
    // S 83
    {0x00,0x7C,0xC6,0x60,0x38,0x0C,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // T 84
    {0x00,0xFF,0xDB,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    // U 85
    {0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    // V 86
    {0x00,0xC3,0xC3,0xC3,0x66,0x66,0x3C,0x18,0x00,0x00,0x00,0x00},
    // W 87
    {0x00,0xC3,0xC3,0xC3,0xDB,0xFF,0x66,0x66,0x00,0x00,0x00,0x00},
    // X 88
    {0x00,0xC3,0x66,0x3C,0x18,0x3C,0x66,0xC3,0x00,0x00,0x00,0x00},
    // Y 89
    {0x00,0xC3,0x66,0x3C,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    // Z 90
    {0x00,0xFE,0xC6,0x0C,0x18,0x30,0x63,0xFE,0x00,0x00,0x00,0x00},
    // [ 91
    {0x00,0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00,0x00,0x00,0x00},
    // \ 92
    {0x00,0xC0,0x60,0x30,0x18,0x0C,0x06,0x02,0x00,0x00,0x00,0x00},
    // ] 93
    {0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00,0x00,0x00,0x00},
    // ^ 94
    {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // _ 95
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00,0x00},
    // ` 96
    {0x00,0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // a 97
    {0x00,0x00,0x00,0x78,0x0C,0x7C,0xCC,0x76,0x00,0x00,0x00,0x00},
    // b 98
    {0x00,0xE0,0x60,0x7C,0x66,0x66,0x66,0xDC,0x00,0x00,0x00,0x00},
    // c 99
    {0x00,0x00,0x00,0x3C,0x66,0x60,0x66,0x3C,0x00,0x00,0x00,0x00},
    // d 100
    {0x00,0x1C,0x0C,0x7C,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00},
    // e 101
    {0x00,0x00,0x00,0x3C,0x66,0xFE,0x60,0x3C,0x00,0x00,0x00,0x00},
    // f 102
    {0x00,0x1C,0x36,0x30,0x7C,0x30,0x30,0x78,0x00,0x00,0x00,0x00},
    // g 103
    {0x00,0x00,0x00,0x76,0xCC,0xCC,0x7C,0x0C,0x78,0x00,0x00,0x00},
    // h 104
    {0x00,0xE0,0x60,0x6C,0x76,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    // i 105
    {0x00,0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    // j 106
    {0x00,0x0C,0x00,0x1C,0x0C,0x0C,0x0C,0xCC,0x78,0x00,0x00,0x00},
    // k 107
    {0x00,0xE0,0x60,0x66,0x6C,0x78,0x6C,0xE6,0x00,0x00,0x00,0x00},
    // l 108
    {0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    // m 109
    {0x00,0x00,0x00,0xFE,0xDB,0xDB,0xDB,0xDB,0x00,0x00,0x00,0x00},
    // n 110
    {0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00},
    // o 111
    {0x00,0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
    // p 112
    {0x00,0x00,0x00,0xDC,0x66,0x66,0x7C,0x60,0xF0,0x00,0x00,0x00},
    // q 113
    {0x00,0x00,0x00,0x76,0xCC,0xCC,0x7C,0x0C,0x1E,0x00,0x00,0x00},
    // r 114
    {0x00,0x00,0x00,0xDC,0x76,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    // s 115
    {0x00,0x00,0x00,0x7C,0xC0,0x78,0x06,0x7C,0x00,0x00,0x00,0x00},
    // t 116
    {0x00,0x10,0x30,0x7C,0x30,0x30,0x34,0x18,0x00,0x00,0x00,0x00},
    // u 117
    {0x00,0x00,0x00,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00},
    // v 118
    {0x00,0x00,0x00,0xC3,0xC3,0x66,0x3C,0x18,0x00,0x00,0x00,0x00},
    // w 119
    {0x00,0x00,0x00,0xC3,0xC3,0xDB,0xDB,0x66,0x00,0x00,0x00,0x00},
    // x 120
    {0x00,0x00,0x00,0xC3,0x66,0x3C,0x66,0xC3,0x00,0x00,0x00,0x00},
    // y 121
    {0x00,0x00,0x00,0xC6,0xC6,0xC6,0x7E,0x06,0x7C,0x00,0x00,0x00},
    // z 122
    {0x00,0x00,0x00,0xFE,0x8C,0x18,0x32,0xFE,0x00,0x00,0x00,0x00},
    // { 123
    {0x00,0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00,0x00,0x00,0x00},
    // | 124
    {0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
    // } 125
    {0x00,0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00,0x00,0x00,0x00},
    // ~ 126
    {0x00,0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}
};

PixelColor sample_colormap(ColormapType cmap, float t) {
    if (!std::isfinite(t)) t = 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);

    if (cmap == ColormapType::Classic) {
        // Jet / Classic piecewise linear
        float r = std::clamp(1.5f - std::abs(4.0f * t - 3.0f), 0.0f, 1.0f);
        float g = std::clamp(1.5f - std::abs(4.0f * t - 2.0f), 0.0f, 1.0f);
        float b = std::clamp(1.5f - std::abs(4.0f * t - 1.0f), 0.0f, 1.0f);
        return PixelColor(static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(b * 255));
    }

    struct Stop { uint8_t r, g, b; };
    static constexpr Stop viridis[5] = {{68, 1, 84}, {59, 82, 139}, {33, 145, 140}, {94, 201, 98}, {253, 231, 37}};
    static constexpr Stop turbo[5]   = {{48, 18, 59}, {40, 188, 235}, {164, 252, 60}, {251, 126, 33}, {122, 4, 3}};
    static constexpr Stop magma[5]   = {{0, 0, 4}, {81, 18, 124}, {183, 55, 121}, {252, 137, 97}, {252, 253, 191}};

    const Stop* stops = magma;
    if (cmap == ColormapType::Viridis) stops = viridis;
    else if (cmap == ColormapType::Turbo) stops = turbo;

    const float pos = t * 4.0f;
    const int idx = std::clamp(static_cast<int>(pos), 0, 3);
    const float frac = pos - static_cast<float>(idx);
    auto lerp_u8 = [&](uint8_t a, uint8_t b) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(a + (b - a) * frac, 0.0f, 255.0f));
    };
    return PixelColor(
        lerp_u8(stops[idx].r, stops[idx + 1].r),
        lerp_u8(stops[idx].g, stops[idx + 1].g),
        lerp_u8(stops[idx].b, stops[idx + 1].b)
    );
}

SoftwareCanvas::SoftwareCanvas(int width, int height, PixelColor bg_color)
    : width_(std::max(0, width)),
      height_(std::max(0, height)),
      pixels_(static_cast<size_t>(std::max(0, width)) * static_cast<size_t>(std::max(0, height)) * 4ULL) {
    clear(bg_color);
}

void SoftwareCanvas::clear(PixelColor color) {
    if (width_ <= 0 || height_ <= 0 || pixels_.empty()) return;
    uint32_t packed = (static_cast<uint32_t>(color.a) << 24)
                    | (static_cast<uint32_t>(color.b) << 16)
                    | (static_cast<uint32_t>(color.g) << 8)
                    | static_cast<uint32_t>(color.r);
    uint32_t* dst = reinterpret_cast<uint32_t*>(pixels_.data());
    std::fill_n(dst, static_cast<size_t>(width_) * static_cast<size_t>(height_), packed);
}

void SoftwareCanvas::set_pixel(int x, int y, PixelColor color) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) * 4ULL;
    pixels_[idx + 0] = color.r;
    pixels_[idx + 1] = color.g;
    pixels_[idx + 2] = color.b;
    pixels_[idx + 3] = color.a;
}

void SoftwareCanvas::blend_pixel(int x, int y, PixelColor color) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) * 4ULL;
    const uint32_t a = color.a;
    const uint32_t inv_a = 255u - a;
    pixels_[idx + 0] = static_cast<uint8_t>((color.r * a + pixels_[idx + 0] * inv_a + 127u) / 255u);
    pixels_[idx + 1] = static_cast<uint8_t>((color.g * a + pixels_[idx + 1] * inv_a + 127u) / 255u);
    pixels_[idx + 2] = static_cast<uint8_t>((color.b * a + pixels_[idx + 2] * inv_a + 127u) / 255u);
}

static float fpart(float x) { return x - std::floor(x); }
static float rfpart(float x) { return 1.0f - fpart(x); }

void SoftwareCanvas::draw_line(int x0_i, int y0_i, int x1_i, int y1_i, PixelColor color, int thickness) {
    if (width_ <= 0 || height_ <= 0 || thickness <= 0) return;

    // Pure horizontal or vertical lines: draw crisp spans respecting exact even/odd thickness and alpha
    if (x0_i == x1_i || y0_i == y1_i) {
        int x_min = std::min(x0_i, x1_i);
        int x_max = std::max(x0_i, x1_i);
        int y_min = std::min(y0_i, y1_i);
        int y_max = std::max(y0_i, y1_i);
        const int r_lo = -(thickness / 2);
        const int r_hi = (thickness - 1) / 2;
        const bool opaque = (color.a == 255);

        if (x0_i == x1_i) {
            y_min = std::max(0, y_min);
            y_max = std::min(height_ - 1, y_max);
            for (int y = y_min; y <= y_max; ++y) {
                for (int rx = r_lo; rx <= r_hi; ++rx) {
                    if (opaque) set_pixel(x0_i + rx, y, color);
                    else blend_pixel(x0_i + rx, y, color);
                }
            }
        } else {
            x_min = std::max(0, x_min);
            x_max = std::min(width_ - 1, x_max);
            for (int ry = r_lo; ry <= r_hi; ++ry) {
                int py = y0_i + ry;
                if (py < 0 || py >= height_) continue;
                for (int x = x_min; x <= x_max; ++x) {
                    if (opaque) set_pixel(x, py, color);
                    else blend_pixel(x, py, color);
                }
            }
        }
        return;
    }

    // High-quality Xiaolin Wu anti-aliased line algorithm with sub-pixel coverage
    auto draw_single_aa = [this](float x0, float y0, float x1, float y1, PixelColor col) {
        bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
        if (steep) {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }

        float dx = x1 - x0;
        float dy = y1 - y0;
        float gradient = (std::abs(dx) < 1e-6f) ? 1.0f : (dy / dx);

        auto plot = [this, steep, &col](int x, int y, float brightness) {
            if (brightness <= 0.001f) return;
            uint8_t a = static_cast<uint8_t>(std::clamp(brightness * col.a, 0.0f, 255.0f));
            if (steep) {
                blend_pixel(y, x, PixelColor(col.r, col.g, col.b, a));
            } else {
                blend_pixel(x, y, PixelColor(col.r, col.g, col.b, a));
            }
        };

        float xend = std::round(x0);
        float yend = y0 + gradient * (xend - x0);
        float xgap = rfpart(x0 + 0.5f);
        int xpxl1 = static_cast<int>(xend);
        int ypxl1 = static_cast<int>(std::floor(yend));

        plot(xpxl1, ypxl1, rfpart(yend) * xgap);
        plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap);
        float intery = yend + gradient;

        xend = std::round(x1);
        yend = y1 + gradient * (xend - x1);
        xgap = fpart(x1 + 0.5f);
        int xpxl2 = static_cast<int>(xend);
        int ypxl2 = static_cast<int>(std::floor(yend));

        plot(xpxl2, ypxl2, rfpart(yend) * xgap);
        plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap);

        const int max_axis = steep ? height_ : width_;
        const int loop_start = std::max(0, xpxl1 + 1);
        const int loop_end = std::min(max_axis, xpxl2);
        if (loop_start > xpxl1 + 1) {
            intery += gradient * static_cast<float>(loop_start - (xpxl1 + 1));
        }
        for (int x = loop_start; x < loop_end; ++x) {
            plot(x, static_cast<int>(std::floor(intery)), rfpart(intery));
            plot(x, static_cast<int>(std::floor(intery)) + 1, fpart(intery));
            intery += gradient;
        }
    };

    float x0 = static_cast<float>(x0_i);
    float y0 = static_cast<float>(y0_i);
    float x1 = static_cast<float>(x1_i);
    float y1 = static_cast<float>(y1_i);

    if (thickness <= 1) {
        draw_single_aa(x0, y0, x1, y1, color);
    } else {
        float dx = x1 - x0;
        float dy = y1 - y0;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-4f) {
            set_pixel(x0_i, y0_i, color);
            return;
        }
        float nx = -dy / len;
        float ny = dx / len;

        int rad = (thickness - 1);
        for (int step = -rad; step <= rad; ++step) {
            float offset = step * 0.5f;
            draw_single_aa(x0 + nx * offset, y0 + ny * offset, x1 + nx * offset, y1 + ny * offset, color);
        }
    }
}

void SoftwareCanvas::draw_rect(int x, int y, int w, int h, PixelColor color, int thickness) {
    draw_line(x, y, x + w, y, color, thickness);
    draw_line(x + w, y, x + w, y + h, color, thickness);
    draw_line(x + w, y + h, x, y + h, color, thickness);
    draw_line(x, y + h, x, y, color, thickness);
}

void SoftwareCanvas::fill_rect(int x, int y, int w, int h, PixelColor color) {
    int x_start = std::max(0, x);
    int x_end = std::min(width_, x + w);
    int y_start = std::max(0, y);
    int y_end = std::min(height_, y + h);
    if (x_start >= x_end || y_start >= y_end) return;

    const size_t span = static_cast<size_t>(x_end - x_start);
    const uint32_t packed = (static_cast<uint32_t>(color.a) << 24)
                          | (static_cast<uint32_t>(color.b) << 16)
                          | (static_cast<uint32_t>(color.g) << 8)
                          | static_cast<uint32_t>(color.r);
    uint32_t* px32 = reinterpret_cast<uint32_t*>(pixels_.data());
    for (int cy = y_start; cy < y_end; ++cy) {
        std::fill_n(&px32[static_cast<size_t>(cy) * width_ + x_start], span, packed);
    }
}

void SoftwareCanvas::fill_rect_alpha(int x, int y, int w, int h, PixelColor color) {
    int x_start = std::max(0, x);
    int x_end = std::min(width_, x + w);
    int y_start = std::max(0, y);
    int y_end = std::min(height_, y + h);
    if (x_start >= x_end || y_start >= y_end) return;
    if (color.a == 255) {
        fill_rect(x, y, w, h, color);
        return;
    }
    if (color.a == 0) return;

    const uint32_t a = color.a;
    const uint32_t inv_a = 255u - a;
    const uint32_t ra = static_cast<uint32_t>(color.r) * a + 127u;
    const uint32_t ga = static_cast<uint32_t>(color.g) * a + 127u;
    const uint32_t ba = static_cast<uint32_t>(color.b) * a + 127u;

    for (int cy = y_start; cy < y_end; ++cy) {
        uint8_t* row = &pixels_[(static_cast<size_t>(cy) * width_ + x_start) * 4];
        for (int cx = x_start; cx < x_end; ++cx, row += 4) {
            row[0] = static_cast<uint8_t>((ra + row[0] * inv_a) / 255u);
            row[1] = static_cast<uint8_t>((ga + row[1] * inv_a) / 255u);
            row[2] = static_cast<uint8_t>((ba + row[2] * inv_a) / 255u);
        }
    }
}

void SoftwareCanvas::draw_circle(int cx, int cy, int radius, PixelColor color, bool fill) {
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            int dist_sq = x * x + y * y;
            if (fill) {
                if (dist_sq <= radius * radius) {
                    set_pixel(cx + x, cy + y, color);
                }
            } else {
                if (dist_sq <= radius * radius && dist_sq >= (radius - 1) * (radius - 1)) {
                    set_pixel(cx + x, cy + y, color);
                }
            }
        }
    }
}

void SoftwareCanvas::draw_text(int x, int y, const std::string& text, PixelColor color, int scale) {
    if (width_ <= 0 || height_ <= 0) return;
    scale = std::max(1, scale);
    int cur_x = x;
    int cur_y = y;

    for (char ch : text) {
        if (ch == '\n') {
            cur_x = x;
            cur_y += 14 * scale;
            continue;
        }
        if (ch < 32 || ch > 126) ch = '?';
        int char_idx = ch - 32;

        for (int row = 0; row < 12; ++row) {
            uint8_t row_bits = FONT_8X12[char_idx][row];
            for (int col = 0; col < 8; ++col) {
                if ((row_bits >> (7 - col)) & 1) {
                    if (scale == 1) {
                        set_pixel(cur_x + col, cur_y + row, color);
                    } else {
                        fill_rect(cur_x + col * scale, cur_y + row * scale, scale, scale, color);
                    }
                }
            }
        }
        cur_x += 8 * scale;
    }
}

int SoftwareCanvas::measure_text_width(const std::string& text, int scale) const {
    scale = std::max(1, scale);
    size_t max_len = 0;
    size_t cur_len = 0;
    for (char ch : text) {
        if (ch == '\n') {
            max_len = std::max(max_len, cur_len);
            cur_len = 0;
        } else {
            cur_len++;
        }
    }
    max_len = std::max(max_len, cur_len);
    return static_cast<int>(max_len) * 8 * scale;
}

void SoftwareCanvas::render_waterfall(
    const float* matrix,
    int rows,
    int cols,
    int dest_x,
    int dest_y,
    int dest_w,
    int dest_h,
    float vmin,
    float vmax,
    ColormapType cmap
) {
    if (!matrix || rows <= 0 || cols <= 0 || dest_w <= 0 || dest_h <= 0 || width_ <= 0 || height_ <= 0) return;
    if (!std::isfinite(vmin)) vmin = -1.0f;
    if (!std::isfinite(vmax)) vmax = vmin + 1.0f;
    float range = vmax - vmin;
    if (range <= 1e-6f) range = 1.0f;
    const float inv_range_255 = 255.0f / range;

    // Precompute 256-entry packed RGBA LUT for cmap
    uint32_t lut[256];
    for (int i = 0; i < 256; ++i) {
        PixelColor c = sample_colormap(cmap, static_cast<float>(i) / 255.0f);
        lut[i] = (static_cast<uint32_t>(c.a) << 24)
               | (static_cast<uint32_t>(c.b) << 16)
               | (static_cast<uint32_t>(c.g) << 8)
               | static_cast<uint32_t>(c.r);
    }

    // Precompute column max-pool bin ranges [c0, c1) for peak-preserving downsampling
    std::vector<int> col_start(static_cast<size_t>(dest_w));
    std::vector<int> col_end(static_cast<size_t>(dest_w));
    for (int px = 0; px < dest_w; ++px) {
        int c0 = std::clamp(static_cast<int>((static_cast<double>(px) * cols) / dest_w), 0, cols - 1);
        int c1 = std::clamp(static_cast<int>((static_cast<double>(px + 1) * cols) / dest_w), c0 + 1, cols);
        col_start[static_cast<size_t>(px)] = c0;
        col_end[static_cast<size_t>(px)] = c1;
    }

    const int px_start = std::max(0, -dest_x);
    const int px_end = std::min(dest_w, width_ - dest_x);
    const int py_start = std::max(0, -dest_y);
    const int py_end = std::min(dest_h, height_ - dest_y);
    if (px_start >= px_end || py_start >= py_end) return;

    uint32_t* canvas_px = reinterpret_cast<uint32_t*>(pixels_.data());

    for (int py = py_start; py < py_end; ++py) {
        // Flip Y so row 0 (t_min) sits at bottom and row (rows-1) (t_max) sits at top
        const int inv_py = dest_h - 1 - py;
        const int r0 = std::clamp(static_cast<int>((static_cast<double>(inv_py) * rows) / dest_h), 0, rows - 1);
        const int r1 = std::clamp(static_cast<int>((static_cast<double>(inv_py + 1) * rows) / dest_h), r0 + 1, rows);
        uint32_t* dst_row = &canvas_px[static_cast<size_t>(dest_y + py) * static_cast<size_t>(width_)];

        for (int px = px_start; px < px_end; ++px) {
            const int c0 = col_start[static_cast<size_t>(px)];
            const int c1 = col_end[static_cast<size_t>(px)];
            float peak = -1e30f;
            bool any_finite = false;
            for (int sr = r0; sr < r1; ++sr) {
                const float* row_ptr = &matrix[static_cast<size_t>(sr) * static_cast<size_t>(cols)];
                for (int sc = c0; sc < c1; ++sc) {
                    const float v = row_ptr[sc];
                    if (std::isfinite(v) && (!any_finite || v > peak)) {
                        peak = v;
                        any_finite = true;
                    }
                }
            }
            const float val = any_finite ? peak : vmin;
            const float norm = (val - vmin) * inv_range_255;
            const int idx = std::isfinite(norm)
                ? static_cast<int>(std::clamp(norm, 0.0f, 255.0f))
                : 0;
            dst_row[static_cast<size_t>(dest_x + px)] = lut[idx];
        }
    }
}

static void compute_waterfall_percentiles(std::vector<float>& vals, float& vmin, float& vmax) {
    vals.erase(std::remove_if(vals.begin(), vals.end(), [](float v) { return !std::isfinite(v); }), vals.end());
    if (vals.empty()) {
        vmin = -1.0f;
        vmax = 1.0f;
        return;
    }
    const size_t n = vals.size();
    if (n == 1) {
        vmin = vals[0] - 0.5f;
        vmax = vals[0] + 0.5f;
        return;
    }
    const size_t i_lo = (n * 5) / 100;
    const size_t i_hi = std::min(n - 1, (n * 995) / 1000);
    std::nth_element(vals.begin(), vals.begin() + i_lo, vals.end());
    vmin = vals[i_lo];
    if (i_hi > i_lo) {
        std::nth_element(vals.begin() + i_lo + 1, vals.begin() + i_hi, vals.end());
        vmax = vals[i_hi] + 0.3f;
    } else {
        vmax = vmin + 0.3f;
    }
    if (vmax <= vmin + 1e-4f) {
        vmax = vmin + 1.0f;
    }
}

bool SoftwareCanvas::save_png(const std::string& filepath) const {
    if (width_ <= 0 || height_ <= 0 || pixels_.empty() || filepath.empty()) return false;
    std::filesystem::path p = astro_fs::utf8_path(filepath);
    if (!p.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    const size_t n_px = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    std::vector<uint8_t> rgb(n_px * 3);
    const uint8_t* src = pixels_.data();
    uint8_t* dst = rgb.data();
    for (size_t i = 0; i < n_px; ++i) {
        dst[3 * i + 0] = src[4 * i + 0];
        dst[3 * i + 1] = src[4 * i + 1];
        dst[3 * i + 2] = src[4 * i + 2];
    }
    static std::once_flag s_stbi_init;
    std::call_once(s_stbi_init, []() {
        stbi_write_png_compression_level = 3;
    });
    int res = stbi_write_png(filepath.c_str(), width_, height_, 3, rgb.data(), width_ * 3);
    return (res != 0);
}

// ------------------------------------------------------------------------------------------------
// PlotRenderer Publication Figures
// ------------------------------------------------------------------------------------------------

static inline double safe_val(double v) {
    return std::isfinite(v) ? v : 0.0;
}

static inline int safe_px(double v) {
    if (!std::isfinite(v)) return 0;
    // Clamp in double space BEFORE llround to prevent UB/wrap-around on huge values
    const double clamped = std::clamp(v, -100000.0, 100000.0);
    return static_cast<int>(std::llround(clamped));
}

static inline void sanitize_axis_range(double& lo, double& hi, double def_lo, double def_hi) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi || !std::isfinite(hi - lo)) {
        lo = def_lo;
        hi = def_hi;
    }
}

static void draw_text_right(SoftwareCanvas& canvas, int right_x, int y, const std::string& text, PixelColor color, int scale = 1) {
    const int tw = canvas.measure_text_width(text, scale);
    canvas.draw_text(std::max(2, right_x - tw), y, text, color, scale);
}

static void draw_text_center(SoftwareCanvas& canvas, int center_x, int y, const std::string& text, PixelColor color, int scale = 1) {
    const int tw = canvas.measure_text_width(text, scale);
    const int max_x = std::max(2, canvas.width() - tw - 2);
    canvas.draw_text(std::clamp(center_x - tw / 2, 2, max_x), y, text, color, scale);
}

static std::string format_signed(double val, int decimals) {
    if (!std::isfinite(val)) val = 0.0;
    const double eps = 0.5 * std::pow(10.0, -decimals);
    if (std::abs(val) < eps) val = 0.0;
    char buf[32];
    if (val > 0.0) {
        std::snprintf(buf, sizeof(buf), "+%.*f", decimals, val);
    } else {
        std::snprintf(buf, sizeof(buf), "%.*f", decimals, val);
    }
    return std::string(buf);
}

static std::string format_fixed(double val, int decimals) {
    if (!std::isfinite(val)) val = 0.0;
    const double eps = 0.5 * std::pow(10.0, -decimals);
    if (std::abs(val) < eps) val = 0.0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, val);
    return std::string(buf);
}

static void draw_clipped_line(
    SoftwareCanvas& canvas,
    int left,
    int top,
    int right,
    int bottom,
    int x0,
    int y0,
    int x1,
    int y1,
    PixelColor color,
    int thickness
) {
    if (left >= right || top >= bottom) return;
    // Liang-Barsky parametric segment clipping against [left, right] x [top, bottom]
    // Promote to double BEFORE subtracting to avoid signed 32-bit int overflow
    double t0 = 0.0;
    double t1 = 1.0;
    const double dx = static_cast<double>(x1) - static_cast<double>(x0);
    const double dy = static_cast<double>(y1) - static_cast<double>(y0);
    auto clip = [&](double p, double q) -> bool {
        if (std::abs(p) < 1e-12) return q >= 0.0;
        const double r = q / p;
        if (p < 0.0) {
            if (r > t1) return false;
            if (r > t0) t0 = r;
        } else {
            if (r < t0) return false;
            if (r < t1) t1 = r;
        }
        return true;
    };
    if (!clip(-dx, static_cast<double>(x0) - static_cast<double>(left))) return;
    if (!clip(dx, static_cast<double>(right) - static_cast<double>(x0))) return;
    if (!clip(-dy, static_cast<double>(y0) - static_cast<double>(top))) return;
    if (!clip(dy, static_cast<double>(bottom) - static_cast<double>(y0))) return;

    const int cx0 = safe_px(x0 + t0 * dx);
    const int cy0 = safe_px(y0 + t0 * dy);
    const int cx1 = safe_px(x0 + t1 * dx);
    const int cy1 = safe_px(y0 + t1 * dy);
    canvas.draw_line(cx0, cy0, cx1, cy1, color, thickness);
}

bool PlotRenderer::render_calibrated_spectrum(
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& vel_kms,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const std::string& output_path
) {
    const size_t n_pts = std::min({freqs_mhz.size(), vel_kms.size(), calibrated_k.size(), calibrated_db.size()});
    if (n_pts < 2) return false;
    const int W = 1200;
    const int H = 640;
    SoftwareCanvas canvas(W, H, PixelColor(13, 17, 23)); // #0d1117

    // Plot area margins (extra top headroom so frequency ticks and top label never collide)
    const int left = 96;
    const int right = W - 44;
    const int top = 96;
    const int bottom = H - 70;
    const int plot_w = right - left;
    const int plot_h = bottom - top;

    // Background of plot box
    canvas.fill_rect(left, top, plot_w, plot_h, PixelColor(22, 27, 34)); // #161b22
    canvas.draw_rect(left, top, plot_w, plot_h, PixelColor(48, 54, 61), 1);

    // Zoom limits: v in [-200, 200]
    const double v_min = -200.0;
    const double v_max = 200.0;

    const size_t edge_bins = std::min<size_t>(
        n_pts / 2,
        static_cast<size_t>(static_cast<double>(n_pts) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)));
    double k_max = 0.5;
    double k_min = -0.5;
    for (size_t i = edge_bins; i + edge_bins < n_pts; ++i) {
        if (std::isfinite(vel_kms[i]) && std::isfinite(calibrated_k[i]) &&
            vel_kms[i] >= v_min && vel_kms[i] <= v_max) {
            k_max = std::max(k_max, calibrated_k[i]);
            k_min = std::min(k_min, calibrated_k[i]);
        }
    }
    k_min = std::min(-1.0, k_min * 1.15 - 0.5);
    k_max = std::max(1.0, k_max * 1.32 + 1.0);
    if (!(k_max > k_min + 1e-6)) {
        k_min = -1.0;
        k_max = 2.0;
    }

    auto map_x = [&](double v) {
        return safe_px(left + (v - v_min) / (v_max - v_min) * plot_w);
    };
    auto map_y = [&](double k) {
        return safe_px(bottom - (k - k_min) / (k_max - k_min) * plot_h);
    };

    // Vertical Grid lines & Velocity/Frequency ticks
    for (double v = -180.0; v <= 180.0; v += 60.0) {
        const int gx = map_x(v);
        canvas.draw_line(gx, top, gx, bottom, PixelColor(38, 46, 56), 1);
        draw_text_center(canvas, gx, bottom + 12, format_signed(v, 0), PixelColor(139, 148, 158), 1);

        // Top axis sky frequency (MHz)
        const double f_mhz = HYDROGEN_REST_FREQ_MHZ * (1.0 - v / SPEED_OF_LIGHT_KMS);
        draw_text_center(canvas, gx, top - 22, format_fixed(f_mhz, 3), PixelColor(164, 176, 190), 1);
    }

    // Horizontal Y grid lines & right-aligned Kelvin ticks
    for (int t = 0; t < 5; ++t) {
        const double k_val = k_min + (k_max - k_min) * (static_cast<double>(t) / 4.0);
        const int gy = map_y(k_val);
        if (gy >= top && gy <= bottom) {
            canvas.draw_line(left, gy, right, gy, PixelColor(38, 46, 56), 1);
            draw_text_right(canvas, left - 8, gy - 5, format_fixed(k_val, 1), PixelColor(139, 148, 158), 1);
        }
    }

    // Zero Kelvin line
    const int y0 = map_y(0.0);
    if (y0 >= top && y0 <= bottom) {
        canvas.draw_line(left, y0, right, y0, PixelColor(139, 148, 158), 1);
    }

    // Rest frame v = 0 line
    const int x0 = map_x(0.0);
    if (x0 >= left && x0 <= right) {
        canvas.draw_line(x0, top, x0, bottom, PixelColor(116, 185, 255), 1);
    }

    // Single-pass vertical gradient fill under positive emission curve (alpha 10 -> 58)
    std::vector<int> col_min_y(static_cast<size_t>(plot_w + 1), bottom + 1);
    for (size_t i = 1; i < n_pts; ++i) {
        const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
        const double k_a = calibrated_k[i - 1], k_b = calibrated_k[i];
        if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(k_a) || !std::isfinite(k_b)) continue;
        if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
        const int xa = map_x(v_a), xb = map_x(v_b);
        const int ya = map_y(k_a), yb = map_y(k_b);

        const int x_step_start = std::max(left, std::min(xa, xb));
        const int x_step_end = std::min(right, std::max(xa, xb));
        for (int sx = x_step_start; sx <= x_step_end; ++sx) {
            const float frac = (xa != xb) ? static_cast<float>(sx - xa) / static_cast<float>(xb - xa) : 0.0f;
            const int cur_y = static_cast<int>(ya + frac * static_cast<float>(yb - ya));
            int& slot = col_min_y[static_cast<size_t>(sx - left)];
            slot = std::min(slot, cur_y);
        }
    }
    const int y0_clamped = std::clamp(y0, top, bottom);
    for (int dx = 0; dx <= plot_w; ++dx) {
        const int cur_y = std::max(top, col_min_y[static_cast<size_t>(dx)]);
        if (cur_y < y0_clamped) {
            const float span = static_cast<float>(std::max(1, y0_clamped - cur_y));
            for (int sy = cur_y; sy < y0_clamped; ++sy) {
                const float grad = 1.0f - static_cast<float>(sy - cur_y) / span;
                const uint8_t a = static_cast<uint8_t>(10.0f + grad * 48.0f);
                canvas.blend_pixel(left + dx, sy, PixelColor(0, 242, 254, a));
            }
        }
    }

    // Calibrated spectrum line (Cyan #00f2fe) - clipped to plot box
    for (size_t i = 1; i < n_pts; ++i) {
        const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
        const double k_a = calibrated_k[i - 1], k_b = calibrated_k[i];
        if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(k_a) || !std::isfinite(k_b)) continue;
        if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
        draw_clipped_line(canvas, left, top, right, bottom,
                          map_x(v_a), map_y(k_a),
                          map_x(v_b), map_y(k_b),
                          PixelColor(0, 242, 254), 2);
    }

    // Line profile fit overlay + multi-component curves (DoubleGaussian / TripleGaussian)
    if (metrics.gauss_fit.success && metrics.gauss_fit.fitted_curve.size() >= n_pts) {
        if (metrics.gauss_fit.has_secondary &&
            metrics.gauss_fit.component1_curve.size() >= n_pts &&
            metrics.gauss_fit.component2_curve.size() >= n_pts) {
            for (size_t i = 1; i < n_pts; ++i) {
                const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
                if (!std::isfinite(v_a) || !std::isfinite(v_b)) continue;
                if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
                if (std::isfinite(metrics.gauss_fit.component1_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component1_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component1_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component1_curve[i]),
                                      PixelColor(251, 146, 60), 1);
                }
                if (std::isfinite(metrics.gauss_fit.component2_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component2_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component2_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component2_curve[i]),
                                      PixelColor(167, 139, 250), 1);
                }
                if (metrics.gauss_fit.has_tertiary && metrics.gauss_fit.component3_curve.size() >= n_pts &&
                    std::isfinite(metrics.gauss_fit.component3_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component3_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component3_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component3_curve[i]),
                                      PixelColor(16, 185, 129), 1);
                }
            }
        }
        for (size_t i = 1; i < n_pts; ++i) {
            const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
            const double f_a = metrics.gauss_fit.fitted_curve[i - 1];
            const double f_b = metrics.gauss_fit.fitted_curve[i];
            if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(f_a) || !std::isfinite(f_b)) continue;
            if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
            draw_clipped_line(canvas, left, top, right, bottom,
                              map_x(v_a), map_y(f_a),
                              map_x(v_b), map_y(f_b),
                              PixelColor(255, 118, 117), 2);
        }
        if (std::isfinite(metrics.gauss_fit.fwhm_left_kms) &&
            std::isfinite(metrics.gauss_fit.fwhm_right_kms) &&
            std::isfinite(metrics.gauss_fit.fwhm_half_temp_k) &&
            metrics.gauss_fit.fwhm_right_kms > metrics.gauss_fit.fwhm_left_kms &&
            metrics.gauss_fit.fwhm_half_temp_k > 0.05) {
            draw_clipped_line(canvas, left, top, right, bottom,
                              map_x(metrics.gauss_fit.fwhm_left_kms), map_y(metrics.gauss_fit.fwhm_half_temp_k),
                              map_x(metrics.gauss_fit.fwhm_right_kms), map_y(metrics.gauss_fit.fwhm_half_temp_k),
                              PixelColor(245, 158, 11), 1);
        }
    }

    // Peak marker circle (clamped inside plot box)
    if (std::isfinite(metrics.peak_vel_kms) && std::isfinite(metrics.peak_temp_k)) {
        const int px = std::clamp(map_x(metrics.peak_vel_kms), left, right);
        const int py = std::clamp(map_y(metrics.peak_temp_k), top, bottom);
        canvas.draw_circle(px, py, 6, PixelColor(255, 71, 87), true);
        canvas.draw_circle(px, py, 7, PixelColor(255, 255, 255), false);
    }

    // Title & Axis Labels (clean vertical separation from top frequency ticks)
    canvas.draw_text(left, 16, "Calibrated Hydrogen Line (HI) Spectrum - Galactic 21cm Analysis", PixelColor(255, 255, 255), 2);
    draw_text_center(canvas, left + plot_w / 2, bottom + 42, "Topocentric Radial Velocity v_topo [Radio Def] (km/s)", PixelColor(201, 209, 217), 1);
    draw_text_center(canvas, left + plot_w / 2, 48, "Topocentric Sky Frequency (MHz)", PixelColor(164, 176, 190), 1);
    canvas.draw_text(14, top - 22, "TA (K)", PixelColor(0, 242, 254), 1);

    // Dynamically sized HUD Metrics Card Box (accommodates TripleGaussian / BusyFunction & baseline names cleanly)
    const double snr_mf = safe_val(metrics.snr_matched > 0.0 ? metrics.snr_matched : metrics.snr);
    std::stringstream hud;
    hud << "Target Line: Neutral Hydrogen (HI 21cm)\n"
        << "Baseline:    " << baseline_method_display_name(config.baseline_method) << "\n"
        << "Line Model:  " << line_analysis_method_display_name(config.line_method) << "\n"
        << "Detection:   " << (metrics.is_detected ? "CONFIRMED" : "NO DETECTION")
        << " (SNR_mf " << format_fixed(snr_mf, 1) << " sigma)\n"
        << "Obs Peak:    " << format_fixed(safe_val(metrics.peak_freq_mhz), 4) << " MHz ("
        << format_signed(safe_val(metrics.peak_vel_kms), 2) << " km/s)\n"
        << "Brightness:  " << format_fixed(safe_val(metrics.peak_temp_k), 2) << " K ("
        << format_signed(safe_val(metrics.peak_power_db), 3) << " dB)\n"
        << "Noise Floor: " << format_fixed(safe_val(metrics.noise_std_k), 3) << " K (SNR_1bin "
        << format_fixed(safe_val(metrics.snr_single_bin > 0.0 ? metrics.snr_single_bin : metrics.snr), 1) << ")\n"
        << "FWHM / W50:  " << format_fixed(safe_val(metrics.fwhm_kms), 1) << " / "
        << format_fixed(safe_val(metrics.w50_kms), 1) << " km/s (W20: "
        << format_fixed(safe_val(metrics.w20_kms), 1) << ")\n"
        << "Flux W:      " << format_fixed(safe_val(metrics.integrated_flux_k_kms), 1) << " K*km/s\n"
        << "N_HI Column: " << std::scientific << std::setprecision(2) << safe_val(metrics.column_density_cm2) << " cm^-2";

    const std::string hud_str = hud.str();
    int num_lines = 1;
    for (char c : hud_str) {
        if (c == '\n') ++num_lines;
    }
    const int hud_w = std::min(plot_w - 32, std::max(360, canvas.measure_text_width(hud_str, 1) + 24));
    const int hud_h = std::min(plot_h - 32, num_lines * 14 + 20);
    const bool peak_on_left = (safe_val(metrics.peak_vel_kms) < 0.5 * (v_min + v_max));
    const int hud_x = peak_on_left ? (right - hud_w - 16) : (left + 16);
    const int hud_y = top + 16;
    canvas.fill_rect_alpha(hud_x, hud_y, hud_w, hud_h, PixelColor(30, 39, 46, 232));
    canvas.draw_rect(hud_x, hud_y, hud_w, hud_h,
                     metrics.is_detected ? PixelColor(16, 185, 129) : PixelColor(0, 210, 211), 1);
    canvas.draw_text(hud_x + 10, hud_y + 10, hud_str, PixelColor(241, 242, 246), 1);

    return canvas.save_png(output_path);
}

bool PlotRenderer::render_raw_and_baseline(
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& raw_power,
    const std::vector<double>& baseline_power,
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const std::string& output_path
) {
    const size_t n = std::min({freqs_mhz.size(), raw_power.size(), baseline_power.size()});
    if (n < 2) return false;

    const double f_min = freqs_mhz.front();
    const double f_max = freqs_mhz[n - 1];
    if (!std::isfinite(f_min) || !std::isfinite(f_max) || std::abs(f_max - f_min) < 1e-12) return false;

    const int W = 1200;
    const int H = 820;
    SoftwareCanvas canvas(W, H, PixelColor(13, 17, 23));

    const int left = 96, right = W - 40;
    const int top1 = 84, bottom1 = 428;
    const int top2 = 496, bottom2 = H - 70;
    const int plot_w = right - left;

    canvas.fill_rect(left, top1, plot_w, bottom1 - top1, PixelColor(22, 27, 34));
    canvas.draw_rect(left, top1, plot_w, bottom1 - top1, PixelColor(48, 54, 61), 1);

    canvas.fill_rect(left, top2, plot_w, bottom2 - top2, PixelColor(22, 27, 34));
    canvas.draw_rect(left, top2, plot_w, bottom2 - top2, PixelColor(48, 54, 61), 1);

    const size_t skirt_bins = std::min<size_t>(
        (n > 2) ? (n / 2 - 1) : 0,
        static_cast<size_t>(static_cast<double>(n) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)));

    std::vector<double> raw_db(n), base_db(n), flat_db(n);
    double y1_min = 1e9, y1_max = -1e9;
    double y2_pb_min = 1e9, y2_pb_max = -1e9;
    double off_line_sq = 0.0;
    size_t off_line_cnt = 0;
    const double hi_half_mhz = config.hi_protection_half_width_khz / 1e3;

    for (size_t i = 0; i < n; ++i) {
        const double rp = std::isfinite(raw_power[i]) ? std::max(raw_power[i], 1e-12) : 1e-12;
        const double bp = std::isfinite(baseline_power[i]) ? std::max(baseline_power[i], 1e-12) : 1e-12;
        raw_db[i] = 10.0 * std::log10(rp);
        base_db[i] = 10.0 * std::log10(bp);
        flat_db[i] = raw_db[i] - base_db[i];

        // Focus Y1 & Y2 scaling on inner passband so passband ripples aren't crushed by steep outer IF skirts
        if (i >= skirt_bins && i + skirt_bins < n) {
            y1_min = std::min({y1_min, raw_db[i], base_db[i]});
            y1_max = std::max({y1_max, raw_db[i], base_db[i]});
            y2_pb_min = std::min(y2_pb_min, flat_db[i]);
            y2_pb_max = std::max(y2_pb_max, flat_db[i]);

            if (std::isfinite(freqs_mhz[i]) &&
                std::abs(freqs_mhz[i] - HYDROGEN_REST_FREQ_MHZ) > hi_half_mhz) {
                off_line_sq += flat_db[i] * flat_db[i];
                ++off_line_cnt;
            }
        }
    }
    if (y1_min > y1_max) {
        for (size_t i = 0; i < n; ++i) {
            y1_min = std::min({y1_min, raw_db[i], base_db[i]});
            y1_max = std::max({y1_max, raw_db[i], base_db[i]});
        }
    }
    const double y1_span = std::max(0.5, y1_max - y1_min);
    y1_min -= y1_span * 0.16;
    y1_max += y1_span * 0.18;
    sanitize_axis_range(y1_min, y1_max, -100.0, 0.0);

    if (y2_pb_min > y2_pb_max) {
        y2_pb_min = -0.35;
        y2_pb_max = 0.45;
    }
    const double y2_span = std::max(0.4, y2_pb_max - y2_pb_min);
    const double y2_pad = std::max(0.10, y2_span * 0.18);
    double y2_min = std::min(-0.20, y2_pb_min - y2_pad);
    double y2_max = std::max(0.20, y2_pb_max + y2_pad);
    sanitize_axis_range(y2_min, y2_max, -1.0, 1.0);

    const double off_line_rms_db = (std::isfinite(metrics.noise_std_db) && metrics.noise_std_db > 0.0)
        ? metrics.noise_std_db
        : ((off_line_cnt > 0) ? std::sqrt(off_line_sq / static_cast<double>(off_line_cnt)) : 0.0);

    auto map_x = [&](double f) {
        return safe_px(left + (f - f_min) / (f_max - f_min) * plot_w);
    };
    auto map_y1 = [&](double p) {
        return safe_px(bottom1 - (p - y1_min) / (y1_max - y1_min) * (bottom1 - top1));
    };
    auto map_y2 = [&](double r) {
        return safe_px(bottom2 - (r - y2_min) / (y2_max - y2_min) * (bottom2 - top2));
    };

    // Shaded outer filter skirts (safely clamped index)
    const int x_pb_start = std::clamp(map_x(freqs_mhz[skirt_bins]), left, right);
    const int x_pb_end = std::clamp(map_x(freqs_mhz[n - 1 - skirt_bins]), left, right);
    if (x_pb_start > left) {
        canvas.fill_rect_alpha(left, top1, x_pb_start - left, bottom1 - top1, PixelColor(87, 96, 111, 70));
        canvas.fill_rect_alpha(left, top2, x_pb_start - left, bottom2 - top2, PixelColor(87, 96, 111, 70));
    }
    if (right > x_pb_end) {
        canvas.fill_rect_alpha(x_pb_end, top1, right - x_pb_end, bottom1 - top1, PixelColor(87, 96, 111, 70));
        canvas.fill_rect_alpha(x_pb_end, top2, right - x_pb_end, bottom2 - top2, PixelColor(87, 96, 111, 70));
    }

    // Protected HI window shaded
    const int x_hi_l = std::clamp(map_x(HYDROGEN_REST_FREQ_MHZ - hi_half_mhz), left, right);
    const int x_hi_r = std::clamp(map_x(HYDROGEN_REST_FREQ_MHZ + hi_half_mhz), left, right);
    if (x_hi_r > x_hi_l) {
        canvas.fill_rect_alpha(x_hi_l, top1, x_hi_r - x_hi_l, bottom1 - top1, PixelColor(16, 172, 132, 42));
        canvas.fill_rect_alpha(x_hi_l, top2, x_hi_r - x_hi_l, bottom2 - top2, PixelColor(16, 172, 132, 42));
    }

    // Vertical gridlines & numeric X-axis Frequency (MHz) ticks across both panels
    constexpr int kNumXTicks = 7;
    for (int t = 0; t < kNumXTicks; ++t) {
        const double frac = static_cast<double>(t) / static_cast<double>(kNumXTicks - 1);
        const double f_tick = f_min + frac * (f_max - f_min);
        const int gx = map_x(f_tick);
        if (gx >= left && gx <= right) {
            canvas.draw_line(gx, top1, gx, bottom1, PixelColor(38, 46, 56), 1);
            canvas.draw_line(gx, top2, gx, bottom2, PixelColor(38, 46, 56), 1);
            canvas.draw_line(gx, bottom2, gx, bottom2 + 5, PixelColor(139, 148, 158), 1);
            draw_text_center(canvas, gx, bottom2 + 10, format_fixed(f_tick, 3), PixelColor(139, 148, 158), 1);
        }
    }

    // Horizontal gridlines & right-aligned Y1 (Power dB) ticks
    for (int t = 0; t < 5; ++t) {
        const double p_val = y1_min + (y1_max - y1_min) * (static_cast<double>(t) / 4.0);
        const int gy = map_y1(p_val);
        if (gy >= top1 && gy <= bottom1) {
            canvas.draw_line(left, gy, right, gy, PixelColor(38, 46, 56), 1);
            canvas.draw_line(left - 4, gy, left, gy, PixelColor(139, 148, 158), 1);
            draw_text_right(canvas, left - 8, gy - 5, format_fixed(p_val, 1), PixelColor(139, 148, 158), 1);
        }
    }

    // Horizontal gridlines & right-aligned Y2 (Residual Delta dB) ticks
    for (int t = 0; t < 5; ++t) {
        const double r_val = y2_min + (y2_max - y2_min) * (static_cast<double>(t) / 4.0);
        const int gy = map_y2(r_val);
        if (gy >= top2 && gy <= bottom2) {
            canvas.draw_line(left, gy, right, gy, PixelColor(38, 46, 56), 1);
            canvas.draw_line(left - 4, gy, left, gy, PixelColor(139, 148, 158), 1);
            draw_text_right(canvas, left - 8, gy - 5, format_signed(r_val, 2), PixelColor(139, 148, 158), 1);
        }
    }

    // DC notch vertical line (LO tuned center frequency)
    const double fc_mhz = config.center_freq_hz / 1e6;
    const int x_dc = map_x(fc_mhz);
    if (x_dc >= left && x_dc <= right) {
        canvas.draw_line(x_dc, top1, x_dc, bottom1, PixelColor(255, 71, 87), 2);
        canvas.draw_line(x_dc, top2, x_dc, bottom2, PixelColor(255, 71, 87), 1);
    }

    // HI line vertical marker
    const int x_hi = map_x(HYDROGEN_REST_FREQ_MHZ);
    if (x_hi >= left && x_hi <= right) {
        canvas.draw_line(x_hi, top1, x_hi, bottom1, PixelColor(0, 210, 211), 1);
        canvas.draw_line(x_hi, top2, x_hi, bottom2, PixelColor(0, 210, 211), 1);
    }

    // Top panel: Raw Spectrum (Sky blue) & Baseline Model (Orange) - clipped to plot box
    for (size_t i = 1; i < n; ++i) {
        if (!std::isfinite(freqs_mhz[i - 1]) || !std::isfinite(freqs_mhz[i])) continue;
        const int xa = map_x(freqs_mhz[i - 1]);
        const int xb = map_x(freqs_mhz[i]);
        draw_clipped_line(canvas, left, top1, right, bottom1,
                          xa, map_y1(raw_db[i - 1]), xb, map_y1(raw_db[i]), PixelColor(84, 160, 255), 1);
    }
    for (size_t i = 1; i < n; ++i) {
        if (!std::isfinite(freqs_mhz[i - 1]) || !std::isfinite(freqs_mhz[i])) continue;
        const int xa = map_x(freqs_mhz[i - 1]);
        const int xb = map_x(freqs_mhz[i]);
        draw_clipped_line(canvas, left, top1, right, bottom1,
                          xa, map_y1(base_db[i - 1]), xb, map_y1(base_db[i]), PixelColor(255, 159, 67), 2);
    }

    // Bottom panel: Flattened Residuals (Emerald green) + +/-1 sigma off-line RMS envelope
    const int y2_zero = std::clamp(map_y2(0.0), top2, bottom2);
    canvas.draw_line(left, y2_zero, right, y2_zero, PixelColor(139, 148, 158), 1);
    if (off_line_rms_db > 1e-5) {
        const int y_sig_hi = map_y2(+off_line_rms_db);
        const int y_sig_lo = map_y2(-off_line_rms_db);
        if (y_sig_hi > top2 && y_sig_hi < bottom2) {
            for (int x = left; x < right; x += 10) {
                canvas.draw_line(x, y_sig_hi, std::min(right, x + 5), y_sig_hi, PixelColor(0, 210, 211), 1);
            }
        }
        if (y_sig_lo > top2 && y_sig_lo < bottom2) {
            for (int x = left; x < right; x += 10) {
                canvas.draw_line(x, y_sig_lo, std::min(right, x + 5), y_sig_lo, PixelColor(0, 210, 211), 1);
            }
        }
    }
    for (size_t i = 1; i < n; ++i) {
        if (!std::isfinite(freqs_mhz[i - 1]) || !std::isfinite(freqs_mhz[i])) continue;
        const int xa = map_x(freqs_mhz[i - 1]);
        const int xb = map_x(freqs_mhz[i]);
        draw_clipped_line(canvas, left, top2, right, bottom2,
                          xa, map_y2(flat_db[i - 1]), xb, map_y2(flat_db[i]), PixelColor(29, 209, 161), 1);
    }

    // Top-right compact color legend pill in top passband subplot
    {
        const int leg_w = 278;
        const int leg_h = 24;
        const int leg_x = right - leg_w - 12;
        const int leg_y = top1 + 8;
        canvas.fill_rect_alpha(leg_x, leg_y, leg_w, leg_h, PixelColor(15, 20, 25, 220));
        canvas.draw_rect(leg_x, leg_y, leg_w, leg_h, PixelColor(48, 54, 61), 1);
        canvas.draw_line(leg_x + 10, leg_y + 12, leg_x + 24, leg_y + 12, PixelColor(84, 160, 255), 2);
        canvas.draw_text(leg_x + 28, leg_y + 8, "Raw PSD", PixelColor(210, 218, 226), 1);
        canvas.draw_line(leg_x + 96, leg_y + 12, leg_x + 110, leg_y + 12, PixelColor(255, 159, 67), 2);
        canvas.draw_text(leg_x + 114, leg_y + 8, "Baseline", PixelColor(210, 218, 226), 1);
        canvas.draw_line(leg_x + 186, leg_y + 12, leg_x + 200, leg_y + 12, PixelColor(0, 210, 211), 2);
        canvas.draw_text(leg_x + 204, leg_y + 8, "HI 21cm", PixelColor(210, 218, 226), 1);
    }

    // Re-stroke borders cleanly on top
    canvas.draw_rect(left, top1, plot_w, bottom1 - top1, PixelColor(48, 54, 61), 1);
    canvas.draw_rect(left, top2, plot_w, bottom2 - top2, PixelColor(48, 54, 61), 1);

    // Titles, Subtitles & Axis Labels
    const std::string base_name = baseline_method_display_name(config.baseline_method);
    canvas.draw_text(left, 16, "Bandpass & Baseline Model - " + base_name, PixelColor(255, 255, 255), 2);

    std::ostringstream sub;
    sub << "Baseline: " << base_name
        << "  |  Off-line RMS: " << format_fixed(off_line_rms_db, 4) << " dB"
        << " (" << format_fixed(safe_val(metrics.off_line_rms_k > 0.0 ? metrics.off_line_rms_k : metrics.noise_std_k), 3) << " K)"
        << "  |  Quality: " << format_fixed(safe_val(metrics.baseline_quality_ratio), 2) << "x thermal";
    canvas.draw_text(left, 48, sub.str(), PixelColor(164, 176, 190), 1);

    canvas.draw_text(left + 14, top1 + 10, "Raw Averaged Spectrum (Blue) vs Baseline Model (Orange)", PixelColor(210, 218, 226), 1);
    canvas.draw_text(left + 14, top2 + 10, "Flattened Residual Spectrum (P_raw - P_base, dB, +/-1 sigma dashed)", PixelColor(29, 209, 161), 1);

    canvas.draw_text(8, top1 - 20, "Power (dB)", PixelColor(84, 160, 255), 1);
    canvas.draw_text(8, top2 - 20, "Delta (dB)", PixelColor(29, 209, 161), 1);
    draw_text_center(canvas, left + plot_w / 2, bottom2 + 34, "Frequency (MHz)", PixelColor(201, 209, 217), 1);

    return canvas.save_png(output_path);
}

bool PlotRenderer::render_doppler_profile(
    const std::vector<double>& vel_kms,
    const std::vector<double>& calibrated_k,
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const std::string& output_path
) {
    const size_t n_pts = std::min(vel_kms.size(), calibrated_k.size());
    if (n_pts < 2) return false;
    const int W = 1000;
    const int H = 560;
    SoftwareCanvas canvas(W, H, PixelColor(13, 17, 23));

    const int left = 96, right = W - 44;
    const int top = 72, bottom = H - 72;
    const int plot_w = right - left;
    const int plot_h = bottom - top;

    canvas.fill_rect(left, top, plot_w, plot_h, PixelColor(22, 27, 34));
    canvas.draw_rect(left, top, plot_w, plot_h, PixelColor(48, 54, 61), 1);

    const double v_min = -250.0;
    const double v_max = 250.0;
    const size_t edge_bins = std::min<size_t>(
        n_pts / 2,
        static_cast<size_t>(static_cast<double>(n_pts) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)));
    double k_max = 0.5, k_min = -0.5;
    for (size_t i = edge_bins; i + edge_bins < n_pts; ++i) {
        if (std::isfinite(vel_kms[i]) && std::isfinite(calibrated_k[i]) &&
            vel_kms[i] >= v_min && vel_kms[i] <= v_max) {
            k_max = std::max(k_max, calibrated_k[i]);
            k_min = std::min(k_min, calibrated_k[i]);
        }
    }
    k_min = std::min(-1.0, k_min * 1.15 - 0.5);
    k_max = std::max(1.0, k_max * 1.32 + 1.0);
    if (!(k_max > k_min + 1e-6)) {
        k_min = -1.0;
        k_max = 2.0;
    }

    auto map_x = [&](double v) {
        return safe_px(left + (v - v_min) / (v_max - v_min) * plot_w);
    };
    auto map_y = [&](double k) {
        return safe_px(bottom - (k - k_min) / (k_max - k_min) * plot_h);
    };

    // Grid lines (X velocity & right-aligned Y Kelvin ticks)
    for (double v = -200.0; v <= 200.0; v += 100.0) {
        const int gx = map_x(v);
        if (gx >= left && gx <= right) {
            canvas.draw_line(gx, top, gx, bottom, PixelColor(38, 46, 56), 1);
            canvas.draw_line(gx, bottom, gx, bottom + 5, PixelColor(139, 148, 158), 1);
            draw_text_center(canvas, gx, bottom + 10, format_signed(v, 0), PixelColor(139, 148, 158), 1);
        }
    }
    for (int t = 0; t < 5; ++t) {
        const double k_val = k_min + (k_max - k_min) * (static_cast<double>(t) / 4.0);
        const int gy = map_y(k_val);
        if (gy >= top && gy <= bottom) {
            canvas.draw_line(left, gy, right, gy, PixelColor(38, 46, 56), 1);
            canvas.draw_line(left - 4, gy, left, gy, PixelColor(139, 148, 158), 1);
            draw_text_right(canvas, left - 8, gy - 5, format_fixed(k_val, 1), PixelColor(139, 148, 158), 1);
        }
    }

    // Zero velocity & rest frame
    const int x0 = std::clamp(map_x(0.0), left, right);
    canvas.draw_line(x0, top, x0, bottom, PixelColor(254, 202, 87), 1);
    const int y0 = map_y(0.0);
    if (y0 >= top && y0 <= bottom) {
        canvas.draw_line(left, y0, right, y0, PixelColor(139, 148, 158), 1);
    }

    // Single-pass vertical gradient fill above 0 K
    std::vector<int> col_min_y(static_cast<size_t>(plot_w + 1), bottom + 1);
    for (size_t i = 1; i < n_pts; ++i) {
        const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
        const double k_a = calibrated_k[i - 1], k_b = calibrated_k[i];
        if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(k_a) || !std::isfinite(k_b)) continue;
        if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
        const int xa = map_x(v_a), xb = map_x(v_b);
        const int ya = map_y(k_a), yb = map_y(k_b);
        const int xs = std::max(left, std::min(xa, xb));
        const int xe = std::min(right, std::max(xa, xb));
        for (int sx = xs; sx <= xe; ++sx) {
            const float frac = (xa != xb) ? static_cast<float>(sx - xa) / static_cast<float>(xb - xa) : 0.0f;
            const int cy = static_cast<int>(ya + frac * static_cast<float>(yb - ya));
            int& slot = col_min_y[static_cast<size_t>(sx - left)];
            slot = std::min(slot, cy);
        }
    }
    const int y0_clamped = std::clamp(y0, top, bottom);
    for (int dx = 0; dx <= plot_w; ++dx) {
        const int cy = std::max(top, col_min_y[static_cast<size_t>(dx)]);
        if (cy < y0_clamped) {
            const float span = static_cast<float>(std::max(1, y0_clamped - cy));
            for (int sy = cy; sy < y0_clamped; ++sy) {
                const float grad = 1.0f - static_cast<float>(sy - cy) / span;
                const uint8_t a = static_cast<uint8_t>(10.0f + grad * 48.0f);
                canvas.blend_pixel(left + dx, sy, PixelColor(72, 219, 251, a));
            }
        }
    }

    // Spectrum curve (Electric blue #48dbfb) - clipped to plot box
    for (size_t i = 1; i < n_pts; ++i) {
        const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
        const double k_a = calibrated_k[i - 1], k_b = calibrated_k[i];
        if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(k_a) || !std::isfinite(k_b)) continue;
        if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
        draw_clipped_line(canvas, left, top, right, bottom,
                          map_x(v_a), map_y(k_a),
                          map_x(v_b), map_y(k_b),
                          PixelColor(72, 219, 251), 2);
    }

    // Line profile fit + multi-component curves - verified vector sizes and clipped to plot box
    if (metrics.gauss_fit.success && metrics.gauss_fit.fitted_curve.size() >= n_pts) {
        if (metrics.gauss_fit.has_secondary &&
            metrics.gauss_fit.component1_curve.size() >= n_pts &&
            metrics.gauss_fit.component2_curve.size() >= n_pts) {
            for (size_t i = 1; i < n_pts; ++i) {
                const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
                if (!std::isfinite(v_a) || !std::isfinite(v_b)) continue;
                if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
                if (std::isfinite(metrics.gauss_fit.component1_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component1_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component1_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component1_curve[i]),
                                      PixelColor(251, 146, 60), 1);
                }
                if (std::isfinite(metrics.gauss_fit.component2_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component2_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component2_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component2_curve[i]),
                                      PixelColor(167, 139, 250), 1);
                }
                if (metrics.gauss_fit.has_tertiary && metrics.gauss_fit.component3_curve.size() >= n_pts &&
                    std::isfinite(metrics.gauss_fit.component3_curve[i - 1]) &&
                    std::isfinite(metrics.gauss_fit.component3_curve[i])) {
                    draw_clipped_line(canvas, left, top, right, bottom,
                                      map_x(v_a), map_y(metrics.gauss_fit.component3_curve[i - 1]),
                                      map_x(v_b), map_y(metrics.gauss_fit.component3_curve[i]),
                                      PixelColor(16, 185, 129), 1);
                }
            }
        }
        for (size_t i = 1; i < n_pts; ++i) {
            const double v_a = vel_kms[i - 1], v_b = vel_kms[i];
            const double f_a = metrics.gauss_fit.fitted_curve[i - 1];
            const double f_b = metrics.gauss_fit.fitted_curve[i];
            if (!std::isfinite(v_a) || !std::isfinite(v_b) || !std::isfinite(f_a) || !std::isfinite(f_b)) continue;
            if (std::max(v_a, v_b) < v_min || std::min(v_a, v_b) > v_max) continue;
            draw_clipped_line(canvas, left, top, right, bottom,
                              map_x(v_a), map_y(f_a),
                              map_x(v_b), map_y(f_b),
                              PixelColor(255, 107, 107), 2);
        }
    }

    // Peak marker (clamped inside plot box)
    if (std::isfinite(metrics.peak_vel_kms) && std::isfinite(metrics.peak_temp_k)) {
        const int px = std::clamp(map_x(metrics.peak_vel_kms), left, right);
        const int py = std::clamp(map_y(metrics.peak_temp_k), top, bottom);
        canvas.draw_circle(px, py, 6, PixelColor(255, 71, 87), true);
    }

    canvas.draw_text(left, 22, "Galactic Neutral Hydrogen Radial Velocity Profile", PixelColor(255, 255, 255), 2);
    draw_text_center(canvas, left + plot_w / 2, bottom + 36, "Topocentric Radial Velocity v_topo (km/s)", PixelColor(201, 209, 217), 1);
    canvas.draw_text(14, top - 24, "TA (K)", PixelColor(72, 219, 251), 1);

    return canvas.save_png(output_path);
}

bool PlotRenderer::render_waterfall_plot(
    const std::vector<float>& waterfall_data,
    int rows,
    int cols,
    const std::vector<double>& waterfall_times,
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& baseline_power,
    const AstroConfig& config,
    const std::string& output_path
) {
    if (rows <= 0 || cols <= 0 ||
        waterfall_data.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols) ||
        freqs_mhz.size() < static_cast<size_t>(cols) ||
        baseline_power.size() < static_cast<size_t>(cols)) {
        return false;
    }

    const int W = 1200;
    const int H = 620;
    SoftwareCanvas canvas(W, H, PixelColor(13, 17, 23));

    // Generous left/right margins (>= 115 px right for colorbar + dB labels without clipping)
    const int left = 96;
    const int right = W - 115;
    const int top = 74;
    const int bottom = H - 72;
    const int plot_w = right - left;
    const int plot_h = bottom - top;

    // Zoom frequency span: +/- 600 kHz around HI line (fallback to full band if outside range)
    const double f_center = HYDROGEN_REST_FREQ_MHZ;
    const double f_span = 0.6; // MHz
    const double f_min = f_center - f_span;
    const double f_max = f_center + f_span;

    int c_start = -1, c_end = -1;
    for (int c = 0; c < cols; ++c) {
        if (std::isfinite(freqs_mhz[static_cast<size_t>(c)]) && freqs_mhz[static_cast<size_t>(c)] >= f_min) {
            c_start = c;
            break;
        }
    }
    for (int c = cols - 1; c >= 0; --c) {
        if (std::isfinite(freqs_mhz[static_cast<size_t>(c)]) && freqs_mhz[static_cast<size_t>(c)] <= f_max) {
            c_end = c;
            break;
        }
    }
    if (c_start < 0 || c_end < 0 || c_start > c_end) {
        c_start = 0;
        c_end = cols - 1;
    }
    if (c_start == c_end && cols > 1) {
        if (c_end + 1 < cols) ++c_end;
        else --c_start;
    }
    const int sub_cols = std::max(1, c_end - c_start + 1);

    // Compute relative dB: 10 * log10(waterfall / baseline)
    std::vector<float> sub_matrix(static_cast<size_t>(rows) * static_cast<size_t>(sub_cols));
    std::vector<float> inv_base(static_cast<size_t>(sub_cols));
    for (int sc = 0; sc < sub_cols; ++sc) {
        const double bp = baseline_power[static_cast<size_t>(c_start + sc)];
        inv_base[static_cast<size_t>(sc)] = 1.0f / static_cast<float>(std::isfinite(bp) ? std::max(1e-12, bp) : 1e-12);
    }
    for (int r = 0; r < rows; ++r) {
        const float* wf_row = &waterfall_data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c_start)];
        float* dst_row = &sub_matrix[static_cast<size_t>(r) * static_cast<size_t>(sub_cols)];
        for (int sc = 0; sc < sub_cols; ++sc) {
            const float val = std::isfinite(wf_row[sc]) ? wf_row[sc] : 0.0f;
            dst_row[sc] = 10.0f * std::log10(std::max(1e-6f, val * inv_base[static_cast<size_t>(sc)]));
        }
    }

    if (sub_matrix.empty()) return false;
    std::vector<float> scratch_vals = sub_matrix;
    float vmin = -1.0f, vmax = 1.0f;
    compute_waterfall_percentiles(scratch_vals, vmin, vmax);

    canvas.render_waterfall(sub_matrix.data(), rows, sub_cols, left, top, plot_w, plot_h, vmin, vmax, ColormapType::Magma);
    canvas.draw_rect(left, top, plot_w, plot_h, PixelColor(48, 54, 61), 1);

    // Numeric X-axis Frequency (MHz) tick labels & HI rest marker
    const double f_sub_min = safe_val(freqs_mhz[static_cast<size_t>(c_start)]);
    const double f_sub_max = safe_val(freqs_mhz[static_cast<size_t>(c_end)]);
    const double f_sub_span = f_sub_max - f_sub_min;
    if (std::abs(f_sub_span) > 1e-9) {
        const int x_hi = safe_px(left + (HYDROGEN_REST_FREQ_MHZ - f_sub_min) / f_sub_span * plot_w);
        if (x_hi >= left && x_hi <= right) {
            canvas.draw_line(x_hi, top, x_hi, bottom, PixelColor(0, 242, 254), 1);
        }
        constexpr int kNumXTicks = 7;
        for (int t = 0; t < kNumXTicks; ++t) {
            const double frac = static_cast<double>(t) / static_cast<double>(kNumXTicks - 1);
            const double f_tick = f_sub_min + frac * f_sub_span;
            const int gx = safe_px(left + frac * plot_w);
            canvas.draw_line(gx, bottom, gx, bottom + 5, PixelColor(139, 148, 158), 1);
            draw_text_center(canvas, gx, bottom + 10, format_fixed(f_tick, 3), PixelColor(139, 148, 158), 1);
        }
    } else {
        draw_text_center(canvas, left + plot_w / 2, bottom + 10, format_fixed(f_sub_min, 3), PixelColor(139, 148, 158), 1);
    }

    // Numeric Y-axis Time (s) tick labels using waterfall_times (row 0 at bottom, row rows-1 at top)
    double t_start = 0.0;
    double t_end = static_cast<double>(rows) * std::max(0.1, config.waterfall_dt_s);
    if (!waterfall_times.empty()) {
        const double tf = safe_val(waterfall_times.front());
        const double tb = safe_val(waterfall_times.back());
        if (rows == 1 || tb <= tf) {
            t_start = 0.0;
            t_end = std::max(tb, std::max(0.1, config.waterfall_dt_s));
        } else {
            t_start = tf;
            t_end = tb;
        }
    }
    constexpr int kNumYTicks = 5;
    const int time_dec = ((t_end - t_start) < 20.0) ? 1 : 0;
    for (int t = 0; t < kNumYTicks; ++t) {
        const double frac = static_cast<double>(t) / static_cast<double>(kNumYTicks - 1);
        const double tv = t_start + frac * (t_end - t_start);
        const int gy = safe_px(bottom - frac * plot_h);
        canvas.draw_line(left - 5, gy, left, gy, PixelColor(139, 148, 158), 1);
        draw_text_right(canvas, left - 8, gy - 5, format_fixed(tv, time_dec) + "s", PixelColor(139, 148, 158), 1);
    }

    // Colorbar on right with safe signed formatting and "dB" unit header
    const int cb_x = right + 16;
    const int cb_w = 18;
    for (int y = 0; y < plot_h; ++y) {
        const float t = 1.0f - static_cast<float>(y) / static_cast<float>(std::max(1, plot_h - 1));
        const PixelColor col = sample_colormap(ColormapType::Magma, t);
        canvas.draw_line(cb_x, top + y, cb_x + cb_w, top + y, col, 1);
    }
    canvas.draw_rect(cb_x, top, cb_w, plot_h, PixelColor(48, 54, 61), 1);
    canvas.draw_text(cb_x, top - 18, "dB", PixelColor(201, 209, 217), 1);

    const float vmid = 0.5f * (vmin + vmax);
    canvas.draw_text(cb_x + cb_w + 8, top - 2, format_signed(vmax, 1), PixelColor(200, 200, 200), 1);
    canvas.draw_text(cb_x + cb_w + 8, top + plot_h / 2 - 5, format_signed(vmid, 1), PixelColor(164, 176, 190), 1);
    canvas.draw_text(cb_x + cb_w + 8, bottom - 10, format_signed(vmin, 1), PixelColor(200, 200, 200), 1);

    canvas.draw_text(left, 22, "Time-Frequency Waterfall (Calibrated Spectrogram)", PixelColor(255, 255, 255), 2);
    draw_text_center(canvas, left + plot_w / 2, bottom + 36, "Frequency (MHz)", PixelColor(201, 209, 217), 1);
    canvas.draw_text(14, top - 20, "Time (s)", PixelColor(201, 209, 217), 1);

    return canvas.save_png(output_path);
}

bool PlotRenderer::render_master_dashboard(
    const std::vector<double>& freqs_mhz,
    const std::vector<double>& vel_kms,
    const std::vector<double>& raw_power,
    const std::vector<double>& baseline_power,
    const std::vector<double>& calibrated_k,
    const std::vector<double>& calibrated_db,
    const std::vector<float>& waterfall_data,
    int wf_rows,
    int wf_cols,
    const std::vector<double>& waterfall_times,
    const AstrophysicalMetrics& metrics,
    const AstroConfig& config,
    const std::string& filepath,
    const std::string& output_path,
    double duration_s_override
) {
    // Guard all spectral vectors including calibrated_db (allow empty waterfall for graceful Q4 placeholder)
    const size_t n = std::min({freqs_mhz.size(), vel_kms.size(), raw_power.size(),
                               baseline_power.size(), calibrated_k.size(), calibrated_db.size()});
    if (n < 2) return false;

    const bool has_waterfall =
        (wf_rows > 0 && wf_cols > 0 &&
         waterfall_data.size() >= static_cast<size_t>(wf_rows) * static_cast<size_t>(wf_cols) &&
         static_cast<size_t>(wf_cols) <= n);

    const int W = 1600;
    const int H = 1000;
    SoftwareCanvas canvas(W, H, PixelColor(13, 17, 23));

    // Header banner (title scale=2, subtitle scale=1 to prevent horizontal overflow)
    const std::string fname_utf8 = astro_fs::path_to_utf8(astro_fs::utf8_path(filepath).filename());
    const double dur_s = (std::isfinite(duration_s_override) && duration_s_override >= 0.0)
        ? duration_s_override
        : (waterfall_times.empty() ? 0.0 : safe_val(waterfall_times.back()));
    const double snr_disp = safe_val(metrics.snr_matched > 0.0 ? metrics.snr_matched : metrics.snr);
    std::stringstream subtitle;
    subtitle << "File: " << fname_utf8
             << "  |  Duration: " << format_fixed(dur_s, 1) << "s (" << format_fixed(dur_s / 60.0, 2) << " min)"
             << "  |  LO: " << format_fixed(safe_val(config.center_freq_hz / 1e6), 4) << " MHz"
             << "  |  Rate: " << format_fixed(safe_val(config.sample_rate_hz / 1e6), 2) << " MSPS"
             << "  |  " << (metrics.is_detected ? "DETECTED" : "NO DETECTION")
             << " (SNR_mf " << format_fixed(snr_disp, 1) << " sigma)";

    canvas.draw_text(40, 16, "RADIOTELESCOPE HYDROGEN LINE ANALYSIS DASHBOARD", PixelColor(255, 255, 255), 2);
    canvas.draw_text(40, 48, subtitle.str(), PixelColor(164, 176, 190), 1);

    // 4 Quadrants:
    // Q1 (top-left): (A) Full Bandpass & Baseline Model
    // Q2 (top-right): (B) Calibrated Frequency Spectrum (HI Region)
    // Q3 (bottom-left): (C) Doppler Velocity & Line Fit
    // Q4 (bottom-right): (D) Time-Frequency Waterfall Spectrogram
    const int pad = 36;
    const int top_margin = 84;
    const int pane_w = (W - 3 * pad) / 2;
    const int pane_h = (H - top_margin - 2 * pad) / 2;

    const int q1_x = pad, q1_y = top_margin;
    const int q2_x = pad + pane_w + pad, q2_y = top_margin;
    const int q3_x = pad, q3_y = top_margin + pane_h + pad;
    const int q4_x = pad + pane_w + pad, q4_y = top_margin + pane_h + pad;

    auto draw_pane = [&](int x, int y, const std::string& title) {
        canvas.fill_rect(x, y, pane_w, pane_h, PixelColor(22, 27, 34));
        canvas.draw_rect(x, y, pane_w, pane_h, PixelColor(48, 54, 61), 1);
        canvas.draw_text(x + 14, y + 10, title, PixelColor(255, 255, 255), 1);
    };

    draw_pane(q1_x, q1_y, "(A) Full Bandpass & Baseline (" + baseline_method_display_name(config.baseline_method) + ")");
    draw_pane(q2_x, q2_y, "(B) Calibrated Frequency Spectrum (HI Region)");
    draw_pane(q3_x, q3_y, "(C) Doppler Velocity & " + line_analysis_method_display_name(config.line_method));
    draw_pane(q4_x, q4_y, "(D) Time-Frequency Waterfall Spectrogram");

    // ------------------------------------------------------------------------
    // Q1: Raw vs Baseline (inner-passband Y scaling + compact numeric X/Y ticks)
    // ------------------------------------------------------------------------
    {
        const int pl = q1_x + 72, pr = q1_x + pane_w - 20;
        const int pt = q1_y + 38, pb = q1_y + pane_h - 46;
        const int pw = pr - pl, ph = pb - pt;
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);

        const double f_min = safe_val(freqs_mhz.front());
        const double f_max = safe_val(freqs_mhz[n - 1]);
        if (std::abs(f_max - f_min) > 1e-12) {
            const size_t skirt_bins = std::min<size_t>(
                (n > 2) ? (n / 2 - 1) : 0,
                static_cast<size_t>(static_cast<double>(n) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)));
            double p_min = 1e9, p_max = -1e9;
            std::vector<double> r_db(n), b_db(n);
            for (size_t i = 0; i < n; ++i) {
                const double rp = std::isfinite(raw_power[i]) ? std::max(raw_power[i], 1e-12) : 1e-12;
                const double bp = std::isfinite(baseline_power[i]) ? std::max(baseline_power[i], 1e-12) : 1e-12;
                r_db[i] = 10.0 * std::log10(rp);
                b_db[i] = 10.0 * std::log10(bp);
                if (i >= skirt_bins && i + skirt_bins < n) {
                    p_min = std::min({p_min, r_db[i], b_db[i]});
                    p_max = std::max({p_max, r_db[i], b_db[i]});
                }
            }
            if (p_min > p_max) {
                for (size_t i = 0; i < n; ++i) {
                    p_min = std::min({p_min, r_db[i], b_db[i]});
                    p_max = std::max({p_max, r_db[i], b_db[i]});
                }
            }
            const double p_span = std::max(0.5, p_max - p_min);
            p_min -= p_span * 0.16;
            p_max += p_span * 0.18;

            auto q1_mx = [&](double f) { return safe_px(pl + (f - f_min) / (f_max - f_min) * pw); };
            auto q1_my = [&](double p) { return safe_px(pb - (p - p_min) / (p_max - p_min) * ph); };

            // Gridlines & ticks
            for (int t = 0; t < 5; ++t) {
                const double frac = static_cast<double>(t) / 4.0;
                const double fv = f_min + frac * (f_max - f_min);
                const int gx = q1_mx(fv);
                canvas.draw_line(gx, pt, gx, pb, PixelColor(35, 42, 52), 1);
                draw_text_center(canvas, gx, pb + 6, format_fixed(fv, 2), PixelColor(139, 148, 158), 1);

                const double pv = p_min + frac * (p_max - p_min);
                const int gy = q1_my(pv);
                canvas.draw_line(pl, gy, pr, gy, PixelColor(35, 42, 52), 1);
                draw_text_right(canvas, pl - 6, gy - 5, format_fixed(pv, 1), PixelColor(139, 148, 158), 1);
            }

            // HI line marker
            const int x_hi = q1_mx(HYDROGEN_REST_FREQ_MHZ);
            if (x_hi >= pl && x_hi <= pr) {
                canvas.draw_line(x_hi, pt, x_hi, pb, PixelColor(0, 210, 211), 1);
            }

            for (size_t i = 1; i < n; ++i) {
                if (!std::isfinite(freqs_mhz[i - 1]) || !std::isfinite(freqs_mhz[i])) continue;
                draw_clipped_line(canvas, pl, pt, pr, pb,
                                  q1_mx(freqs_mhz[i - 1]), q1_my(r_db[i - 1]),
                                  q1_mx(freqs_mhz[i]), q1_my(r_db[i]), PixelColor(84, 160, 255), 1);
                draw_clipped_line(canvas, pl, pt, pr, pb,
                                  q1_mx(freqs_mhz[i - 1]), q1_my(b_db[i - 1]),
                                  q1_mx(freqs_mhz[i]), q1_my(b_db[i]), PixelColor(255, 159, 67), 2);
            }
        }
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);
        canvas.draw_text(q1_x + 10, pt - 14, "dB", PixelColor(84, 160, 255), 1);
        draw_text_center(canvas, pl + pw / 2, pb + 24, "Frequency (MHz)", PixelColor(164, 176, 190), 1);
    }

    // ------------------------------------------------------------------------
    // Q2: Calibrated dB around HI (auto-scaled Y + compact numeric X/Y ticks)
    // ------------------------------------------------------------------------
    {
        const int pl = q2_x + 72, pr = q2_x + pane_w - 20;
        const int pt = q2_y + 38, pb = q2_y + pane_h - 46;
        const int pw = pr - pl, ph = pb - pt;
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);

        const double f_hi_min = HYDROGEN_REST_FREQ_MHZ - 0.6;
        const double f_hi_max = HYDROGEN_REST_FREQ_MHZ + 0.6;
        double q2_min = 1e9, q2_max = -1e9;
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(freqs_mhz[i]) && std::isfinite(calibrated_db[i]) &&
                freqs_mhz[i] >= f_hi_min && freqs_mhz[i] <= f_hi_max) {
                q2_min = std::min(q2_min, calibrated_db[i]);
                q2_max = std::max(q2_max, calibrated_db[i]);
            }
        }
        if (q2_min >= q2_max) {
            q2_min = -0.2;
            q2_max = 0.8;
        } else {
            const double q2_pad = std::max(0.08, (q2_max - q2_min) * 0.14);
            q2_min = std::min(q2_min - q2_pad, -0.08);
            q2_max = std::max(q2_max + q2_pad,  0.20);
        }

        auto q2_mx = [&](double f) { return safe_px(pl + (f - f_hi_min) / (f_hi_max - f_hi_min) * pw); };
        auto q2_my = [&](double db) { return safe_px(pb - (db - q2_min) / (q2_max - q2_min) * ph); };

        for (int t = 0; t < 5; ++t) {
            const double frac = static_cast<double>(t) / 4.0;
            const double fv = f_hi_min + frac * (f_hi_max - f_hi_min);
            const int gx = q2_mx(fv);
            canvas.draw_line(gx, pt, gx, pb, PixelColor(35, 42, 52), 1);
            draw_text_center(canvas, gx, pb + 6, format_fixed(fv, 3), PixelColor(139, 148, 158), 1);

            const double dbv = q2_min + frac * (q2_max - q2_min);
            const int gy = q2_my(dbv);
            canvas.draw_line(pl, gy, pr, gy, PixelColor(35, 42, 52), 1);
            draw_text_right(canvas, pl - 6, gy - 5, format_signed(dbv, 2), PixelColor(139, 148, 158), 1);
        }

        const int q2_y0 = std::clamp(q2_my(0.0), pt, pb);
        canvas.draw_line(pl, q2_y0, pr, q2_y0, PixelColor(139, 148, 158), 1);
        const int q2_xhi = q2_mx(HYDROGEN_REST_FREQ_MHZ);
        if (q2_xhi >= pl && q2_xhi <= pr) {
            canvas.draw_line(q2_xhi, pt, q2_xhi, pb, PixelColor(0, 210, 211), 1);
        }
        for (size_t i = 1; i < n; ++i) {
            const double fa = freqs_mhz[i - 1], fb = freqs_mhz[i];
            const double da = calibrated_db[i - 1], db = calibrated_db[i];
            if (!std::isfinite(fa) || !std::isfinite(fb) || !std::isfinite(da) || !std::isfinite(db)) continue;
            if (std::max(fa, fb) < f_hi_min || std::min(fa, fb) > f_hi_max) continue;
            draw_clipped_line(canvas, pl, pt, pr, pb,
                              q2_mx(fa), q2_my(da),
                              q2_mx(fb), q2_my(db), PixelColor(29, 209, 161), 1);
        }
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);
        canvas.draw_text(q2_x + 8, pt - 14, "dB", PixelColor(29, 209, 161), 1);
        draw_text_center(canvas, pl + pw / 2, pb + 24, "Sky Frequency (MHz)", PixelColor(164, 176, 190), 1);
    }

    // ------------------------------------------------------------------------
    // Q3: Doppler Velocity Profile with Line Fit (+ compact numeric X/Y ticks)
    // ------------------------------------------------------------------------
    {
        const int pl = q3_x + 72, pr = q3_x + pane_w - 20;
        const int pt = q3_y + 38, pb = q3_y + pane_h - 46;
        const int pw = pr - pl, ph = pb - pt;
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);

        const double v_min = -180.0, v_max = 180.0;
        const size_t q3_edge = std::min<size_t>(
            n / 2,
            static_cast<size_t>(static_cast<double>(n) * std::clamp(config.passband_edge_fraction, 0.0, 0.45)));
        double q3_kmin = -0.5, q3_kmax = std::max(1.0, safe_val(metrics.peak_temp_k) * 1.35 + 0.5);
        for (size_t i = q3_edge; i + q3_edge < n; ++i) {
            if (std::isfinite(vel_kms[i]) && std::isfinite(calibrated_k[i]) &&
                vel_kms[i] >= v_min && vel_kms[i] <= v_max) {
                q3_kmin = std::min(q3_kmin, calibrated_k[i]);
                q3_kmax = std::max(q3_kmax, calibrated_k[i] * 1.25 + 0.5);
            }
        }
        q3_kmin = std::min(-0.5, q3_kmin * 1.15 - 0.5);
        if (!(q3_kmax > q3_kmin + 1e-6)) {
            q3_kmin = -1.0;
            q3_kmax = 2.0;
        }

        auto q3_mx = [&](double v) { return safe_px(pl + (v - v_min) / (v_max - v_min) * pw); };
        auto q3_my = [&](double k) { return safe_px(pb - (k - q3_kmin) / (q3_kmax - q3_kmin) * ph); };

        for (int t = 0; t < 5; ++t) {
            const double frac = static_cast<double>(t) / 4.0;
            const double vv = v_min + frac * (v_max - v_min);
            const int gx = q3_mx(vv);
            canvas.draw_line(gx, pt, gx, pb, PixelColor(35, 42, 52), 1);
            draw_text_center(canvas, gx, pb + 6, format_signed(vv, 0), PixelColor(139, 148, 158), 1);

            const double kv = q3_kmin + frac * (q3_kmax - q3_kmin);
            const int gy = q3_my(kv);
            canvas.draw_line(pl, gy, pr, gy, PixelColor(35, 42, 52), 1);
            draw_text_right(canvas, pl - 6, gy - 5, format_fixed(kv, 1), PixelColor(139, 148, 158), 1);
        }

        const int q3_y0 = std::clamp(q3_my(0.0), pt, pb);
        canvas.draw_line(pl, q3_y0, pr, q3_y0, PixelColor(139, 148, 158), 1);
        const int q3_x0 = std::clamp(q3_mx(0.0), pl, pr);
        canvas.draw_line(q3_x0, pt, q3_x0, pb, PixelColor(254, 202, 87), 1);

        for (size_t i = 1; i < n; ++i) {
            const double va = vel_kms[i - 1], vb = vel_kms[i];
            const double ka = calibrated_k[i - 1], kb = calibrated_k[i];
            if (!std::isfinite(va) || !std::isfinite(vb) || !std::isfinite(ka) || !std::isfinite(kb)) continue;
            if (std::max(va, vb) < v_min || std::min(va, vb) > v_max) continue;
            draw_clipped_line(canvas, pl, pt, pr, pb,
                              q3_mx(va), q3_my(ka),
                              q3_mx(vb), q3_my(kb), PixelColor(0, 242, 254), 2);
        }
        if (metrics.gauss_fit.success && metrics.gauss_fit.fitted_curve.size() >= n) {
            if (metrics.gauss_fit.has_secondary &&
                metrics.gauss_fit.component1_curve.size() >= n &&
                metrics.gauss_fit.component2_curve.size() >= n) {
                for (size_t i = 1; i < n; ++i) {
                    const double va = vel_kms[i - 1], vb = vel_kms[i];
                    if (!std::isfinite(va) || !std::isfinite(vb)) continue;
                    if (std::max(va, vb) < v_min || std::min(va, vb) > v_max) continue;
                    if (std::isfinite(metrics.gauss_fit.component1_curve[i - 1]) &&
                        std::isfinite(metrics.gauss_fit.component1_curve[i])) {
                        draw_clipped_line(canvas, pl, pt, pr, pb,
                                          q3_mx(va), q3_my(metrics.gauss_fit.component1_curve[i - 1]),
                                          q3_mx(vb), q3_my(metrics.gauss_fit.component1_curve[i]),
                                          PixelColor(251, 146, 60), 1);
                    }
                    if (std::isfinite(metrics.gauss_fit.component2_curve[i - 1]) &&
                        std::isfinite(metrics.gauss_fit.component2_curve[i])) {
                        draw_clipped_line(canvas, pl, pt, pr, pb,
                                          q3_mx(va), q3_my(metrics.gauss_fit.component2_curve[i - 1]),
                                          q3_mx(vb), q3_my(metrics.gauss_fit.component2_curve[i]),
                                          PixelColor(167, 139, 250), 1);
                    }
                    if (metrics.gauss_fit.has_tertiary && metrics.gauss_fit.component3_curve.size() >= n &&
                        std::isfinite(metrics.gauss_fit.component3_curve[i - 1]) &&
                        std::isfinite(metrics.gauss_fit.component3_curve[i])) {
                        draw_clipped_line(canvas, pl, pt, pr, pb,
                                          q3_mx(va), q3_my(metrics.gauss_fit.component3_curve[i - 1]),
                                          q3_mx(vb), q3_my(metrics.gauss_fit.component3_curve[i]),
                                          PixelColor(16, 185, 129), 1);
                    }
                }
            }
            for (size_t i = 1; i < n; ++i) {
                const double va = vel_kms[i - 1], vb = vel_kms[i];
                const double fa = metrics.gauss_fit.fitted_curve[i - 1];
                const double fb = metrics.gauss_fit.fitted_curve[i];
                if (!std::isfinite(va) || !std::isfinite(vb) || !std::isfinite(fa) || !std::isfinite(fb)) continue;
                if (std::max(va, vb) < v_min || std::min(va, vb) > v_max) continue;
                draw_clipped_line(canvas, pl, pt, pr, pb,
                                  q3_mx(va), q3_my(fa),
                                  q3_mx(vb), q3_my(fb), PixelColor(255, 118, 117), 2);
            }
        }
        canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);
        canvas.draw_text(q3_x + 8, pt - 14, "TA (K)", PixelColor(0, 242, 254), 1);
        draw_text_center(canvas, pl + pw / 2, pb + 24, "Topocentric Velocity v_topo (km/s)", PixelColor(164, 176, 190), 1);
    }

    // ------------------------------------------------------------------------
    // Q4: Waterfall slice (with numeric X/Y ticks + colorbar, or graceful placeholder)
    // ------------------------------------------------------------------------
    {
        const int pl = q4_x + 72;
        const int pr = q4_x + pane_w - 64;
        const int pt = q4_y + 38;
        const int pb = q4_y + pane_h - 46;
        const int pw = pr - pl;
        const int ph = pb - pt;

        if (!has_waterfall) {
            canvas.fill_rect(pl, pt, pw, ph, PixelColor(16, 20, 26));
            canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);
            draw_text_center(canvas, pl + pw / 2, pt + ph / 2 - 14,
                             "No waterfall frames available", PixelColor(164, 176, 190), 1);
            draw_text_center(canvas, pl + pw / 2, pt + ph / 2 + 6,
                             "(Capture shorter than waterfall integration window)", PixelColor(110, 120, 132), 1);
            canvas.draw_text(q4_x + 8, pt - 14, "Time", PixelColor(164, 176, 190), 1);
            draw_text_center(canvas, pl + pw / 2, pb + 24, "Frequency (MHz)", PixelColor(164, 176, 190), 1);
        } else {
            int c_start = -1, c_end = -1;
            for (int c = 0; c < wf_cols; ++c) {
                if (std::isfinite(freqs_mhz[static_cast<size_t>(c)]) &&
                    freqs_mhz[static_cast<size_t>(c)] >= HYDROGEN_REST_FREQ_MHZ - 0.5) {
                    c_start = c;
                    break;
                }
            }
            for (int c = wf_cols - 1; c >= 0; --c) {
                if (std::isfinite(freqs_mhz[static_cast<size_t>(c)]) &&
                    freqs_mhz[static_cast<size_t>(c)] <= HYDROGEN_REST_FREQ_MHZ + 0.5) {
                    c_end = c;
                    break;
                }
            }
            if (c_start < 0 || c_end < 0 || c_start > c_end) {
                c_start = 0;
                c_end = wf_cols - 1;
            }
            if (c_start == c_end && wf_cols > 1) {
                if (c_end + 1 < wf_cols) ++c_end;
                else --c_start;
            }
            const int sub_cols = std::max(1, c_end - c_start + 1);

            std::vector<float> wf_sub(static_cast<size_t>(wf_rows) * static_cast<size_t>(sub_cols));
            std::vector<float> inv_base(static_cast<size_t>(sub_cols));
            for (int sc = 0; sc < sub_cols; ++sc) {
                const double bp = baseline_power[static_cast<size_t>(c_start + sc)];
                inv_base[static_cast<size_t>(sc)] = 1.0f / static_cast<float>(std::isfinite(bp) ? std::max(1e-12, bp) : 1e-12);
            }
            for (int r = 0; r < wf_rows; ++r) {
                const float* wf_row = &waterfall_data[static_cast<size_t>(r) * static_cast<size_t>(wf_cols) + static_cast<size_t>(c_start)];
                float* dst_row = &wf_sub[static_cast<size_t>(r) * static_cast<size_t>(sub_cols)];
                for (int sc = 0; sc < sub_cols; ++sc) {
                    const float v = std::isfinite(wf_row[sc]) ? wf_row[sc] : 0.0f;
                    dst_row[sc] = 10.0f * std::log10(std::max(1e-6f, v * inv_base[static_cast<size_t>(sc)]));
                }
            }
            std::vector<float> scratch_vals = wf_sub;
            float vmin = -1.0f, vmax = 1.0f;
            compute_waterfall_percentiles(scratch_vals, vmin, vmax);

            canvas.render_waterfall(wf_sub.data(), wf_rows, sub_cols, pl, pt, pw, ph, vmin, vmax, ColormapType::Magma);
            canvas.draw_rect(pl, pt, pw, ph, PixelColor(48, 54, 61), 1);

            // X ticks (Frequency MHz) + HI line
            const double f_sub_min = safe_val(freqs_mhz[static_cast<size_t>(c_start)]);
            const double f_sub_max = safe_val(freqs_mhz[static_cast<size_t>(c_end)]);
            const double f_sub_span = f_sub_max - f_sub_min;
            if (std::abs(f_sub_span) > 1e-9) {
                const int x_hi = safe_px(pl + (HYDROGEN_REST_FREQ_MHZ - f_sub_min) / f_sub_span * pw);
                if (x_hi >= pl && x_hi <= pr) {
                    canvas.draw_line(x_hi, pt, x_hi, pb, PixelColor(0, 242, 254), 1);
                }
                for (int t = 0; t < 5; ++t) {
                    const double frac = static_cast<double>(t) / 4.0;
                    const double fv = f_sub_min + frac * f_sub_span;
                    const int gx = safe_px(pl + frac * pw);
                    canvas.draw_line(gx, pb, gx, pb + 4, PixelColor(139, 148, 158), 1);
                    draw_text_center(canvas, gx, pb + 6, format_fixed(fv, 3), PixelColor(139, 148, 158), 1);
                }
            }

            // Y ticks (Time s)
            double t_start = 0.0;
            double t_end = static_cast<double>(wf_rows) * std::max(0.1, config.waterfall_dt_s);
            if (!waterfall_times.empty()) {
                const double tf = safe_val(waterfall_times.front());
                const double tb = safe_val(waterfall_times.back());
                if (wf_rows == 1 || tb <= tf) {
                    t_start = 0.0;
                    t_end = std::max(tb, std::max(0.1, config.waterfall_dt_s));
                } else {
                    t_start = tf;
                    t_end = tb;
                }
            }
            const int time_dec = ((t_end - t_start) < 20.0) ? 1 : 0;
            for (int t = 0; t < 5; ++t) {
                const double frac = static_cast<double>(t) / 4.0;
                const double tv = t_start + frac * (t_end - t_start);
                const int gy = safe_px(pb - frac * ph);
                canvas.draw_line(pl - 4, gy, pl, gy, PixelColor(139, 148, 158), 1);
                draw_text_right(canvas, pl - 6, gy - 5, format_fixed(tv, time_dec) + "s", PixelColor(139, 148, 158), 1);
            }

            // Compact colorbar on right of Q4
            const int cb_x = pr + 8;
            const int cb_w = 10;
            for (int y = 0; y < ph; ++y) {
                const float ct = 1.0f - static_cast<float>(y) / static_cast<float>(std::max(1, ph - 1));
                canvas.draw_line(cb_x, pt + y, cb_x + cb_w, pt + y, sample_colormap(ColormapType::Magma, ct), 1);
            }
            canvas.draw_rect(cb_x, pt, cb_w, ph, PixelColor(48, 54, 61), 1);
            canvas.draw_text(cb_x, pt - 14, "dB", PixelColor(164, 176, 190), 1);
            const std::string lbl_max = format_signed(vmax, 1);
            const std::string lbl_min = format_signed(vmin, 1);
            const int lbl_x_max = std::min(cb_x + cb_w + 4, q4_x + pane_w - canvas.measure_text_width(lbl_max, 1) - 4);
            const int lbl_x_min = std::min(cb_x + cb_w + 4, q4_x + pane_w - canvas.measure_text_width(lbl_min, 1) - 4);
            canvas.draw_text(std::max(cb_x + cb_w + 2, lbl_x_max), pt - 2, lbl_max, PixelColor(180, 188, 196), 1);
            canvas.draw_text(std::max(cb_x + cb_w + 2, lbl_x_min), pb - 10, lbl_min, PixelColor(180, 188, 196), 1);

            canvas.draw_text(q4_x + 8, pt - 14, "Time", PixelColor(164, 176, 190), 1);
            draw_text_center(canvas, pl + pw / 2, pb + 24, "Frequency (MHz)", PixelColor(164, 176, 190), 1);
        }
    }

    return canvas.save_png(output_path);
}

} // namespace astro
