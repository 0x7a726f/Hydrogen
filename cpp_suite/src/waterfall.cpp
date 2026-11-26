#include "waterfall.h"
#include "simd_dsp.h"
#include <algorithm>
#include <cstring>
#include <cmath>

struct ColorAnchor {
    float pos;
    uint8_t r, g, b;
};

static inline uint32_t make_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    // Little-endian layout: byte0=R, byte1=G, byte2=B, byte3=A
    return ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
}

static constexpr float kEmptyDbSentinel = -1.0e9f;

Waterfall::Waterfall(int width, int height)
    : width_(std::max(1, width)), height_(std::max(1, height)) {
    const size_t total = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    ring_.assign(total, make_rgba(13, 17, 23));
    db_ring_.assign(total, kEmptyDbSentinel);
    head_row_ = 0;
    filled_rows_ = 0;
    revision_ = 1;
    rebuild_palette_lut();
}

void Waterfall::linearize_into(uint32_t* dst) const {
    if (!dst || width_ <= 0 || height_ <= 0 || ring_.empty()) return;
    const size_t row_bytes = static_cast<size_t>(width_) * sizeof(uint32_t);
    const int top_rows = height_ - head_row_;
    std::memcpy(dst, &ring_[static_cast<size_t>(head_row_) * width_], static_cast<size_t>(top_rows) * row_bytes);
    if (head_row_ > 0) {
        std::memcpy(dst + static_cast<size_t>(top_rows) * width_, &ring_[0], static_cast<size_t>(head_row_) * row_bytes);
    }
}

void Waterfall::copy_linear_pixels(uint32_t* dst) const {
    linearize_into(dst);
}

const uint32_t* Waterfall::get_pixel_data() const {
    if (linear_cache_rev_ != revision_ || linear_cache_.size() != ring_.size()) {
        linear_cache_.resize(ring_.size());
        linearize_into(linear_cache_.data());
        linear_cache_rev_ = revision_;
    }
    return linear_cache_.data();
}

void Waterfall::map_db_row_to_rgba(const float* src_db, uint32_t* dst_rgba, int count,
                                   float min_db, float max_db) const {
    const float range_db = std::max(max_db - min_db, 1.0f);
    const float scale = 255.0f / range_db;
    int x = 0;
#if defined(__AVX2__)
    const __m256 v_min   = _mm256_set1_ps(min_db);
    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_255   = _mm256_set1_ps(255.0f);
    for (; x + 8 <= count; x += 8) {
        __m256 p = _mm256_loadu_ps(src_db + x);
        __m256 scaled = _mm256_mul_ps(_mm256_sub_ps(p, v_min), v_scale);
        __m256 valid = _mm256_cmp_ps(scaled, scaled, _CMP_ORD_Q);
        scaled = _mm256_and_ps(scaled, valid);
        scaled = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_255);
        alignas(32) int32_t idx[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx), _mm256_cvttps_epi32(scaled));
        dst_rgba[x + 0] = palette_lut_[idx[0]];
        dst_rgba[x + 1] = palette_lut_[idx[1]];
        dst_rgba[x + 2] = palette_lut_[idx[2]];
        dst_rgba[x + 3] = palette_lut_[idx[3]];
        dst_rgba[x + 4] = palette_lut_[idx[4]];
        dst_rgba[x + 5] = palette_lut_[idx[5]];
        dst_rgba[x + 6] = palette_lut_[idx[6]];
        dst_rgba[x + 7] = palette_lut_[idx[7]];
    }
#endif
    for (; x < count; ++x) {
        float norm = (src_db[x] - min_db) * scale;
        if (!(norm >= 0.0f)) norm = 0.0f;
        else if (norm > 255.0f) norm = 255.0f;
        dst_rgba[x] = palette_lut_[static_cast<uint8_t>(norm)];
    }
}

void Waterfall::recolor_all_rows() {
    if (width_ <= 0 || height_ <= 0 || filled_rows_ <= 0 || ring_.empty() || db_ring_.size() != ring_.size()) {
        return;
    }
    const int n_valid = std::min(filled_rows_, height_);
    for (int i = 0; i < n_valid; ++i) {
        const int row = (head_row_ + i) % height_;
        const size_t off = static_cast<size_t>(row) * static_cast<size_t>(width_);
        map_db_row_to_rgba(&db_ring_[off], &ring_[off], width_, last_min_db_, last_max_db_);
    }
    ++revision_;
}

void Waterfall::recolor(float min_db, float max_db) {
    if (!std::isfinite(min_db) || !std::isfinite(max_db)) return;
    if (std::abs(min_db - last_min_db_) < 1e-4f && std::abs(max_db - last_max_db_) < 1e-4f) {
        return;
    }
    last_min_db_ = min_db;
    last_max_db_ = max_db;
    recolor_all_rows();
}

void Waterfall::resize(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        const size_t total = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        ring_.assign(total, make_rgba(13, 17, 23));
        db_ring_.assign(total, kEmptyDbSentinel);
        head_row_ = 0;
        filled_rows_ = 0;
        ++revision_;
    }
}

void Waterfall::resize_preserving_history(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (width == width_ && height == height_) return;
    if (ring_.empty() || db_ring_.size() != ring_.size() || width_ <= 0 || height_ <= 0) {
        resize(width, height);
        return;
    }

    const uint32_t bg = make_rgba(13, 17, 23);
    const size_t next_total = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint32_t> next_ring(next_total, bg);
    std::vector<float> next_db(next_total, kEmptyDbSentinel);

    const int copy_rows = std::min(std::min(filled_rows_, height_), height);
    const float scale_x = (width > 1 && width_ > 1)
        ? (static_cast<float>(width_ - 1) / static_cast<float>(width - 1))
        : 0.0f;

    for (int y = 0; y < copy_rows; ++y) {
        const int src_row_idx = (head_row_ + y) % height_;
        const float* src_db = &db_ring_[static_cast<size_t>(src_row_idx) * static_cast<size_t>(width_)];
        float* dst_db = &next_db[static_cast<size_t>(y) * static_cast<size_t>(width)];
        uint32_t* dst_px = &next_ring[static_cast<size_t>(y) * static_cast<size_t>(width)];

        if (width == width_) {
            std::memcpy(dst_db, src_db, static_cast<size_t>(width) * sizeof(float));
        } else {
            for (int x = 0; x < width; ++x) {
                const float fx = static_cast<float>(x) * scale_x;
                const int x0 = static_cast<int>(fx);
                const int x1 = std::min(x0 + 1, width_ - 1);
                const float t = fx - static_cast<float>(x0);
                const float v0 = src_db[x0];
                const float v1 = src_db[x1];
                const bool ok0 = std::isfinite(v0) && v0 > kEmptyDbSentinel * 0.5f;
                const bool ok1 = std::isfinite(v1) && v1 > kEmptyDbSentinel * 0.5f;
                if (ok0 && ok1) {
                    dst_db[x] = v0 + t * (v1 - v0);
                } else if (ok0) {
                    dst_db[x] = v0;
                } else if (ok1) {
                    dst_db[x] = v1;
                } else {
                    dst_db[x] = last_min_db_;
                }
            }
        }
        map_db_row_to_rgba(dst_db, dst_px, width, last_min_db_, last_max_db_);
    }

    width_ = width;
    height_ = height;
    ring_ = std::move(next_ring);
    db_ring_ = std::move(next_db);
    head_row_ = 0;
    filled_rows_ = copy_rows;
    ++revision_;
}

void Waterfall::clear() {
    std::fill(ring_.begin(), ring_.end(), make_rgba(13, 17, 23));
    std::fill(db_ring_.begin(), db_ring_.end(), kEmptyDbSentinel);
    head_row_ = 0;
    filled_rows_ = 0;
    ++revision_;
}

void Waterfall::set_palette(WaterfallPalette palette) {
    const bool changed = (current_palette_ != palette);
    current_palette_ = palette;
    rebuild_palette_lut();
    if (changed) {
        recolor_all_rows();
    }
}

const std::vector<std::string>& Waterfall::get_palette_names() {
    static const std::vector<std::string> names = {
        "Classic SatDump",
        "Viridis",
        "Turbo",
        "Inferno",
        "Magma"
    };
    return names;
}

static void interpolate_palette(const std::vector<ColorAnchor>& anchors, uint32_t* lut) {
    for (int i = 0; i < 256; ++i) {
        float t = (float)i / 255.0f;
        if (t <= anchors.front().pos) {
            lut[i] = make_rgba(anchors.front().r, anchors.front().g, anchors.front().b);
            continue;
        }
        if (t >= anchors.back().pos) {
            lut[i] = make_rgba(anchors.back().r, anchors.back().g, anchors.back().b);
            continue;
        }
        for (size_t a = 0; a < anchors.size() - 1; ++a) {
            if (t >= anchors[a].pos && t <= anchors[a + 1].pos) {
                float seg = (t - anchors[a].pos) / (anchors[a + 1].pos - anchors[a].pos);
                uint8_t r = (uint8_t)(anchors[a].r + seg * (anchors[a + 1].r - anchors[a].r));
                uint8_t g = (uint8_t)(anchors[a].g + seg * (anchors[a + 1].g - anchors[a].g));
                uint8_t b = (uint8_t)(anchors[a].b + seg * (anchors[a + 1].b - anchors[a].b));
                lut[i] = make_rgba(r, g, b);
                break;
            }
        }
    }
}

void Waterfall::rebuild_palette_lut() {
    switch (current_palette_) {
        case WaterfallPalette::ClassicYoussef: {
            static const std::vector<ColorAnchor> anchors = {
                {0.00f,  10,  15,  45},
                {0.20f,   0,  50, 160},
                {0.38f,   0, 160, 210},
                {0.55f,  30, 200,  70},
                {0.72f, 250, 220,   0},
                {0.88f, 240,  50,   0},
                {1.00f, 255, 255, 255}
            };
            interpolate_palette(anchors, palette_lut_);
            break;
        }
        case WaterfallPalette::Viridis: {
            static const std::vector<ColorAnchor> anchors = {
                {0.00f,  68,   1,  84},
                {0.25f,  59,  82, 139},
                {0.50f,  33, 145, 140},
                {0.75f,  94, 201,  98},
                {1.00f, 253, 231,  37}
            };
            interpolate_palette(anchors, palette_lut_);
            break;
        }
        case WaterfallPalette::Turbo: {
            static const std::vector<ColorAnchor> anchors = {
                {0.00f,  48,  18,  59},
                {0.20f,  70, 134, 251},
                {0.40f,  27, 229, 181},
                {0.60f, 164, 252,  60},
                {0.80f, 251, 126,  33},
                {1.00f, 122,   4,   3}
            };
            interpolate_palette(anchors, palette_lut_);
            break;
        }
        case WaterfallPalette::Inferno: {
            static const std::vector<ColorAnchor> anchors = {
                {0.00f,   0,   0,   4},
                {0.25f,  87,  16, 110},
                {0.50f, 187,  55,  84},
                {0.75f, 249, 142,   9},
                {1.00f, 252, 255, 164}
            };
            interpolate_palette(anchors, palette_lut_);
            break;
        }
        case WaterfallPalette::Magma: {
            static const std::vector<ColorAnchor> anchors = {
                {0.00f,   0,   0,   4},
                {0.25f,  81,  18, 124},
                {0.50f, 182,  54, 121},
                {0.75f, 251, 136,  97},
                {1.00f, 252, 253, 191}
            };
            interpolate_palette(anchors, palette_lut_);
            break;
        }
    }
}

void Waterfall::add_line(const std::vector<float>& power_db, float min_db, float max_db) {
    if (width_ <= 0 || height_ <= 0 || power_db.empty() || ring_.empty()) return;

    const float safe_min = std::isfinite(min_db) ? min_db : last_min_db_;
    const float safe_max = std::isfinite(max_db) ? max_db : last_max_db_;

    if (filled_rows_ > 0 &&
        (std::abs(safe_min - last_min_db_) > 1e-4f || std::abs(safe_max - last_max_db_) > 1e-4f)) {
        last_min_db_ = safe_min;
        last_max_db_ = safe_max;
        recolor_all_rows();
    } else {
        last_min_db_ = safe_min;
        last_max_db_ = safe_max;
    }

    // Step newest row pointer backwards in O(1) instead of memmoving entire history
    head_row_ = (head_row_ == 0) ? (height_ - 1) : (head_row_ - 1);
    const size_t row_off = static_cast<size_t>(head_row_) * static_cast<size_t>(width_);
    float* dst_db = &db_ring_[row_off];
    uint32_t* dst_row = &ring_[row_off];

    const size_t n_pwr = power_db.size();
    if (n_pwr >= static_cast<size_t>(width_)) {
        for (int x = 0; x < width_; ++x) {
            size_t b0 = static_cast<size_t>(x) * n_pwr / static_cast<size_t>(width_);
            size_t b1 = static_cast<size_t>(x + 1) * n_pwr / static_cast<size_t>(width_);
            if (b1 <= b0) b1 = std::min(b0 + 1, n_pwr);
            float peak = safe_min;
            bool has_finite = false;
            for (size_t b = b0; b < b1; ++b) {
                const float v = power_db[b];
                if (std::isfinite(v) && (!has_finite || v > peak)) {
                    peak = v;
                    has_finite = true;
                }
            }
            dst_db[x] = peak;
        }
    } else {
        for (int x = 0; x < width_; ++x) {
            const float pos = (width_ > 1)
                ? (static_cast<float>(x) / static_cast<float>(width_ - 1)) * static_cast<float>(n_pwr - 1)
                : 0.0f;
            const size_t b0 = static_cast<size_t>(pos);
            const size_t b1 = std::min(b0 + 1, n_pwr - 1);
            const float frac = pos - static_cast<float>(b0);
            const float v0 = power_db[b0];
            const float v1 = power_db[b1];
            if (std::isfinite(v0) && std::isfinite(v1)) {
                dst_db[x] = v0 + frac * (v1 - v0);
            } else if (std::isfinite(v0)) {
                dst_db[x] = v0;
            } else if (std::isfinite(v1)) {
                dst_db[x] = v1;
            } else {
                dst_db[x] = safe_min;
            }
        }
    }

    map_db_row_to_rgba(dst_db, dst_row, width_, safe_min, safe_max);
    if (filled_rows_ < height_) {
        ++filled_rows_;
    }
    ++revision_;
}
