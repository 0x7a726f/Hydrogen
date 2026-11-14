#pragma once

#include <vector>
#include <string>
#include <cstdint>

enum class WaterfallPalette {
    ClassicYoussef,
    Viridis,
    Turbo,
    Inferno,
    Magma
};

class Waterfall {
public:
    Waterfall(int width = 800, int height = 260);
    ~Waterfall() = default;

    void resize(int width, int height);
    void resize_preserving_history(int width, int height);
    int get_width() const { return width_; }
    int get_height() const { return height_; }

    void set_palette(WaterfallPalette palette);
    WaterfallPalette get_palette() const { return current_palette_; }
    static const std::vector<std::string>& get_palette_names();

    // Push one spectrum slice into the top row, rolling existing rows down in O(width)
    void add_line(const std::vector<float>& power_db, float min_db, float max_db);

    // Direct access to linearized RGBA32 pixels (width * height * 4 bytes, newest row at top) and active 256-color LUT
    const uint32_t* get_pixel_data() const;
    void copy_linear_pixels(uint32_t* dst) const;
    uint64_t get_revision() const { return revision_; }
    const uint32_t* get_palette_lut() const { return palette_lut_; }
    size_t get_pixel_count() const { return ring_.size(); }
    size_t get_byte_size() const { return ring_.size() * sizeof(uint32_t); }

    void clear();

private:
    void rebuild_palette_lut();
    void linearize_into(uint32_t* dst) const;

    int width_ = 800;
    int height_ = 260;
    int head_row_ = 0; // Index of newest slice inside ring_
    uint64_t revision_ = 1;
    WaterfallPalette current_palette_ = WaterfallPalette::ClassicYoussef;

    std::vector<uint32_t> ring_;
    mutable std::vector<uint32_t> linear_cache_;
    mutable uint64_t linear_cache_rev_ = 0;
    uint32_t palette_lut_[256];
};
