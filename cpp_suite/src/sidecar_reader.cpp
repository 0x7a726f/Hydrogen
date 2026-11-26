#include "sidecar_reader.h"
#include "utf8_path.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace astro {

static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n\"'");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n\"'");
    return str.substr(start, end - start + 1);
}

static std::string ltrim_ws(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : str.substr(start);
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static void append_utf8_codepoint(std::string& out, uint32_t cp) {
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

// JSON key-value scanner that skips quoted string literals and tracks {} object depth
static size_t find_json_value_start(const std::string& content, const std::string& key) {
    const size_t n = content.size();
    size_t i = 0;
    int brace_depth = 0;
    int bracket_depth = 0;
    while (i < n) {
        char ch = content[i];
        if (ch == '{') { ++brace_depth; ++i; continue; }
        if (ch == '}') { if (brace_depth > 0) --brace_depth; ++i; continue; }
        if (ch == '[') { ++bracket_depth; ++i; continue; }
        if (ch == ']') { if (bracket_depth > 0) --bracket_depth; ++i; continue; }
        if (ch == '"') {
            ++i;
            std::string decoded_token;
            bool closed = false;
            while (i < n) {
                char c = content[i];
                if (c == '\\' && i + 1 < n) {
                    char next = content[i + 1];
                    switch (next) {
                        case '"':  decoded_token.push_back('"'); break;
                        case '\\': decoded_token.push_back('\\'); break;
                        case '/':  decoded_token.push_back('/'); break;
                        case 'b':  decoded_token.push_back('\b'); break;
                        case 'f':  decoded_token.push_back('\f'); break;
                        case 'n':  decoded_token.push_back('\n'); break;
                        case 'r':  decoded_token.push_back('\r'); break;
                        case 't':  decoded_token.push_back('\t'); break;
                        default:   decoded_token.push_back(next); break;
                    }
                    i += 2;
                } else if (c == '"') {
                    closed = true;
                    ++i;
                    break;
                } else {
                    decoded_token.push_back(c);
                    ++i;
                }
            }
            if (!closed) break;

            size_t p = i;
            while (p < n && std::isspace(static_cast<unsigned char>(content[p]))) ++p;
            if (p < n && content[p] == ':') {
                ++p;
                while (p < n && std::isspace(static_cast<unsigned char>(content[p]))) ++p;
                if (brace_depth == 1 && bracket_depth == 0 && decoded_token == key) {
                    return p;
                }
                i = p;
                continue;
            }
            continue;
        }
        ++i;
    }
    return std::string::npos;
}

// Strict RFC 8259 JSON number syntax (rejects hex, leading '+', NaN/Inf, trailing junk)
static bool json_extract_number(const std::string& content, const std::string& key, double& out_val) {
    size_t p = find_json_value_start(content, key);
    if (p == std::string::npos || p >= content.size()) return false;

    const size_t n = content.size();
    size_t cur = p;
    if (cur < n && content[cur] == '-') ++cur;
    if (cur >= n || !std::isdigit(static_cast<unsigned char>(content[cur]))) return false;

    if (content[cur] == '0') {
        ++cur;
        // Leading zeros like 00 or 0x10 are invalid JSON numbers
        if (cur < n && std::isdigit(static_cast<unsigned char>(content[cur]))) return false;
    } else {
        while (cur < n && std::isdigit(static_cast<unsigned char>(content[cur]))) ++cur;
    }

    if (cur < n && content[cur] == '.') {
        ++cur;
        if (cur >= n || !std::isdigit(static_cast<unsigned char>(content[cur]))) return false;
        while (cur < n && std::isdigit(static_cast<unsigned char>(content[cur]))) ++cur;
    }

    if (cur < n && (content[cur] == 'e' || content[cur] == 'E')) {
        ++cur;
        if (cur < n && (content[cur] == '+' || content[cur] == '-')) ++cur;
        if (cur >= n || !std::isdigit(static_cast<unsigned char>(content[cur]))) return false;
        while (cur < n && std::isdigit(static_cast<unsigned char>(content[cur]))) ++cur;
    }

    // Must be followed by a valid JSON value delimiter or EOF
    if (cur < n) {
        char delim = content[cur];
        if (!std::isspace(static_cast<unsigned char>(delim)) &&
            delim != ',' && delim != '}' && delim != ']') {
            return false;
        }
    }

    const std::string token = content.substr(p, cur - p);
    char* end = nullptr;
    double val = std::strtod(token.c_str(), &end);
    if (end == token.c_str() + token.size() && std::isfinite(val)) {
        out_val = val;
        return true;
    }
    return false;
}

static bool json_extract_string(const std::string& content, const std::string& key, std::string& out_str) {
    size_t p = find_json_value_start(content, key);
    if (p == std::string::npos || p >= content.size() || content[p] != '"') return false;
    std::string decoded;
    for (size_t i = p + 1; i < content.size(); ++i) {
        char c = content[i];
        if (c == '\\' && i + 1 < content.size()) {
            char next = content[++i];
            switch (next) {
                case '"':  decoded.push_back('"'); break;
                case '\\': decoded.push_back('\\'); break;
                case '/':  decoded.push_back('/'); break;
                case 'b':  decoded.push_back('\b'); break;
                case 'f':  decoded.push_back('\f'); break;
                case 'n':  decoded.push_back('\n'); break;
                case 'r':  decoded.push_back('\r'); break;
                case 't':  decoded.push_back('\t'); break;
                case 'u':
                    if (i + 4 < content.size()) {
                        int h0 = hex_nibble(content[i + 1]);
                        int h1 = hex_nibble(content[i + 2]);
                        int h2 = hex_nibble(content[i + 3]);
                        int h3 = hex_nibble(content[i + 4]);
                        if (h0 >= 0 && h1 >= 0 && h2 >= 0 && h3 >= 0) {
                            uint32_t cp = static_cast<uint32_t>((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
                            i += 4;
                            // Handle UTF-16 surrogate pair if present
                            if (cp >= 0xD800u && cp <= 0xDBFFu && i + 6 < content.size() &&
                                content[i + 1] == '\\' && content[i + 2] == 'u') {
                                int l0 = hex_nibble(content[i + 3]);
                                int l1 = hex_nibble(content[i + 4]);
                                int l2 = hex_nibble(content[i + 5]);
                                int l3 = hex_nibble(content[i + 6]);
                                if (l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0) {
                                    uint32_t low = static_cast<uint32_t>((l0 << 12) | (l1 << 8) | (l2 << 4) | l3);
                                    if (low >= 0xDC00u && low <= 0xDFFFu) {
                                        cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                                        i += 6;
                                    }
                                }
                            }
                            append_utf8_codepoint(decoded, cp);
                        } else {
                            decoded.push_back('u');
                        }
                    } else {
                        decoded.push_back('u');
                    }
                    break;
                default:   decoded.push_back(next); break;
            }
        } else if (c == '"') {
            out_str = std::move(decoded);
            return true;
        } else {
            decoded.push_back(c);
        }
    }
    return false;
}

static bool json_extract_bool(const std::string& content, const std::string& key, bool& out_bool) {
    size_t p = find_json_value_start(content, key);
    if (p == std::string::npos || p >= content.size()) return false;
    if (content.compare(p, 4, "true") == 0) {
        out_bool = true;
        return true;
    }
    if (content.compare(p, 5, "false") == 0) {
        out_bool = false;
        return true;
    }
    return false;
}

bool SidecarReader::parse_json_sidecar(const std::string& json_path, ObservationMetadata& meta) {
    std::ifstream file(astro_fs::utf8_path(json_path));
    if (!file.is_open()) return false;

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    bool has_data = false;
    double val = 0.0;
    if ((json_extract_number(content, "center_freq_hz", val) ||
         json_extract_number(content, "center_frequency_hz", val) ||
         json_extract_number(content, "fc_hz", val)) && val > 0.0) {
        meta.center_freq_hz = val;
        has_data = true;
    }
    if ((json_extract_number(content, "sample_rate_hz", val) ||
         json_extract_number(content, "sampling_rate_hz", val) ||
         json_extract_number(content, "fs_hz", val)) && val > 0.0) {
        meta.sample_rate_hz = val;
        has_data = true;
    }
    if (json_extract_number(content, "gain_db", val)) {
        meta.gain_db = val;
        meta.has_gain = true;
        has_data = true;
    }
    if (json_extract_number(content, "lo_offset_khz", val)) {
        meta.lo_offset_khz = val;
        has_data = true;
    }
    bool bval = false;
    if (json_extract_bool(content, "bias_tee", bval)) {
        meta.bias_tee = bval;
        has_data = true;
    }
    std::string sval;
    if (json_extract_string(content, "device", sval) || json_extract_string(content, "device_name", sval)) {
        meta.device_name = sval;
        has_data = true;
    }
    if (json_extract_string(content, "timestamp", sval)) {
        meta.timestamp = sval;
        has_data = true;
    }

    if (has_data) {
        meta.found_sidecar = true;
        meta.sidecar_source = astro_fs::path_to_utf8(astro_fs::utf8_path(json_path).filename());
        return true;
    }
    return false;
}

bool SidecarReader::parse_txt_sidecar(const std::string& txt_path, ObservationMetadata& meta) {
    std::ifstream file(astro_fs::utf8_path(txt_path));
    if (!file.is_open()) return false;

    auto starts_with = [](const std::string& s, const char* prefix) {
        return s.rfind(prefix, 0) == 0;
    };

    std::string raw_line;
    bool has_data = false;

    while (std::getline(file, raw_line)) {
        const std::string line = ltrim_ws(raw_line);
        if (line.empty()) continue;

        if (starts_with(line, "Tuned SDR Center LO:") ||
            starts_with(line, "Center Frequency (LO):") ||
            starts_with(line, "Center Frequency:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                double parsed_fc = 0.0;
                if (try_parse_frequency_or_rate(val_str, parsed_fc, true) && parsed_fc > 0.0) {
                    meta.center_freq_hz = parsed_fc;
                    has_data = true;
                }
            }
        } else if (starts_with(line, "Sampling Rate (Fs):") ||
                   starts_with(line, "SDR Sample Rate:") ||
                   starts_with(line, "Sample Rate:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                double parsed_fs = 0.0;
                if (try_parse_sample_rate(val_str, parsed_fs) && parsed_fs > 0.0) {
                    meta.sample_rate_hz = parsed_fs;
                    has_data = true;
                }
            }
        } else if (starts_with(line, "RF Gain:") || starts_with(line, "Gain:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                try {
                    double parsed_gain = std::stod(val_str);
                    if (std::isfinite(parsed_gain)) {
                        meta.gain_db = parsed_gain;
                        meta.has_gain = true;
                        has_data = true;
                    }
                } catch (...) {}
            }
        } else if (starts_with(line, "Bias-Tee") || starts_with(line, "Bias Tee")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                std::transform(val_str.begin(), val_str.end(), val_str.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                bool disabled = (val_str.find("off") != std::string::npos ||
                                 val_str.find("disabled") != std::string::npos ||
                                 val_str.find("false") != std::string::npos ||
                                 val_str.find("none") != std::string::npos);
                meta.bias_tee = !disabled && (val_str == "on" ||
                                              val_str.rfind("on ", 0) == 0 ||
                                              val_str.find("enabled") != std::string::npos ||
                                              val_str.find("true") != std::string::npos);
                has_data = true;
            }
        } else if (starts_with(line, "LO Frequency Offset:") ||
                   starts_with(line, "LO Offset") ||
                   starts_with(line, "Offset Shift:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                val_str.erase(std::remove(val_str.begin(), val_str.end(), ','), val_str.end());
                try {
                    size_t idx = 0;
                    double parsed_off = std::stod(val_str, &idx);
                    if (std::isfinite(parsed_off)) {
                        std::string unit_tail = val_str.substr(idx);
                        std::transform(unit_tail.begin(), unit_tail.end(), unit_tail.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                        // Strip leading spaces to inspect the immediate unit token
                        unit_tail = ltrim_ws(unit_tail);
                        if (unit_tail.rfind("mhz", 0) == 0) {
                            parsed_off *= 1e3;
                        } else if (unit_tail.rfind("ghz", 0) == 0) {
                            parsed_off *= 1e6;
                        } else if (unit_tail.rfind("hz", 0) == 0) {
                            parsed_off *= 1e-3;
                        }
                        meta.lo_offset_khz = parsed_off;
                        has_data = true;
                    }
                } catch (...) {}
            }
        } else if (starts_with(line, "Capture Timestamp:") ||
                   starts_with(line, "Timestamp:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string ts = trim(line.substr(colon + 1));
                if (!ts.empty()) {
                    meta.timestamp = std::move(ts);
                    has_data = true;
                }
            }
        } else if (starts_with(line, "Device:") || starts_with(line, "Frontend Device:") ||
                   starts_with(line, "Device Name:") || starts_with(line, "Receiver Hardware:")) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string dev = trim(line.substr(colon + 1));
                if (!dev.empty()) {
                    meta.device_name = std::move(dev);
                    has_data = true;
                }
            }
        }
    }

    if (has_data) {
        meta.found_sidecar = true;
        meta.sidecar_source = astro_fs::path_to_utf8(astro_fs::utf8_path(txt_path).filename());
        return true;
    }
    return false;
}

bool SidecarReader::parse_filename_fallback(const std::string& cs16_path, ObservationMetadata& meta) {
    std::string fname = astro_fs::path_to_utf8(astro_fs::utf8_path(cs16_path).filename());
    // Supported patterns include:
    //   2026-11-11_17-47-26_2400000SPS_1420410000Hz.cs16
    //   hi_capture_20261118_120000_1420.500MHz_2.40MSPS.cs16
    static const std::regex rate_regex(R"(([0-9]+(?:\.[0-9]+)?)\s*([kmg]?)sps)", std::regex::icase);
    static const std::regex freq_regex(R"(([0-9]+(?:\.[0-9]+)?)\s*([kmg]?)hz)", std::regex::icase);

    auto prefix_mult = [](const std::string& p) -> double {
        if (p.empty()) return 1.0;
        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(p[0])));
        if (c == 'k') return 1e3;
        if (c == 'm') return 1e6;
        if (c == 'g') return 1e9;
        return 1.0;
    };

    std::smatch match;
    bool found = false;
    if (std::regex_search(fname, match, rate_regex)) {
        try {
            double v = std::stod(match[1].str()) * prefix_mult(match[2].str());
            if (std::isfinite(v) && v > 0.0) {
                meta.sample_rate_hz = v;
                found = true;
            }
        } catch (...) {}
    }
    if (std::regex_search(fname, match, freq_regex)) {
        try {
            double v = std::stod(match[1].str()) * prefix_mult(match[2].str());
            if (std::isfinite(v) && v > 0.0) {
                meta.center_freq_hz = v;
                found = true;
            }
        } catch (...) {}
    }
    if (found) {
        meta.found_sidecar = true;
        meta.sidecar_source = "filename inference";
        return true;
    }
    return false;
}

ObservationMetadata SidecarReader::parse(const std::string& cs16_path) {
    ObservationMetadata meta;
    std::filesystem::path p = astro_fs::utf8_path(cs16_path);
    std::filesystem::path base_path = p.parent_path() / p.stem();

    auto merge_missing = [](ObservationMetadata& dst, const ObservationMetadata& src) {
        if (!src.found_sidecar) return;
        bool merged_any = false;
        if (dst.center_freq_hz <= 0.0 && src.center_freq_hz > 0.0) {
            dst.center_freq_hz = src.center_freq_hz;
            merged_any = true;
        }
        if (dst.sample_rate_hz <= 0.0 && src.sample_rate_hz > 0.0) {
            dst.sample_rate_hz = src.sample_rate_hz;
            merged_any = true;
        }
        if (!dst.has_gain && src.has_gain) {
            dst.gain_db = src.gain_db;
            dst.has_gain = true;
            merged_any = true;
        }
        if (dst.device_name.empty() && !src.device_name.empty()) {
            dst.device_name = src.device_name;
            merged_any = true;
        }
        if (dst.timestamp.empty() && !src.timestamp.empty()) {
            dst.timestamp = src.timestamp;
            merged_any = true;
        }
        if (!dst.found_sidecar) {
            dst = src;
        } else if (merged_any && !src.sidecar_source.empty() &&
                   dst.sidecar_source.find(src.sidecar_source) == std::string::npos) {
            dst.sidecar_source += " + " + src.sidecar_source;
        }
    };

    std::error_code ec;
    // 1. Try .json sidecar
    std::filesystem::path json_path = base_path;
    json_path += ".json";
    if (std::filesystem::exists(json_path, ec)) {
        parse_json_sidecar(astro_fs::path_to_utf8(json_path), meta);
    }

    // If JSON was missing or lacked core RF fields, merge from .txt sidecar
    if (meta.center_freq_hz <= 0.0 || meta.sample_rate_hz <= 0.0) {
        std::filesystem::path txt_path = base_path;
        txt_path += ".txt";
        if (std::filesystem::exists(txt_path, ec)) {
            ObservationMetadata txt_meta;
            if (parse_txt_sidecar(astro_fs::path_to_utf8(txt_path), txt_meta)) {
                merge_missing(meta, txt_meta);
            }
        }
    }

    // 3. Fallback: Merge any still-missing RF fields from filename tokens
    if (meta.center_freq_hz <= 0.0 || meta.sample_rate_hz <= 0.0) {
        ObservationMetadata fname_meta;
        if (parse_filename_fallback(cs16_path, fname_meta)) {
            merge_missing(meta, fname_meta);
        }
    }

    return meta;
}

bool SidecarReader::apply_to_config(const ObservationMetadata& meta, AstroConfig& config) {
    if (!meta.found_sidecar) return false;
    bool applied = false;
    if (std::isfinite(meta.center_freq_hz) && meta.center_freq_hz > 1e5) {
        config.center_freq_hz = meta.center_freq_hz;
        applied = true;
    }
    if (std::isfinite(meta.sample_rate_hz) && meta.sample_rate_hz > 1e4) {
        config.sample_rate_hz = meta.sample_rate_hz;
        applied = true;
    }
    return applied;
}

std::vector<std::string> SidecarReader::scan_cs16_files(const std::vector<std::string>& search_paths) {
    struct FoundEntry {
        std::string path;
        std::filesystem::file_time_type mtime;
    };
    std::vector<FoundEntry> entries;
    std::unordered_set<std::string> seen_canonical;

    for (const auto& path_str : search_paths) {
        std::error_code ec;
        const std::filesystem::path search_fs = astro_fs::utf8_path(path_str);
        if (!std::filesystem::exists(search_fs, ec) || ec) continue;

        std::filesystem::recursive_directory_iterator it(
            search_fs, std::filesystem::directory_options::skip_permission_denied, ec);
        std::filesystem::recursive_directory_iterator end;
        if (ec) continue;

        for (; !ec && it != end; it.increment(ec)) {
            const auto& entry = *it;
            std::error_code file_ec;
            if (!entry.is_regular_file(file_ec) || file_ec) continue;

            std::string ext = astro_fs::path_to_utf8(entry.path().extension());
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext != ".cs16") continue;

            std::error_code canon_ec;
            auto canon = std::filesystem::weakly_canonical(entry.path(), canon_ec);
            std::error_code abs_ec;
            std::string abs_path = astro_fs::path_to_utf8(std::filesystem::absolute(entry.path(), abs_ec));
            if (abs_ec || abs_path.empty()) abs_path = astro_fs::path_to_utf8(entry.path());

            std::string canon_key = canon_ec
                ? astro_fs::path_to_generic_utf8(astro_fs::utf8_path(abs_path))
                : astro_fs::path_to_generic_utf8(canon);
#ifdef _WIN32
            std::transform(canon_key.begin(), canon_key.end(), canon_key.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
            if (seen_canonical.insert(std::move(canon_key)).second) {
                std::error_code time_ec;
                auto mtime = entry.last_write_time(time_ec);
                entries.push_back({std::move(abs_path), mtime});
            }
        }
    }

    // Sort by cached last modified time (newest first)
    std::sort(entries.begin(), entries.end(), [](const FoundEntry& a, const FoundEntry& b) {
        return a.mtime > b.mtime;
    });

    std::vector<std::string> results;
    results.reserve(entries.size());
    for (auto& e : entries) {
        results.push_back(std::move(e.path));
    }
    return results;
}

} // namespace astro
