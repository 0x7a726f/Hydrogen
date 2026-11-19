#include "sidecar_reader.h"
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

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

// JSON key-value scanner that properly parses and skips quoted string literals
static size_t find_json_value_start(const std::string& content, const std::string& key) {
    const size_t n = content.size();
    size_t i = 0;
    while (i < n) {
        if (content[i] == '"') {
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
                if (decoded_token == key) {
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

static bool json_extract_number(const std::string& content, const std::string& key, double& out_val) {
    size_t p = find_json_value_start(content, key);
    if (p == std::string::npos || p >= content.size()) return false;
    const char* start = content.c_str() + p;
    char* end = nullptr;
    double val = std::strtod(start, &end);
    if (end != start && std::isfinite(val)) {
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
                            int cp = (h0 << 12) | (h1 << 8) | (h2 << 4) | h3;
                            if (cp < 0x80) {
                                decoded.push_back(static_cast<char>(cp));
                            } else {
                                decoded.push_back('?');
                            }
                            i += 4;
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
    std::ifstream file(json_path);
    if (!file.is_open()) return false;

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    bool has_data = false;
    double val = 0.0;
    if (json_extract_number(content, "center_freq_hz", val) ||
        json_extract_number(content, "center_frequency_hz", val) ||
        json_extract_number(content, "fc_hz", val)) {
        meta.center_freq_hz = val;
        has_data = true;
    }
    if (json_extract_number(content, "sample_rate_hz", val) ||
        json_extract_number(content, "sampling_rate_hz", val) ||
        json_extract_number(content, "fs_hz", val)) {
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
        meta.sidecar_source = std::filesystem::path(json_path).filename().string();
        return true;
    }
    return false;
}

bool SidecarReader::parse_txt_sidecar(const std::string& txt_path, ObservationMetadata& meta) {
    std::ifstream file(txt_path);
    if (!file.is_open()) return false;

    std::string line;
    bool has_data = false;

    while (std::getline(file, line)) {
        if (line.find("Tuned SDR Center LO:") != std::string::npos ||
            line.find("Center Frequency (LO):") != std::string::npos ||
            line.find("Center Frequency:") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                double parsed_fc = 0.0;
                if (try_parse_frequency_or_rate(val_str, parsed_fc, true)) {
                    meta.center_freq_hz = parsed_fc;
                    has_data = true;
                }
            }
        } else if (line.find("Sampling Rate (Fs):") != std::string::npos ||
                   line.find("SDR Sample Rate:") != std::string::npos ||
                   line.find("Sample Rate:") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                double parsed_fs = 0.0;
                if (try_parse_sample_rate(val_str, parsed_fs)) {
                    meta.sample_rate_hz = parsed_fs;
                    has_data = true;
                }
            }
        } else if (line.find("RF Gain:") != std::string::npos || line.find("Gain:") != std::string::npos) {
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
        } else if (line.find("Bias-Tee") != std::string::npos || line.find("Bias Tee") != std::string::npos) {
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
        } else if (line.find("LO Frequency Offset:") != std::string::npos ||
                   line.find("LO Offset") != std::string::npos ||
                   line.find("Offset Shift:") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string val_str = trim(line.substr(colon + 1));
                val_str.erase(std::remove(val_str.begin(), val_str.end(), ','), val_str.end());
                try {
                    double parsed_off = std::stod(val_str);
                    if (std::isfinite(parsed_off)) {
                        meta.lo_offset_khz = parsed_off;
                        has_data = true;
                    }
                } catch (...) {}
            }
        } else if (line.find("Capture Timestamp:") != std::string::npos ||
                   line.find("Timestamp:") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string ts = trim(line.substr(colon + 1));
                if (!ts.empty()) {
                    meta.timestamp = std::move(ts);
                    has_data = true;
                }
            }
        } else if (line.find("Device:") != std::string::npos || line.find("Frontend Device:") != std::string::npos ||
                   line.find("Device Name:") != std::string::npos || line.find("Receiver Hardware:") != std::string::npos) {
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
        meta.sidecar_source = std::filesystem::path(txt_path).filename().string();
        return true;
    }
    return false;
}

bool SidecarReader::parse_filename_fallback(const std::string& cs16_path, ObservationMetadata& meta) {
    std::string fname = std::filesystem::path(cs16_path).filename().string();
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
    std::filesystem::path p(cs16_path);
    std::filesystem::path base_path = p.parent_path() / p.stem();

    std::error_code ec;
    // 1. Try .json sidecar
    std::filesystem::path json_path = base_path;
    json_path += ".json";
    if (std::filesystem::exists(json_path, ec) && parse_json_sidecar(json_path.string(), meta)) {
        return meta;
    }

    // 2. Try .txt sidecar
    std::filesystem::path txt_path = base_path;
    txt_path += ".txt";
    if (std::filesystem::exists(txt_path, ec) && parse_txt_sidecar(txt_path.string(), meta)) {
        return meta;
    }

    // 3. Fallback: Parse from filename tokens
    parse_filename_fallback(cs16_path, meta);
    return meta;
}

bool SidecarReader::apply_to_config(const ObservationMetadata& meta, AstroConfig& config) {
    if (!meta.found_sidecar) return false;
    if (meta.center_freq_hz > 1e5) {
        config.center_freq_hz = meta.center_freq_hz;
    }
    if (meta.sample_rate_hz > 1e4) {
        config.sample_rate_hz = meta.sample_rate_hz;
    }
    return true;
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
        if (!std::filesystem::exists(path_str, ec) || ec) continue;

        std::filesystem::recursive_directory_iterator it(
            path_str, std::filesystem::directory_options::skip_permission_denied, ec);
        std::filesystem::recursive_directory_iterator end;
        if (ec) continue;

        for (; !ec && it != end; it.increment(ec)) {
            const auto& entry = *it;
            std::error_code file_ec;
            if (!entry.is_regular_file(file_ec) || file_ec) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext != ".cs16") continue;

            std::error_code canon_ec;
            auto canon = std::filesystem::weakly_canonical(entry.path(), canon_ec);
            std::error_code abs_ec;
            std::string abs_path = std::filesystem::absolute(entry.path(), abs_ec).string();
            if (abs_ec || abs_path.empty()) abs_path = entry.path().string();

            std::string canon_key = canon_ec ? std::filesystem::path(abs_path).generic_string() : canon.generic_string();
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
