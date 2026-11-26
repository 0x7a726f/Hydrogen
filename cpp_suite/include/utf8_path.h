#pragma once

#include <climits>
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace astro_fs {

#ifdef _WIN32
inline std::wstring utf8_to_wide(const std::string& str) {
    if (str.empty() || str.size() > static_cast<size_t>(INT_MAX)) return std::wstring();
    const int src_len = static_cast<int>(str.size());
    const int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), src_len, nullptr, 0);
    if (size <= 0) return std::wstring();
    std::wstring result(static_cast<size_t>(size), L'\0');
    const int written = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), src_len, &result[0], size);
    if (written <= 0) return std::wstring();
    return result;
}

inline std::string wide_to_utf8(const std::wstring& wstr) {
    if (wstr.empty() || wstr.size() > static_cast<size_t>(INT_MAX)) return std::string();
    const int src_len = static_cast<int>(wstr.size());
    const int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), src_len, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return std::string();
    std::string result(static_cast<size_t>(size), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), src_len, &result[0], size, nullptr, nullptr);
    if (written <= 0) return std::string();
    return result;
}

inline std::filesystem::path utf8_path(const std::string& str) {
    if (str.empty()) return std::filesystem::path();
    std::wstring w = utf8_to_wide(str);
    return w.empty() ? std::filesystem::path(str) : std::filesystem::path(std::move(w));
}

inline std::string path_to_utf8(const std::filesystem::path& p) {
    return wide_to_utf8(p.wstring());
}

inline std::string path_to_generic_utf8(const std::filesystem::path& p) {
    return wide_to_utf8(p.generic_wstring());
}
#else
inline std::filesystem::path utf8_path(const std::string& str) {
    return std::filesystem::path(str);
}

inline std::string path_to_utf8(const std::filesystem::path& p) {
    return p.string();
}

inline std::string path_to_generic_utf8(const std::filesystem::path& p) {
    return p.generic_string();
}
#endif

} // namespace astro_fs
