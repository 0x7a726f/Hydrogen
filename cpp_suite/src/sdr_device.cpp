#include "sdr_device.h"
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#endif
#include <algorithm>
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

#ifdef _WIN32
static std::string to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    std::string result(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &result[0], size, nullptr, nullptr);
    return result;
}
#endif

RtlSdrDevice::RtlSdrDevice() {
    load_dll();
}

RtlSdrDevice::~RtlSdrDevice() {
    close();
    unload_dll();
}

bool RtlSdrDevice::load_dll() {
    if (dll_handle_) return true;

#ifdef _WIN32
    std::vector<std::wstring> candidates;
    wchar_t exe_path[MAX_PATH] = {0};
    if (GetModuleFileNameW(nullptr, exe_path, MAX_PATH)) {
        wchar_t* last_slash = wcsrchr(exe_path, L'\\');
        if (last_slash) {
            *last_slash = L'\0';
            candidates.push_back(std::wstring(exe_path) + L"\\rtlsdr.dll");
            candidates.push_back(std::wstring(exe_path) + L"\\bin\\rtlsdr.dll");
        }
    }
    candidates.push_back(L"rtlsdr.dll");

    HMODULE mod = nullptr;
    for (const auto& path : candidates) {
        mod = LoadLibraryW(path.c_str());
        if (mod) break;
    }

    if (!mod) {
        status_detail_ = "rtlsdr.dll could not be loaded.";
        return false;
    }

    dll_handle_ = (void*)mod;

    #define LOAD_PROC(name) fn_##name = reinterpret_cast<pfn_rtlsdr_##name>(reinterpret_cast<uintptr_t>(GetProcAddress(mod, "rtlsdr_" #name)))
#else
    const char* so_candidates[] = {"librtlsdr.so.0", "librtlsdr.so"};
    void* mod = nullptr;
    for (const char* path : so_candidates) {
        mod = dlopen(path, RTLD_LAZY);
        if (mod) break;
    }

    if (!mod) {
        status_detail_ = "librtlsdr shared library not found; virtual mode available.";
        return false;
    }

    dll_handle_ = mod;

    #define LOAD_PROC(name) fn_##name = reinterpret_cast<pfn_rtlsdr_##name>(dlsym(mod, "rtlsdr_" #name))
#endif
    LOAD_PROC(get_device_count);
    LOAD_PROC(get_device_name);
    LOAD_PROC(get_device_usb_strings);
    LOAD_PROC(open);
    LOAD_PROC(close);
    LOAD_PROC(set_center_freq);
    LOAD_PROC(get_center_freq);
    LOAD_PROC(set_sample_rate);
    LOAD_PROC(get_sample_rate);
    LOAD_PROC(set_tuner_gain_mode);
    LOAD_PROC(set_tuner_gain);
    LOAD_PROC(get_tuner_gains);
    LOAD_PROC(set_freq_correction);
    LOAD_PROC(reset_buffer);
    LOAD_PROC(read_sync);
    LOAD_PROC(set_bias_tee);
    #undef LOAD_PROC

    if (!fn_open || !fn_close || !fn_read_sync) {
        status_detail_ = "RTL-SDR library is missing required API exports.";
        unload_dll();
        return false;
    }

    status_detail_ = "RTL-SDR library loaded.";
    return true;
}

void RtlSdrDevice::unload_dll() {
    if (dll_handle_) {
#ifdef _WIN32
        FreeLibrary((HMODULE)dll_handle_);
#else
        dlclose(dll_handle_);
#endif
        dll_handle_ = nullptr;
    }
}

std::vector<std::string> RtlSdrDevice::detect_blocking_processes() {
    std::vector<std::string> blocking;
#ifdef _WIN32
    static std::mutex cache_mutex;
    static std::vector<std::string> cached_result;
    static uint64_t last_probe_ms = 0;
    static bool has_cached = false;

    {
        std::lock_guard<std::mutex> lk(cache_mutex);
        const uint64_t now_ms = GetTickCount64();
        if (has_cached && (now_ms - last_probe_ms) < 1500ULL) {
            return cached_result;
        }
    }

    static const wchar_t* const known_sdr_w[] = {
        L"satdump", L"sdrpp", L"sdrangel", L"tv_app", L"rtl_tcp", L"dump1090", L"welle"
    };

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return blocking;

    PROCESSENTRY32W pe;
    pe.dwSize = sizeof(pe);

    if (Process32FirstW(snap, &pe)) {
        do {
            wchar_t lower_exe[MAX_PATH];
            size_t len = 0;
            while (len + 1 < MAX_PATH && pe.szExeFile[len] != L'\0') {
                wchar_t c = pe.szExeFile[len];
                if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c + 32);
                lower_exe[len++] = c;
            }
            lower_exe[len] = L'\0';
            for (const wchar_t* target : known_sdr_w) {
                if (wcsstr(lower_exe, target) != nullptr) {
                    blocking.push_back(to_utf8(pe.szExeFile));
                    break;
                }
            }
        } while (Process32NextW(snap, &pe));
    }

    CloseHandle(snap);

    {
        std::lock_guard<std::mutex> lk(cache_mutex);
        cached_result = blocking;
        last_probe_ms = GetTickCount64();
        has_cached = true;
    }
#endif
    return blocking;
}

std::vector<std::string> RtlSdrDevice::scan_devices() {
    RtlSdrDevice dev_probe;
    if (!dev_probe.load_dll() || !dev_probe.fn_get_device_count) {
        return {"Virtual SDR Mode (Simulation)"};
    }

    uint32_t count = dev_probe.fn_get_device_count();
    if (count == 0) {
        return {"Virtual SDR Mode (Simulation)"};
    }

    std::vector<std::string> result;
    for (uint32_t i = 0; i < count; ++i) {
        char mfg[256] = {0}, prod[256] = {0}, ser[256] = {0};
        int ret = -1;
        if (dev_probe.fn_get_device_usb_strings) {
            ret = dev_probe.fn_get_device_usb_strings(i, mfg, prod, ser);
        }
        if (ret == 0 && (prod[0] != '\0' || mfg[0] != '\0')) {
            std::string name = std::string(mfg) + " " + prod;
            if (ser[0] != '\0') {
                name += " #" + std::string(ser);
            }
            result.push_back(name);
        } else if (dev_probe.fn_get_device_name) {
            const char* name = dev_probe.fn_get_device_name(i);
            result.push_back(name ? name : "Generic RTL2832U #" + std::to_string(i));
        } else {
            result.push_back("Generic RTL2832U #" + std::to_string(i));
        }
    }
    return result;
}

bool RtlSdrDevice::open(uint32_t index) {
    close();

    auto blocking = detect_blocking_processes();
    if (!blocking.empty()) {
        status_detail_ = "USB device held by active SDR process. Running in Virtual mode.";
        return false;
    }

    if (!load_dll() || !fn_open) {
        if (status_detail_.empty()) status_detail_ = "RTL-SDR library unavailable.";
        return false;
    }

    uint32_t count = fn_get_device_count ? fn_get_device_count() : 0;
    if (count == 0 || index >= count) {
        status_detail_ = "No hardware RTL-SDR dongle detected.";
        return false;
    }

    void* dev = nullptr;
    int ret = fn_open(&dev, index);
    if (ret != 0 || !dev) {
        status_detail_ = "Hardware found but open failed (error code " + std::to_string(ret) + ").";
        return false;
    }

    dev_ = dev;
    {
        char mfg[256] = {0}, prod[256] = {0}, ser[256] = {0};
        int usb_ret = fn_get_device_usb_strings ? fn_get_device_usb_strings(index, mfg, prod, ser) : -1;
        if (usb_ret == 0 && (prod[0] != '\0' || mfg[0] != '\0')) {
            device_name_ = std::string(mfg) + " " + prod;
            if (ser[0] != '\0') device_name_ += " #" + std::string(ser);
        } else if (fn_get_device_name) {
            const char* name = fn_get_device_name(index);
            device_name_ = name ? name : ("Generic RTL2832U #" + std::to_string(index));
        } else {
            device_name_ = "RTL-SDR #" + std::to_string(index);
        }
    }
    status_detail_ = "Connected: " + device_name_;

    // Set initial configuration
    set_sample_rate(sample_rate_hz_);
    set_frequency(center_freq_hz_);
    set_gain(gain_db_, auto_gain_);
    set_bias_tee(bias_tee_);
    set_ppm(ppm_);

    return true;
}

void RtlSdrDevice::close() {
    if (dev_ && fn_close) {
        if (fn_set_bias_tee && bias_tee_) {
            fn_set_bias_tee(dev_, 0);
        }
        fn_close(dev_);
        dev_ = nullptr;
    }
}

std::vector<float> RtlSdrDevice::get_valid_gains() {
    if (dev_ && fn_get_tuner_gains) {
        int gains_buf[64] = {0};
        int n = fn_get_tuner_gains(dev_, gains_buf);
        if (n > 0) {
            std::vector<float> list;
            list.reserve(n);
            for (int i = 0; i < n; ++i) {
                list.push_back(gains_buf[i] / 10.0f);
            }
            return list;
        }
    }
    // Fallback standard R820T / R828D gains
    return {0.0f, 0.9f, 1.4f, 2.7f, 3.7f, 7.7f, 8.7f, 12.5f, 14.4f, 15.7f, 16.6f,
            19.7f, 20.7f, 22.9f, 25.4f, 28.0f, 29.7f, 32.8f, 33.8f, 36.4f, 37.2f,
            38.6f, 40.2f, 42.1f, 43.4f, 43.9f, 44.5f, 48.0f, 49.6f};
}

bool RtlSdrDevice::set_sample_rate(uint32_t rate_hz) {
    sample_rate_hz_ = rate_hz;
    if (dev_ && fn_set_sample_rate) {
        int ret = fn_set_sample_rate(dev_, rate_hz);
        if (fn_reset_buffer) fn_reset_buffer(dev_);
        return (ret == 0);
    }
    return false;
}

bool RtlSdrDevice::set_frequency(uint64_t freq_hz) {
    if (freq_hz == 0 || freq_hz > 0xFFFFFFFFULL) {
        status_detail_ = "Frequency out of 32-bit RTL-SDR tuner range.";
        return false;
    }
    if (dev_ && fn_set_center_freq) {
        int ret = -1;
        // Retry up to 3 times in case of transient I2C bus collision
        for (int attempt = 0; attempt < 3; ++attempt) {
            ret = fn_set_center_freq(dev_, static_cast<uint32_t>(freq_hz));
            if (ret == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        if (ret != 0) {
            status_detail_ = "Error: rtlsdr_set_center_freq failed after retries (code " + std::to_string(ret) + ")";
            return false;
        }

        // Verify PLL lock by querying tuner frequency back from driver
        if (fn_get_center_freq) {
            uint32_t actual = fn_get_center_freq(dev_);
            if (actual == 0) {
                status_detail_ = "Warning: rtlsdr_get_center_freq returned 0 (Tuner PLL lock check failed)";
            } else {
                center_freq_hz_ = actual;
                if (std::abs((int64_t)actual - (int64_t)freq_hz) > 200000) {
                    status_detail_ = "Tuned to " + std::to_string(actual / 1e6) + " MHz (PLL clamped/quantized)";
                } else {
                    status_detail_ = "Connected: " + device_name_;
                }
            }
        } else {
            center_freq_hz_ = freq_hz;
        }

        if (fn_reset_buffer) {
            fn_reset_buffer(dev_);
        }
        return true;
    }
    return false;
}

bool RtlSdrDevice::set_gain(float gain_db, bool auto_gain) {
    auto_gain_ = auto_gain;
    if (!auto_gain) {
        // Snap requested gain to closest discrete hardware gain step
        auto valid = get_valid_gains();
        if (!valid.empty()) {
            float best = valid.front();
            float best_diff = std::abs(gain_db - best);
            for (float g : valid) {
                float d = std::abs(gain_db - g);
                if (d < best_diff) {
                    best_diff = d;
                    best = g;
                }
            }
            gain_db_ = best;
        } else {
            gain_db_ = gain_db;
        }
    } else {
        gain_db_ = gain_db;
    }

    if (dev_ && fn_set_tuner_gain_mode) {
        if (auto_gain) {
            return (fn_set_tuner_gain_mode(dev_, 0) == 0);
        } else {
            if (fn_set_tuner_gain_mode(dev_, 1) != 0) return false;
            if (fn_set_tuner_gain) {
                return (fn_set_tuner_gain(dev_, static_cast<int>(std::lround(gain_db_ * 10.0f))) == 0);
            }
        }
    }
    return false;
}

bool RtlSdrDevice::set_bias_tee(bool enabled) {
    bias_tee_ = enabled;
    if (dev_ && fn_set_bias_tee) {
        return (fn_set_bias_tee(dev_, enabled ? 1 : 0) == 0);
    }
    return false;
}

bool RtlSdrDevice::set_ppm(int ppm) {
    ppm_ = ppm;
    if (dev_ && fn_set_freq_correction) {
        return (fn_set_freq_correction(dev_, ppm) == 0);
    }
    return false;
}

int RtlSdrDevice::read_samples(uint8_t* buffer, int length, int* bytes_read) {
    if (dev_ && fn_read_sync) {
        return fn_read_sync(dev_, buffer, length, bytes_read);
    }
    return -1;
}

void RtlSdrDevice::reset_buffer() {
    if (dev_ && fn_reset_buffer) {
        fn_reset_buffer(dev_);
    }
}
