#include "sdr_device.h"
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#endif
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

namespace {

std::vector<float> default_r820t_gains() {
    return {0.0f, 0.9f, 1.4f, 2.7f, 3.7f, 7.7f, 8.7f, 12.5f, 14.4f, 15.7f, 16.6f,
            19.7f, 20.7f, 22.9f, 25.4f, 28.0f, 29.7f, 32.8f, 33.8f, 36.4f, 37.2f,
            38.6f, 40.2f, 42.1f, 43.4f, 43.9f, 44.5f, 48.0f, 49.6f};
}

#ifdef _WIN32
std::string to_utf8(const std::wstring& wstr) {
    if (wstr.empty() || wstr.size() > static_cast<size_t>(INT_MAX)) return std::string();
    const int src_len = static_cast<int>(wstr.size());
    const int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), src_len, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return std::string();
    std::string result(static_cast<size_t>(size), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), src_len, &result[0], size, nullptr, nullptr);
    if (written <= 0) return std::string();
    return result;
}
#endif

} // namespace

RtlSdrDevice::RtlSdrDevice() {
    load_dll();
}

RtlSdrDevice::~RtlSdrDevice() {
    close();
    unload_dll();
}

std::string RtlSdrDevice::get_name() const {
    std::lock_guard<std::mutex> lk(info_mutex_);
    return device_name_;
}

std::string RtlSdrDevice::get_status_detail() const {
    std::lock_guard<std::mutex> lk(info_mutex_);
    return status_detail_;
}

void RtlSdrDevice::set_status_detail(std::string detail) {
    std::lock_guard<std::mutex> lk(info_mutex_);
    status_detail_ = std::move(detail);
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
        set_status_detail("rtlsdr.dll could not be loaded.");
        return false;
    }

    dll_handle_ = reinterpret_cast<void*>(mod);

    #define LOAD_PROC(name) fn_##name = reinterpret_cast<pfn_rtlsdr_##name>(reinterpret_cast<uintptr_t>(GetProcAddress(mod, "rtlsdr_" #name)))
#else
    const char* so_candidates[] = {"librtlsdr.so.0", "librtlsdr.so"};
    void* mod = nullptr;
    for (const char* path : so_candidates) {
        mod = dlopen(path, RTLD_LAZY);
        if (mod) break;
    }

    if (!mod) {
        set_status_detail("librtlsdr shared library not found; virtual mode available.");
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
        set_status_detail("RTL-SDR library is missing required API exports.");
        unload_dll();
        return false;
    }

    set_status_detail("RTL-SDR library loaded.");
    return true;
}

void RtlSdrDevice::unload_dll() {
    if (dll_handle_) {
#ifdef _WIN32
        FreeLibrary(reinterpret_cast<HMODULE>(dll_handle_));
#else
        dlclose(dll_handle_);
#endif
        dll_handle_ = nullptr;
    }

    fn_get_device_count = nullptr;
    fn_get_device_name = nullptr;
    fn_get_device_usb_strings = nullptr;
    fn_open = nullptr;
    fn_close = nullptr;
    fn_set_center_freq = nullptr;
    fn_get_center_freq = nullptr;
    fn_set_sample_rate = nullptr;
    fn_get_sample_rate = nullptr;
    fn_set_tuner_gain_mode = nullptr;
    fn_set_tuner_gain = nullptr;
    fn_get_tuner_gains = nullptr;
    fn_set_freq_correction = nullptr;
    fn_reset_buffer = nullptr;
    fn_read_sync = nullptr;
    fn_set_bias_tee = nullptr;
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

void RtlSdrDevice::cache_tuner_gains() {
    std::vector<float> gains;
    if (dev_ && fn_get_tuner_gains) {
        const int count = fn_get_tuner_gains(dev_, nullptr);
        if (count > 0 && count <= 4096) {
            std::vector<int> gains_buf(static_cast<size_t>(count), 0);
            const int n = fn_get_tuner_gains(dev_, gains_buf.data());
            const int valid_n = std::min(n, count);
            if (valid_n > 0) {
                gains.reserve(static_cast<size_t>(valid_n));
                for (int i = 0; i < valid_n; ++i) {
                    gains.push_back(static_cast<float>(gains_buf[static_cast<size_t>(i)]) / 10.0f);
                }
            }
        }
    }
    if (gains.empty()) {
        gains = default_r820t_gains();
    }
    std::lock_guard<std::mutex> lk(info_mutex_);
    cached_valid_gains_ = std::move(gains);
}

bool RtlSdrDevice::open(uint32_t index) {
    close();

    auto blocking = detect_blocking_processes();
    if (!blocking.empty()) {
        set_status_detail("USB device held by active SDR process. Running in Virtual mode.");
        return false;
    }

    if (!load_dll() || !fn_open) {
        if (get_status_detail().empty()) set_status_detail("RTL-SDR library unavailable.");
        return false;
    }

    uint32_t count = fn_get_device_count ? fn_get_device_count() : 0;
    if (count == 0 || index >= count) {
        set_status_detail("No hardware RTL-SDR dongle detected.");
        return false;
    }

    // Query USB strings BEFORE opening device handle to avoid transient libusb open conflicts
    std::string probed_name;
    {
        char mfg[256] = {0}, prod[256] = {0}, ser[256] = {0};
        const int usb_ret = fn_get_device_usb_strings ? fn_get_device_usb_strings(index, mfg, prod, ser) : -1;
        if (usb_ret == 0 && (prod[0] != '\0' || mfg[0] != '\0')) {
            probed_name = std::string(mfg) + " " + prod;
            if (ser[0] != '\0') probed_name += " #" + std::string(ser);
        } else if (fn_get_device_name) {
            const char* name = fn_get_device_name(index);
            probed_name = name ? name : ("Generic RTL2832U #" + std::to_string(index));
        } else {
            probed_name = "RTL-SDR #" + std::to_string(index);
        }
    }

    void* dev = nullptr;
    const int ret = fn_open(&dev, index);
    if (ret != 0 || !dev) {
        set_status_detail("Hardware found but open failed (error code " + std::to_string(ret) + ").");
        return false;
    }

    dev_ = dev;
    {
        std::lock_guard<std::mutex> lk(info_mutex_);
        device_name_ = probed_name;
        status_detail_ = "Connected: " + device_name_;
    }

    // Cache discrete hardware gain table once on open so get_valid_gains() never performs USB I/O during streaming
    cache_tuner_gains();

    // Apply initial configuration
    set_sample_rate(sample_rate_hz_.load());
    set_frequency(center_freq_hz_.load());
    set_gain(gain_db_.load(), auto_gain_.load());
    set_bias_tee(bias_tee_.load());
    set_ppm(ppm_.load());

    return true;
}

void RtlSdrDevice::close() {
    if (dev_ && fn_close) {
        if (fn_set_bias_tee && bias_tee_.load()) {
            fn_set_bias_tee(dev_, 0);
        }
        fn_close(dev_);
        dev_ = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(info_mutex_);
        cached_valid_gains_.clear();
    }
}

std::vector<float> RtlSdrDevice::get_valid_gains() {
    std::lock_guard<std::mutex> lk(info_mutex_);
    if (!cached_valid_gains_.empty()) {
        return cached_valid_gains_;
    }
    return default_r820t_gains();
}

bool RtlSdrDevice::set_sample_rate(uint32_t rate_hz) {
    if (rate_hz == 0 || !dev_ || !fn_set_sample_rate) {
        return false;
    }
    const int ret = fn_set_sample_rate(dev_, rate_hz);
    if (ret != 0) {
        return false;
    }
    if (fn_get_sample_rate) {
        const uint32_t actual = fn_get_sample_rate(dev_);
        sample_rate_hz_.store(actual > 0 ? actual : rate_hz);
    } else {
        sample_rate_hz_.store(rate_hz);
    }
    if (fn_reset_buffer) fn_reset_buffer(dev_);
    return true;
}

bool RtlSdrDevice::set_frequency(uint64_t freq_hz) {
    if (freq_hz == 0 || freq_hz > 0xFFFFFFFFULL) {
        set_status_detail("Frequency out of 32-bit RTL-SDR tuner range.");
        return false;
    }
    if (!dev_ || !fn_set_center_freq) {
        return false;
    }

    int ret = -1;
    // Retry up to 3 times in case of transient I2C bus collision
    for (int attempt = 0; attempt < 3; ++attempt) {
        ret = fn_set_center_freq(dev_, static_cast<uint32_t>(freq_hz));
        if (ret == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (ret != 0) {
        set_status_detail("Error: rtlsdr_set_center_freq failed after retries (code " + std::to_string(ret) + ")");
        return false;
    }

    // Verify PLL lock by querying tuner frequency back from driver
    if (fn_get_center_freq) {
        const uint32_t actual = fn_get_center_freq(dev_);
        if (actual == 0) {
            set_status_detail("Error: rtlsdr_get_center_freq returned 0 (Tuner PLL lock check failed)");
            return false;
        }
        center_freq_hz_.store(actual);
        if (std::abs(static_cast<int64_t>(actual) - static_cast<int64_t>(freq_hz)) > 200000LL) {
            set_status_detail("Tuned to " + std::to_string(static_cast<double>(actual) / 1e6) + " MHz (PLL clamped/quantized)");
        } else {
            std::lock_guard<std::mutex> lk(info_mutex_);
            status_detail_ = "Connected: " + device_name_;
        }
    } else {
        center_freq_hz_.store(freq_hz);
    }

    if (fn_reset_buffer) {
        fn_reset_buffer(dev_);
    }
    return true;
}

bool RtlSdrDevice::set_gain(float gain_db, bool auto_gain) {
    if (!dev_ || !fn_set_tuner_gain_mode || !std::isfinite(gain_db)) {
        return false;
    }

    if (auto_gain) {
        if (fn_set_tuner_gain_mode(dev_, 0) != 0) {
            return false;
        }
        auto_gain_.store(true);
        gain_db_.store(gain_db);
        return true;
    }

    if (!fn_set_tuner_gain) {
        return false;
    }

    // Snap requested gain to closest discrete hardware gain step from cached table
    float target_gain = gain_db;
    const auto valid = get_valid_gains();
    if (!valid.empty()) {
        float best = valid.front();
        float best_diff = std::abs(gain_db - best);
        for (float g : valid) {
            const float d = std::abs(gain_db - g);
            if (d < best_diff) {
                best_diff = d;
                best = g;
            }
        }
        target_gain = best;
    }

    if (fn_set_tuner_gain_mode(dev_, 1) != 0) {
        return false;
    }
    if (fn_set_tuner_gain(dev_, static_cast<int>(std::lround(target_gain * 10.0f))) != 0) {
        return false;
    }

    auto_gain_.store(false);
    gain_db_.store(target_gain);
    return true;
}

bool RtlSdrDevice::set_bias_tee(bool enabled) {
    if (!dev_ || !fn_set_bias_tee) {
        return false;
    }
    if (fn_set_bias_tee(dev_, enabled ? 1 : 0) != 0) {
        return false;
    }
    bias_tee_.store(enabled);
    return true;
}

bool RtlSdrDevice::set_ppm(int ppm) {
    if (!dev_ || !fn_set_freq_correction) {
        return false;
    }
    // librtlsdr returns -2 when the requested PPM equals the currently active correction (no-op)
    const int ret = fn_set_freq_correction(dev_, ppm);
    if (ret != 0 && ret != -2) {
        return false;
    }
    ppm_.store(ppm);
    return true;
}

int RtlSdrDevice::read_samples(uint8_t* buffer, int length, int* bytes_read) {
    if (bytes_read) *bytes_read = 0;
    if (!buffer || length <= 0 || !dev_ || !fn_read_sync) {
        return -1;
    }
    int local_read = 0;
    const int ret = fn_read_sync(dev_, buffer, length, &local_read);
    if (bytes_read) *bytes_read = local_read;
    return ret;
}

void RtlSdrDevice::reset_buffer() {
    if (dev_ && fn_reset_buffer) {
        fn_reset_buffer(dev_);
    }
}
