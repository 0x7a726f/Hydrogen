#include "cs16_writer.h"
#include "simd_dsp.h"
#include "utf8_path.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

namespace fs = std::filesystem;

static void safe_localtime(std::time_t tt, std::tm& out) {
#ifdef _WIN32
    localtime_s(&out, &tt);
#else
    localtime_r(&tt, &out);
#endif
}

Cs16Writer::Cs16Writer(const std::string& output_dir)
    : output_dir_(output_dir) {
    std::error_code ec;
    fs::create_directories(astro_fs::utf8_path(output_dir_), ec);
}

Cs16Writer::~Cs16Writer() {
    stop_recording();
}

static std::string format_iso_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
    safe_localtime(tt, tm_now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_now);
    return std::string(buf);
}

static std::string format_time_slug_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
    safe_localtime(tt, tm_now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_now);
    return std::string(buf);
}

std::string Cs16Writer::start_recording(
    uint32_t sample_rate_hz,
    uint64_t center_freq_hz,
    const std::string& device_name,
    float gain_db,
    bool bias_tee,
    const std::string& custom_filepath,
    double max_duration_s
) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_recording_.load(std::memory_order_acquire)) {
        return get_current_filepath();
    }

    // Stop intake, join any finished auto-stopped worker thread, and drain all active feeders
    // before mutating shared state or resetting the ring buffer.
    // seq_cst Dekker handshake prevents Store-Load reordering with feed_samples().
    accepting_samples_.store(false, std::memory_order_seq_cst);
    if (writer_thread_.joinable()) {
        stop_worker_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
        }
        queue_cv_.notify_all();
        writer_thread_.join();
    }
    while (active_feeders_.load(std::memory_order_seq_cst) > 0) {
        std::this_thread::yield();
    }

    sample_rate_hz_.store(sample_rate_hz);
    center_freq_hz_ = center_freq_hz;
    device_name_ = device_name;
    gain_db_ = gain_db;
    bias_tee_ = bias_tee;
    max_duration_s_ = max_duration_s;
    iso_start_timestamp_ = format_iso_now();

    std::string path_to_use;
    if (!custom_filepath.empty()) {
        path_to_use = custom_filepath;
    } else {
        std::ostringstream ss;
        ss.imbue(std::locale::classic());
        ss << output_dir_ << "/hi_capture_" << format_time_slug_now() << "_"
           << std::fixed << std::setprecision(3) << (static_cast<double>(center_freq_hz) / 1e6) << "MHz_"
           << std::setprecision(2) << (static_cast<double>(sample_rate_hz) / 1e6) << "MSPS.cs16";
        path_to_use = ss.str();
    }

    std::error_code ec;
    const fs::path target_fs_path = astro_fs::utf8_path(path_to_use);
    fs::path parent_dir = target_fs_path.parent_path();
    if (!parent_dir.empty()) {
        fs::create_directories(parent_dir, ec);
    }

#ifdef _WIN32
    file_handle_ = CreateFileW(
        astro_fs::utf8_to_wide(path_to_use).c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        io_error_.store(true, std::memory_order_release);
        return "";
    }
#else
    {
        std::ofstream probe(target_fs_path, std::ios::binary | std::ios::trunc);
        if (!probe.is_open()) {
            io_error_.store(true, std::memory_order_release);
            return "";
        }
    }
#endif

    {
        std::lock_guard<std::mutex> lock(path_mutex_);
        current_filepath_ = path_to_use;
        start_time_ = std::chrono::steady_clock::now();
    }

    bytes_written_.store(0, std::memory_order_relaxed);
    samples_written_.store(0, std::memory_order_relaxed);
    dropped_blocks_.store(0, std::memory_order_relaxed);
    io_error_.store(false, std::memory_order_release);
    sidecars_written_.store(false, std::memory_order_release);
    stop_worker_.store(false, std::memory_order_release);

    try {
        if (!ring_buffer_) {
            ring_buffer_ = std::make_unique<LockFreeSpscRingBuffer<uint8_t>>(8ULL * 1024ULL * 1024ULL);
        } else {
            ring_buffer_->reset();
        }

        accepting_samples_.store(true, std::memory_order_seq_cst);
        is_recording_.store(true, std::memory_order_release);

        writer_thread_ = std::thread(&Cs16Writer::writer_worker, this);
    } catch (...) {
        accepting_samples_.store(false, std::memory_order_seq_cst);
        while (active_feeders_.load(std::memory_order_seq_cst) > 0) {
            std::this_thread::yield();
        }
        is_recording_.store(false, std::memory_order_release);
        io_error_.store(true, std::memory_order_release);
        sidecars_written_.store(true, std::memory_order_release);
#ifdef _WIN32
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#endif
        throw;
    }
    return path_to_use;
}

void Cs16Writer::finalize_sidecars(double duration_s) {
    if (sidecars_written_.exchange(true)) return;
    std::string cur_path = get_current_filepath();
    std::error_code ec;
    const fs::path p = astro_fs::utf8_path(cur_path);
    if (!cur_path.empty() && fs::exists(p, ec)) {
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            last_recorded_filepath_ = cur_path;
        }
        fs::path base = p.parent_path() / p.stem();
        const std::string base_utf8 = astro_fs::path_to_utf8(base);
        write_accompanied_txt(base_utf8 + ".txt", duration_s);
        write_sidecar_json(base_utf8 + ".json", duration_s);
    }
}

bool Cs16Writer::stop_recording() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    // seq_cst Dekker handshake prevents Store-Load reordering with feed_samples().
    accepting_samples_.store(false, std::memory_order_seq_cst);
    while (active_feeders_.load(std::memory_order_seq_cst) > 0) {
        std::this_thread::yield();
    }
    bool was_recording = is_recording_.load(std::memory_order_acquire);
    bool has_thread = writer_thread_.joinable();
    if (!was_recording && !has_thread) {
#ifdef _WIN32
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#endif
        return false;
    }

    stop_worker_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
    }
    queue_cv_.notify_all();

    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
#ifdef _WIN32
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#endif
    is_recording_.store(false, std::memory_order_release);

    std::chrono::steady_clock::time_point st;
    {
        std::lock_guard<std::mutex> lock(path_mutex_);
        st = start_time_;
    }
    auto now = std::chrono::steady_clock::now();
    double duration_s = std::chrono::duration<double>(now - st).count();
    finalize_sidecars(duration_s);

    return was_recording || has_thread;
}

std::string Cs16Writer::get_current_filepath() const {
    std::lock_guard<std::mutex> lock(path_mutex_);
    return current_filepath_;
}

std::string Cs16Writer::get_last_recorded_filepath() const {
    std::lock_guard<std::mutex> lock(path_mutex_);
    return last_recorded_filepath_;
}

void Cs16Writer::feed_samples(const uint8_t* data, size_t length) {
    // Enforce even I/Q byte alignment so odd transfers can never swap I and Q channels
    length &= ~size_t(1);
    if (!accepting_samples_.load(std::memory_order_acquire) || !ring_buffer_ || !data || length == 0) return;

    struct FeederGuard {
        std::atomic<int>& cnt;
        explicit FeederGuard(std::atomic<int>& c) : cnt(c) { cnt.fetch_add(1, std::memory_order_seq_cst); }
        ~FeederGuard() { cnt.fetch_sub(1, std::memory_order_release); }
    } guard(active_feeders_);

    if (!accepting_samples_.load(std::memory_order_seq_cst)) return;

    if (max_duration_s_ > 0.0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed >= max_duration_s_) {
            // Stop intake and wake writer_worker so it flushes, closes file & writes sidecars before clearing is_recording_
            accepting_samples_.store(false, std::memory_order_seq_cst);
            stop_worker_.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(queue_mutex_);
            }
            queue_cv_.notify_all();
            return;
        }
    }

    // All-or-nothing block admission prevents partial writes from breaking I/Q framing
    if (!ring_buffer_->write_exact(data, length)) {
        dropped_blocks_.fetch_add(1, std::memory_order_relaxed);
    } else {
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
        }
        queue_cv_.notify_one();
    }
}

void Cs16Writer::writer_worker() {
#ifdef _WIN32
    // Elevate thread priority to HIGHEST to prevent Windows IO thread stalls
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
#endif

#ifndef _WIN32
    // Declare file_buf BEFORE out so out closes/flushes before file_buf is destroyed,
    // and call pubsetbuf() BEFORE open() so libstdc++ honors the custom 2 MB buffer.
    std::vector<char> file_buf(2097152);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(file_buf.data(), static_cast<std::streamsize>(file_buf.size()));
    out.open(astro_fs::utf8_path(get_current_filepath()), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        // Prevent stop_recording() from writing sidecars / updating last_recorded_filepath_
        // for the 0-byte probe file after open failure.
        io_error_.store(true, std::memory_order_release);
        sidecars_written_.store(true, std::memory_order_release);
        accepting_samples_.store(false, std::memory_order_seq_cst);
        is_recording_.store(false, std::memory_order_release);
        return;
    }
#endif

    constexpr size_t kBatchCap = 2ULL * 1024ULL * 1024ULL; // 2 MB u8 -> 4 MB CS16 (even)
    std::vector<uint8_t> u8_staging(kBatchCap);
    std::vector<int16_t> cs16_staging(kBatchCap);

    auto check_duration_expired = [this]() -> bool {
        if (max_duration_s_ <= 0.0) return false;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed >= max_duration_s_) {
            accepting_samples_.store(false, std::memory_order_release);
            stop_worker_.store(true, std::memory_order_release);
            return true;
        }
        return false;
    };

    while (true) {
        check_duration_expired();

        size_t got = ring_buffer_ ? ring_buffer_->read(u8_staging.data(), kBatchCap) : 0;
        if (got == 0) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_worker_.load(std::memory_order_acquire) &&
                (!ring_buffer_ || ring_buffer_->available_read() == 0)) {
                break;
            }
            queue_cv_.wait_for(lock, std::chrono::milliseconds(25), [this] {
                return stop_worker_.load(std::memory_order_acquire) ||
                       (ring_buffer_ && ring_buffer_->available_read() > 0);
            });
            lock.unlock();

            check_duration_expired();
            got = ring_buffer_ ? ring_buffer_->read(u8_staging.data(), kBatchCap) : 0;
            if (got == 0 && stop_worker_.load(std::memory_order_acquire)) {
                break;
            }
            if (got == 0) continue;
        }

        got &= ~size_t(1);
        if (got == 0) continue;

        SimdDsp::convert_u8_to_cs16_fast(u8_staging.data(), cs16_staging.data(), got);

        const size_t total_bytes = got * sizeof(int16_t);
        bool write_ok = true;
#ifdef _WIN32
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(cs16_staging.data());
        size_t rem = total_bytes;
        while (rem > 0) {
            DWORD chunk = static_cast<DWORD>(std::min<size_t>(rem, 32ULL * 1024ULL * 1024ULL));
            DWORD written = 0;
            if (!WriteFile(file_handle_, ptr, chunk, &written, nullptr) || written != chunk) {
                write_ok = false;
                break;
            }
            ptr += written;
            rem -= written;
        }
#else
        out.write(reinterpret_cast<const char*>(cs16_staging.data()), static_cast<std::streamsize>(total_bytes));
        write_ok = out.good();
#endif

        if (write_ok) {
            bytes_written_.fetch_add(total_bytes, std::memory_order_relaxed);
            samples_written_.fetch_add(total_bytes / 4, std::memory_order_relaxed);
        } else {
            // Surface disk-full / I/O write failure cleanly and halt recording
            io_error_.store(true, std::memory_order_release);
            dropped_blocks_.fetch_add(1, std::memory_order_relaxed);
            accepting_samples_.store(false, std::memory_order_release);
            stop_worker_.store(true, std::memory_order_release);
            break;
        }
    }

#ifdef _WIN32
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(file_handle_);
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    out.flush();
    out.close();
#endif

    // Ensure sidecar metadata is emitted and file is finalized BEFORE clearing is_recording_
    auto now = std::chrono::steady_clock::now();
    double duration_s = std::chrono::duration<double>(now - start_time_).count();
    finalize_sidecars(duration_s);
    accepting_samples_.store(false, std::memory_order_release);
    is_recording_.store(false, std::memory_order_release);
}

static std::string escape_json_str(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (unsigned char uc : input) {
        switch (uc) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20u) {
                    char hex_buf[8];
                    std::snprintf(hex_buf, sizeof(hex_buf), "\\u%04x", static_cast<unsigned int>(uc));
                    out += hex_buf;
                } else {
                    out.push_back(static_cast<char>(uc));
                }
                break;
        }
    }
    return out;
}

void Cs16Writer::write_accompanied_txt(const std::string& txt_path, double duration_s) {
    double fc_hz = static_cast<double>(center_freq_hz_);
    double fc_mhz = fc_hz / 1e6;
    double fs_hz = static_cast<double>(sample_rate_hz_.load());
    double fs_msps = fs_hz / 1e6;
    double offset_hz = fc_hz - SdrConstants::HI_REST_FREQ_HZ;
    double offset_khz = offset_hz / 1e3;
    double dur_min = duration_s / 60.0;
    uint64_t total_bytes_w = bytes_written_.load();
    double size_mb = static_cast<double>(total_bytes_w) / (1024.0 * 1024.0);
    double size_gb = static_cast<double>(total_bytes_w) / (1024.0 * 1024.0 * 1024.0);
    const std::string cur_file = get_current_filepath();
    std::error_code ec;
    const fs::path cur_fs = astro_fs::utf8_path(cur_file);
    std::string cs16_filename = astro_fs::path_to_utf8(cur_fs.filename());
    std::string full_path = astro_fs::path_to_utf8(fs::absolute(cur_fs, ec));
    if (full_path.empty()) full_path = cur_file;

    double passband_min_mhz = (fc_hz - fs_hz / 2.0) / 1e6;
    double passband_max_mhz = (fc_hz + fs_hz / 2.0) / 1e6;

    std::string offset_desc;
    if (std::abs(offset_khz) < 50.0) {
        std::ostringstream ss;
        ss.imbue(std::locale::classic());
        ss << std::showpos << std::fixed << std::setprecision(2) << offset_khz
           << " kHz from HI rest (LO DC notch lies near the HI window)";
        offset_desc = ss.str();
    } else {
        std::ostringstream ss;
        ss.imbue(std::locale::classic());
        ss << std::showpos << std::fixed << std::setprecision(2) << offset_khz
           << " kHz from HI rest (LO DC spike is clear of the HI rest window)";
        offset_desc = ss.str();
    }

    std::ofstream f(astro_fs::utf8_path(txt_path));
    if (!f.is_open()) return;
    f.imbue(std::locale::classic());

    f << "HYDROGEN LINE (21cm) OBSERVATION METADATA\n"
      << "======================================================================\n"
      << "Recording File:             " << cs16_filename << "\n"
      << "Full Path:                  " << full_path << "\n"
      << "Capture Timestamp:          " << iso_start_timestamp_ << "\n"
      << "Receiver Hardware:          " << device_name_ << "\n\n"
      << "Frequencies & Offset\n"
      << "----------------------------------------------------------------------\n"
      << "HI Rest Frequency:          1420.405752 MHz\n"
      << std::fixed << std::setprecision(6)
      << "Tuned SDR Center LO:        " << fc_mhz << " MHz (" << static_cast<int64_t>(fc_hz) << " Hz)\n"
      << std::showpos << std::setprecision(3)
      << "LO Frequency Offset:        " << offset_khz << " kHz (" << static_cast<int64_t>(offset_hz) << " Hz)\n"
      << std::noshowpos
      << "Offset Assessment:          " << offset_desc << "\n\n"
      << "SDR Configuration\n"
      << "----------------------------------------------------------------------\n"
      << std::setprecision(3)
      << "Sampling Rate (Fs):         " << fs_msps << " MSPS (" << static_cast<int64_t>(fs_hz) << " Hz)\n"
      << "Passband Coverage:          " << passband_min_mhz << " to " << passband_max_mhz << " MHz\n"
      << std::setprecision(1)
      << "RF Gain:                    " << gain_db_ << " dB\n"
      << "Bias-Tee (4.5V):            " << (bias_tee_ ? "Enabled" : "Disabled") << "\n"
      << "Data Format:                CS16 (Signed 16-bit Complex IQ, Little-Endian)\n\n"
      << "Acquisition Summary\n"
      << "----------------------------------------------------------------------\n"
      << std::setprecision(2)
      << "Observation Duration:       " << duration_s << " s (" << dur_min << " min)\n"
      << "Total Complex Samples:      " << samples_written_.load() << "\n"
      << "Total File Size:            " << total_bytes_w << " bytes (" << size_mb << " MB / "
      << std::setprecision(3) << size_gb << " GB)\n"
      << std::setprecision(2)
      << "Nominal Data Rate:          " << ((fs_hz * 4.0) / (1024.0 * 1024.0)) << " MB/s\n\n"
      << "CLI Analysis Command\n"
      << "----------------------------------------------------------------------\n"
      << std::fixed << std::setprecision(6)
      << "  hydrogen_cli analyze \"" << cs16_filename << "\" --fc " << fc_mhz << "M --fs "
      << std::setprecision(3) << fs_msps << "M -o output\n";
    f.flush();
    f.close();
}

void Cs16Writer::write_sidecar_json(const std::string& json_path, double duration_s) {
    std::ofstream f(astro_fs::utf8_path(json_path));
    if (!f.is_open()) return;
    f.imbue(std::locale::classic());

    double offset_hz = static_cast<double>(center_freq_hz_) - SdrConstants::HI_REST_FREQ_HZ;
    double offset_khz = offset_hz / 1e3;

    std::string clean_path;
    {
        std::lock_guard<std::mutex> lock(path_mutex_);
        std::error_code ec;
        clean_path = astro_fs::path_to_generic_utf8(fs::absolute(astro_fs::utf8_path(current_filepath_), ec));
        if (clean_path.empty()) clean_path = current_filepath_;
    }

    f << "{\n"
      << "  \"format\": \"cs16\",\n"
      << "  \"endianness\": \"little\",\n"
      << "  \"bytes_per_sample\": 4,\n"
      << "  \"sample_rate_hz\": " << sample_rate_hz_.load() << ",\n"
      << "  \"center_freq_hz\": " << center_freq_hz_ << ",\n"
      << "  \"hi_rest_freq_hz\": " << std::fixed << std::setprecision(3) << SdrConstants::HI_REST_FREQ_HZ << ",\n"
      << std::fixed << std::setprecision(3)
      << "  \"lo_offset_hz\": " << offset_hz << ",\n"
      << "  \"lo_offset_khz\": " << offset_khz << ",\n"
      << "  \"timestamp\": \"" << escape_json_str(iso_start_timestamp_) << "\",\n"
      << std::setprecision(2)
      << "  \"duration_seconds\": " << duration_s << ",\n"
      << "  \"total_samples\": " << samples_written_.load() << ",\n"
      << "  \"file_size_bytes\": " << bytes_written_.load() << ",\n"
      << "  \"device\": \"" << escape_json_str(device_name_) << "\",\n"
      << std::setprecision(1)
      << "  \"gain_db\": " << gain_db_ << ",\n"
      << "  \"bias_tee\": " << (bias_tee_ ? "true" : "false") << ",\n"
      << "  \"file_path\": \"" << escape_json_str(clean_path) << "\"\n"
      << "}\n";
    f.flush();
    f.close();
}

RecordingStats Cs16Writer::get_stats() {
    RecordingStats s;
    const bool rec = is_recording_.load(std::memory_order_acquire);
    s.status = rec ? "RECORDING" : "IDLE";
    s.io_error = io_error_.load(std::memory_order_acquire);

    std::string cur_path;
    std::chrono::steady_clock::time_point st;
    {
        std::lock_guard<std::mutex> lock(path_mutex_);
        cur_path = current_filepath_;
        st = start_time_;
    }

    s.filename = cur_path.empty() ? "None" : astro_fs::path_to_utf8(astro_fs::utf8_path(cur_path).filename());
    s.full_path = cur_path;
    s.bytes_written = bytes_written_.load();
    s.samples_written = samples_written_.load();
    s.dropped_blocks = dropped_blocks_.load();

    if (rec) {
        auto now = std::chrono::steady_clock::now();
        s.elapsed_seconds = std::max(0.0, std::chrono::duration<double>(now - st).count());
    } else {
        s.elapsed_seconds = 0.0;
    }

    int total_s = static_cast<int>(s.elapsed_seconds);
    int hrs = total_s / 3600;
    int mins = (total_s % 3600) / 60;
    int secs = total_s % 60;
    char el_buf[32];
    std::snprintf(el_buf, sizeof(el_buf), "%02d:%02d:%02d", hrs, mins, secs);
    s.elapsed_str = el_buf;

    // File size string
    double mb = static_cast<double>(s.bytes_written) / (1024.0 * 1024.0);
    char sz_buf[32];
    if (mb >= 1024.0) {
        std::snprintf(sz_buf, sizeof(sz_buf), "%.2f GB", mb / 1024.0);
    } else {
        std::snprintf(sz_buf, sizeof(sz_buf), "%.2f MB", mb);
    }
    s.size_str = sz_buf;

    // Query free space on the actual target file's parent directory (fallback output_dir_ / ".")
    std::string space_dir = output_dir_.empty() ? "." : output_dir_;
    if (!cur_path.empty()) {
        fs::path parent = astro_fs::utf8_path(cur_path).parent_path();
        space_dir = parent.empty() ? "." : astro_fs::path_to_utf8(parent);
    }

    uint64_t free_bytes = 0;
    bool have_free = false;
#ifdef _WIN32
    ULARGE_INTEGER free_bytes_avail, total_bytes, total_free_bytes;
    std::wstring dir_w = astro_fs::utf8_to_wide(space_dir);
    if (!dir_w.empty() && GetDiskFreeSpaceExW(dir_w.c_str(), &free_bytes_avail, &total_bytes, &total_free_bytes)) {
        free_bytes = free_bytes_avail.QuadPart;
        have_free = true;
    }
#else
    std::error_code space_ec;
    auto si = fs::space(astro_fs::utf8_path(space_dir), space_ec);
    if (!space_ec) {
        free_bytes = static_cast<uint64_t>(si.available);
        have_free = true;
    }
#endif

    if (have_free) {
        s.free_space_gb = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
        char free_buf[32];
        std::snprintf(free_buf, sizeof(free_buf), "%.2f GB", s.free_space_gb);
        s.free_space_str = free_buf;

        uint64_t bytes_per_sec = static_cast<uint64_t>(sample_rate_hz_.load()) * 4ULL;
        if (bytes_per_sec > 0) {
            uint64_t time_left_s = free_bytes / bytes_per_sec;
            int tl_days = static_cast<int>(time_left_s / 86400);
            int tl_hrs = static_cast<int>((time_left_s % 86400) / 3600);
            int tl_mins = static_cast<int>((time_left_s % 3600) / 60);
            int tl_secs = static_cast<int>(time_left_s % 60);
            char tl_buf[64];
            if (tl_days > 0) {
                std::snprintf(tl_buf, sizeof(tl_buf), "%02d:%02d:%02d:%02d", tl_days, tl_hrs, tl_mins, tl_secs);
            } else {
                std::snprintf(tl_buf, sizeof(tl_buf), "%02d:%02d:%02d", tl_hrs, tl_mins, tl_secs);
            }
            s.time_left_str = tl_buf;
        } else {
            s.time_left_str = "--:--:--";
        }
    } else {
        s.free_space_str = "Unknown";
        s.time_left_str = "--:--:--";
    }

    return s;
}
