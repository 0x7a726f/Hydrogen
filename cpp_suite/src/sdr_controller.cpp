#include "sdr_controller.h"
#include "virtual_sdr.h"
#ifdef _WIN32
#include <windows.h>
#endif

SdrController::SdrController()
    : rx_buffer_(SdrConstants::SDR_CHUNK_BYTES),
      dispatch_buf_(SdrConstants::SDR_CHUNK_BYTES) {
}

SdrController::~SdrController() {
    {
        std::lock_guard<std::mutex> clk(cb_mutex_);
        sample_callbacks_.clear();
        freq_changed_cb_ = nullptr;
    }
    shutdown();
}

void SdrController::notify_freq_changed(const std::vector<uint64_t>& freq_events) {
    if (freq_events.empty()) return;
    FreqChangedCallback cb_copy;
    {
        std::lock_guard<std::mutex> clk(cb_mutex_);
        cb_copy = freq_changed_cb_;
    }
    if (cb_copy) {
        for (uint64_t actual : freq_events) {
            cb_copy(actual);
        }
    }
}

void SdrController::refresh_cached_metadata_locked(const std::string& status_override) {
    std::string name = "None";
    std::string status = "Disconnected";
    std::vector<float> gains;

    if (device_) {
        name = device_->get_name();
        status = status_override.empty() ? device_->get_status_detail() : status_override;
        gains = device_->get_valid_gains();
    }

    std::lock_guard<std::mutex> mlk(meta_mutex_);
    cached_device_name_ = std::move(name);
    cached_status_detail_ = std::move(status);
    cached_valid_gains_ = std::move(gains);
}

void SdrController::stop_workers_locked(std::vector<uint64_t>& freq_events) {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    {
        std::lock_guard<std::mutex> dlk(dispatch_mutex_);
        dispatch_stop_.store(true, std::memory_order_release);
    }
    dispatch_cv_.notify_all();
    if (dispatcher_thread_.joinable()) {
        dispatcher_thread_.join();
    }
    is_streaming_.store(false, std::memory_order_release);
    workers_active_.store(false, std::memory_order_release);
    process_pending_commands(freq_events);
}

bool SdrController::initialize(bool force_virtual, uint32_t device_index) {
    std::vector<uint64_t> freq_events;
    {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        stop_workers_locked(freq_events);
        if (device_) {
            device_->close();
            device_.reset();
        }

        bool hardware_ok = false;
        std::string last_hw_error;
        if (!force_virtual) {
            auto dev = std::make_unique<RtlSdrDevice>();
            if (dev->open(device_index)) {
                device_ = std::move(dev);
                is_simulated_.store(false);
                hardware_ok = true;
            } else {
                last_hw_error = dev->get_status_detail();
            }
        }

        std::string status_override;
        if (!hardware_ok) {
            auto vdev = std::make_unique<VirtualSdrDevice>();
            vdev->open(device_index);
            device_ = std::move(vdev);
            is_simulated_.store(true);
            if (!force_virtual && !last_hw_error.empty()) {
                status_override = "Virtual SDR Active (HW fallback: " + last_hw_error + ")";
            }
        }

        // Apply saved controller settings to BOTH hardware and virtual devices
        device_->set_sample_rate(current_rate_.load());
        device_->set_frequency(current_freq_.load());
        device_->set_gain(current_gain_.load(), auto_gain_.load());
        device_->set_bias_tee(bias_tee_.load());
        device_->set_ppm(ppm_.load());

        current_freq_.store(device_->get_frequency());
        current_rate_.store(device_->get_sample_rate());
        current_gain_.store(device_->get_gain());
        auto_gain_.store(device_->is_auto_gain());
        bias_tee_.store(device_->get_bias_tee());
        ppm_.store(device_->get_ppm());

        refresh_cached_metadata_locked(status_override);
    }

    notify_freq_changed(freq_events);
    return true;
}

void SdrController::shutdown() {
    std::vector<uint64_t> freq_events;
    {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        stop_workers_locked(freq_events);
        if (device_) {
            device_->close();
            device_.reset();
        }
        refresh_cached_metadata_locked();
    }
    notify_freq_changed(freq_events);
}

bool SdrController::start_streaming() {
    std::vector<uint64_t> freq_events;
    {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        if (is_streaming_.load()) return true;
        stop_workers_locked(freq_events);

        if (!device_) {
            auto dev = std::make_unique<RtlSdrDevice>();
            std::string status_override;
            if (dev->open(0)) {
                device_ = std::move(dev);
                is_simulated_.store(false);
            } else {
                const std::string last_hw_error = dev->get_status_detail();
                auto vdev = std::make_unique<VirtualSdrDevice>();
                vdev->open(0);
                device_ = std::move(vdev);
                is_simulated_.store(true);
                if (!last_hw_error.empty()) {
                    status_override = "Virtual SDR Active (HW fallback: " + last_hw_error + ")";
                }
            }
            device_->set_sample_rate(current_rate_.load());
            device_->set_frequency(current_freq_.load());
            device_->set_gain(current_gain_.load(), auto_gain_.load());
            device_->set_bias_tee(bias_tee_.load());
            device_->set_ppm(ppm_.load());

            current_freq_.store(device_->get_frequency());
            current_rate_.store(device_->get_sample_rate());
            current_gain_.store(device_->get_gain());
            auto_gain_.store(device_->is_auto_gain());
            bias_tee_.store(device_->get_bias_tee());
            ppm_.store(device_->get_ppm());

            refresh_cached_metadata_locked(status_override);
        }

        ring_buffer_.reset();
        stop_requested_.store(false, std::memory_order_release);
        dispatch_stop_.store(false, std::memory_order_release);
        workers_active_.store(true, std::memory_order_release);
        is_streaming_.store(true, std::memory_order_release);
        dispatcher_thread_ = std::thread(&SdrController::dispatch_worker, this);
        worker_thread_ = std::thread(&SdrController::stream_worker, this);
    }

    notify_freq_changed(freq_events);
    return true;
}

void SdrController::stop_streaming() {
    std::vector<uint64_t> freq_events;
    {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        if (!is_streaming_.load() && !worker_thread_.joinable() && !dispatcher_thread_.joinable()) return;
        stop_workers_locked(freq_events);
    }
    notify_freq_changed(freq_events);
}

std::future<bool> SdrController::enqueue_command(SdrCommand cmd) {
    auto prom = std::make_shared<std::promise<bool>>();
    auto fut = prom->get_future();
    cmd.promise = std::move(prom);

    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        cmd_queue_.push_back(std::move(cmd));
    }

    std::vector<uint64_t> freq_events;
    if (!is_streaming_.load(std::memory_order_acquire) &&
        !workers_active_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        if (!is_streaming_.load(std::memory_order_acquire) &&
            !workers_active_.load(std::memory_order_acquire)) {
            process_pending_commands(freq_events);
        }
    }
    notify_freq_changed(freq_events);
    return fut;
}

std::future<bool> SdrController::request_frequency(uint64_t freq_hz) {
    SdrCommand cmd;
    cmd.type = SdrCommandType::SetFrequency;
    cmd.uint_param = freq_hz;
    return enqueue_command(std::move(cmd));
}

std::future<bool> SdrController::request_sample_rate(uint32_t rate_hz) {
    SdrCommand cmd;
    cmd.type = SdrCommandType::SetSampleRate;
    cmd.uint_param = rate_hz;
    return enqueue_command(std::move(cmd));
}

std::future<bool> SdrController::request_gain(float gain_db, bool auto_gain) {
    SdrCommand cmd;
    cmd.type = SdrCommandType::SetGain;
    cmd.float_param = gain_db;
    cmd.bool_param = auto_gain;
    return enqueue_command(std::move(cmd));
}

std::future<bool> SdrController::request_bias_tee(bool enabled) {
    SdrCommand cmd;
    cmd.type = SdrCommandType::SetBiasTee;
    cmd.bool_param = enabled;
    return enqueue_command(std::move(cmd));
}

std::future<bool> SdrController::request_ppm(int ppm) {
    SdrCommand cmd;
    cmd.type = SdrCommandType::SetPpm;
    cmd.int_param = ppm;
    return enqueue_command(std::move(cmd));
}

std::string SdrController::get_device_name() const {
    std::lock_guard<std::mutex> mlk(meta_mutex_);
    return cached_device_name_;
}

std::string SdrController::get_status_detail() const {
    std::lock_guard<std::mutex> mlk(meta_mutex_);
    return cached_status_detail_;
}

std::vector<float> SdrController::get_valid_gains() {
    std::lock_guard<std::mutex> mlk(meta_mutex_);
    return cached_valid_gains_;
}

void SdrController::add_sample_callback(SampleCallback cb) {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    sample_callbacks_.push_back(std::move(cb));
}

void SdrController::clear_sample_callbacks() {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    sample_callbacks_.clear();
}

void SdrController::set_freq_changed_callback(FreqChangedCallback cb) {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    freq_changed_cb_ = std::move(cb);
}

std::vector<std::string> SdrController::scan_all_devices() {
    return RtlSdrDevice::scan_devices();
}

std::vector<std::string> SdrController::check_blocking_apps() {
    return RtlSdrDevice::detect_blocking_processes();
}

void SdrController::process_pending_commands(std::vector<uint64_t>& freq_events) {
    while (true) {
        std::vector<SdrCommand> local_queue;
        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            if (cmd_queue_.empty()) break;
            local_queue = std::move(cmd_queue_);
            cmd_queue_.clear();
        }

        // Coalesce idempotent commands to eliminate tuning lag, driver thrashing, and freezes
        int last_freq_idx = -1;
        int last_rate_idx = -1;
        int last_gain_idx = -1;
        int last_bias_idx = -1;
        int last_ppm_idx  = -1;

        for (int i = 0; i < static_cast<int>(local_queue.size()); ++i) {
            switch (local_queue[static_cast<size_t>(i)].type) {
                case SdrCommandType::SetFrequency:  last_freq_idx = i; break;
                case SdrCommandType::SetSampleRate: last_rate_idx = i; break;
                case SdrCommandType::SetGain:       last_gain_idx = i; break;
                case SdrCommandType::SetBiasTee:    last_bias_idx = i; break;
                case SdrCommandType::SetPpm:        last_ppm_idx  = i; break;
                default: break;
            }
        }

        for (int i = 0; i < static_cast<int>(local_queue.size()); ++i) {
            auto& cmd = local_queue[static_cast<size_t>(i)];

            bool is_superseded = false;
            if (cmd.type == SdrCommandType::SetFrequency && i < last_freq_idx) is_superseded = true;
            else if (cmd.type == SdrCommandType::SetSampleRate && i < last_rate_idx) is_superseded = true;
            else if (cmd.type == SdrCommandType::SetGain && i < last_gain_idx) is_superseded = true;
            else if (cmd.type == SdrCommandType::SetBiasTee && i < last_bias_idx) is_superseded = true;
            else if (cmd.type == SdrCommandType::SetPpm && i < last_ppm_idx) is_superseded = true;

            if (is_superseded) {
                if (cmd.promise) {
                    cmd.promise->set_value(true);
                }
                continue;
            }

            bool success = false;
            if (device_) {
                switch (cmd.type) {
                    case SdrCommandType::SetFrequency: {
                        success = device_->set_frequency(cmd.uint_param);
                        const uint64_t actual = device_->get_frequency();
                        current_freq_.store(actual);
                        if (!is_simulated_.load()) {
                            std::string detail = device_->get_status_detail();
                            std::lock_guard<std::mutex> mlk(meta_mutex_);
                            cached_status_detail_ = std::move(detail);
                        }
                        freq_events.push_back(actual);
                        break;
                    }
                    case SdrCommandType::SetSampleRate: {
                        success = device_->set_sample_rate(static_cast<uint32_t>(cmd.uint_param));
                        current_rate_.store(device_->get_sample_rate());
                        break;
                    }
                    case SdrCommandType::SetGain: {
                        success = device_->set_gain(cmd.float_param, cmd.bool_param);
                        current_gain_.store(device_->get_gain());
                        auto_gain_.store(device_->is_auto_gain());
                        break;
                    }
                    case SdrCommandType::SetBiasTee: {
                        success = device_->set_bias_tee(cmd.bool_param);
                        bias_tee_.store(device_->get_bias_tee());
                        break;
                    }
                    case SdrCommandType::SetPpm: {
                        success = device_->set_ppm(cmd.int_param);
                        ppm_.store(device_->get_ppm());
                        break;
                    }
                    case SdrCommandType::ResetBuffer: {
                        device_->reset_buffer();
                        success = true;
                        break;
                    }
                    case SdrCommandType::Stop: {
                        success = true;
                        break;
                    }
                }
            }

            if (cmd.promise) {
                cmd.promise->set_value(success);
            }
        }
    }
}

void SdrController::stream_worker() {
#ifdef _WIN32
    // Elevate worker thread priority to TIME_CRITICAL to eliminate OS scheduling jitter and USB underruns
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
        // Fallback to HIGHEST if TIME_CRITICAL is restricted by OS policy
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
#endif

    std::vector<uint64_t> freq_events;
    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Safe point to execute queued hardware control commands (never holds lifecycle_mutex_)
        freq_events.clear();
        process_pending_commands(freq_events);
        notify_freq_changed(freq_events);

        if (!is_streaming_.load(std::memory_order_acquire) || !device_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        int bytes_read = 0;
        int ret = device_->read_samples(rx_buffer_.data(), static_cast<int>(rx_buffer_.size()), &bytes_read);
        if (ret == 0 && bytes_read > 1) {
            const size_t aligned_bytes = static_cast<size_t>(bytes_read) & ~size_t(1);
            // Producer pushes solely to lock-free SPSC ring buffer; dispatch_worker drains and runs callbacks
            ring_buffer_.write_lossy(rx_buffer_.data(), aligned_bytes);
            {
                std::lock_guard<std::mutex> dlk(dispatch_mutex_);
            }
            dispatch_cv_.notify_one();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

void SdrController::dispatch_worker() {
    std::vector<SampleCallback> local_cbs;
    while (true) {
        size_t avail = ring_buffer_.available_read();
        if (avail == 0) {
            std::unique_lock<std::mutex> dlk(dispatch_mutex_);
            if (dispatch_stop_.load(std::memory_order_acquire) && ring_buffer_.available_read() == 0) {
                break;
            }
            dispatch_cv_.wait_for(dlk, std::chrono::milliseconds(5), [&]() {
                return dispatch_stop_.load(std::memory_order_acquire) || ring_buffer_.available_read() > 0;
            });
            if (dispatch_stop_.load(std::memory_order_acquire) && ring_buffer_.available_read() == 0) {
                break;
            }
            continue;
        }

        size_t got = ring_buffer_.read(dispatch_buf_.data(), dispatch_buf_.size());
        if (got > 0) {
            {
                std::lock_guard<std::mutex> clk(cb_mutex_);
                local_cbs = sample_callbacks_;
            }
            for (auto& cb : local_cbs) {
                if (cb) cb(dispatch_buf_.data(), got);
            }
        }
    }
}
