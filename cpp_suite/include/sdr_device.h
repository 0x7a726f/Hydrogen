#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class ISdrDevice {
public:
    virtual ~ISdrDevice() = default;

    virtual bool open(uint32_t index) = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
    virtual bool is_simulated() const = 0;

    virtual std::string get_name() const = 0;
    virtual std::string get_status_detail() const = 0;
    virtual std::vector<float> get_valid_gains() = 0;

    virtual bool set_sample_rate(uint32_t rate_hz) = 0;
    virtual uint32_t get_sample_rate() const = 0;

    virtual bool set_frequency(uint64_t freq_hz) = 0;
    virtual uint64_t get_frequency() const = 0;

    virtual bool set_gain(float gain_db, bool auto_gain) = 0;
    virtual float get_gain() const = 0;
    virtual bool is_auto_gain() const = 0;

    virtual bool set_bias_tee(bool enabled) = 0;
    virtual bool get_bias_tee() const = 0;

    virtual bool set_ppm(int ppm) = 0;
    virtual int get_ppm() const = 0;

    virtual int read_samples(uint8_t* buffer, int length, int* bytes_read) = 0;
    virtual void reset_buffer() = 0;
};

class RtlSdrDevice : public ISdrDevice {
public:
    RtlSdrDevice();
    ~RtlSdrDevice() override;

    RtlSdrDevice(const RtlSdrDevice&) = delete;
    RtlSdrDevice& operator=(const RtlSdrDevice&) = delete;
    RtlSdrDevice(RtlSdrDevice&&) = delete;
    RtlSdrDevice& operator=(RtlSdrDevice&&) = delete;

    bool open(uint32_t index) override;
    void close() override;
    bool is_open() const override { return dev_ != nullptr; }
    bool is_simulated() const override { return false; }

    std::string get_name() const override;
    std::string get_status_detail() const override;
    std::vector<float> get_valid_gains() override;

    bool set_sample_rate(uint32_t rate_hz) override;
    uint32_t get_sample_rate() const override { return sample_rate_hz_.load(); }

    bool set_frequency(uint64_t freq_hz) override;
    uint64_t get_frequency() const override { return center_freq_hz_.load(); }

    bool set_gain(float gain_db, bool auto_gain) override;
    float get_gain() const override { return gain_db_.load(); }
    bool is_auto_gain() const override { return auto_gain_.load(); }

    bool set_bias_tee(bool enabled) override;
    bool get_bias_tee() const override { return bias_tee_.load(); }

    bool set_ppm(int ppm) override;
    int get_ppm() const override { return ppm_.load(); }

    int read_samples(uint8_t* buffer, int length, int* bytes_read) override;
    void reset_buffer() override;

    static std::vector<std::string> scan_devices();
    static std::vector<std::string> detect_blocking_processes();

private:
    bool load_dll();
    void unload_dll();
    void set_status_detail(std::string detail);
    void cache_tuner_gains();

    void* dll_handle_ = nullptr;
    void* dev_ = nullptr;
    mutable std::mutex info_mutex_;
    std::string device_name_ = "RTL-SDR";
    std::string status_detail_;
    std::vector<float> cached_valid_gains_;

    std::atomic<uint32_t> sample_rate_hz_{2400000};
    std::atomic<uint64_t> center_freq_hz_{1420500000};
    std::atomic<float> gain_db_{25.4f};
    std::atomic<bool> auto_gain_{false};
    std::atomic<bool> bias_tee_{false};
    std::atomic<int> ppm_{0};

    // Function pointers for dynamic loading of rtlsdr.dll
    typedef uint32_t (*pfn_rtlsdr_get_device_count)(void);
    typedef const char* (*pfn_rtlsdr_get_device_name)(uint32_t index);
    typedef int (*pfn_rtlsdr_get_device_usb_strings)(uint32_t index, char *manufact, char *product, char *serial);
    typedef int (*pfn_rtlsdr_open)(void **dev, uint32_t index);
    typedef int (*pfn_rtlsdr_close)(void *dev);
    typedef int (*pfn_rtlsdr_set_center_freq)(void *dev, uint32_t freq);
    typedef uint32_t (*pfn_rtlsdr_get_center_freq)(void *dev);
    typedef int (*pfn_rtlsdr_set_sample_rate)(void *dev, uint32_t rate);
    typedef uint32_t (*pfn_rtlsdr_get_sample_rate)(void *dev);
    typedef int (*pfn_rtlsdr_set_tuner_gain_mode)(void *dev, int manual);
    typedef int (*pfn_rtlsdr_set_tuner_gain)(void *dev, int gain);
    typedef int (*pfn_rtlsdr_get_tuner_gains)(void *dev, int *gains);
    typedef int (*pfn_rtlsdr_set_freq_correction)(void *dev, int ppm);
    typedef int (*pfn_rtlsdr_reset_buffer)(void *dev);
    typedef int (*pfn_rtlsdr_read_sync)(void *dev, void *buf, int len, int *n_read);
    typedef int (*pfn_rtlsdr_set_bias_tee)(void *dev, int on);

    pfn_rtlsdr_get_device_count fn_get_device_count = nullptr;
    pfn_rtlsdr_get_device_name fn_get_device_name = nullptr;
    pfn_rtlsdr_get_device_usb_strings fn_get_device_usb_strings = nullptr;
    pfn_rtlsdr_open fn_open = nullptr;
    pfn_rtlsdr_close fn_close = nullptr;
    pfn_rtlsdr_set_center_freq fn_set_center_freq = nullptr;
    pfn_rtlsdr_get_center_freq fn_get_center_freq = nullptr;
    pfn_rtlsdr_set_sample_rate fn_set_sample_rate = nullptr;
    pfn_rtlsdr_get_sample_rate fn_get_sample_rate = nullptr;
    pfn_rtlsdr_set_tuner_gain_mode fn_set_tuner_gain_mode = nullptr;
    pfn_rtlsdr_set_tuner_gain fn_set_tuner_gain = nullptr;
    pfn_rtlsdr_get_tuner_gains fn_get_tuner_gains = nullptr;
    pfn_rtlsdr_set_freq_correction fn_set_freq_correction = nullptr;
    pfn_rtlsdr_reset_buffer fn_reset_buffer = nullptr;
    pfn_rtlsdr_read_sync fn_read_sync = nullptr;
    pfn_rtlsdr_set_bias_tee fn_set_bias_tee = nullptr;
};
