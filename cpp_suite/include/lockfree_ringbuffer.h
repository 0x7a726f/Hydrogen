#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include "simd_dsp.h"

// Single-Producer Single-Consumer (SPSC) Lock-Free Circular Ring Buffer
// Engineered with 64-byte cache-line separation to eliminate false sharing.
template <typename T>
class LockFreeSpscRingBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeSpscRingBuffer requires a trivially copyable element type");

public:
    explicit LockFreeSpscRingBuffer(size_t capacity_power_of_2 = 8388608) { // Default 8 MB (8,388,608 elements)
        // Ensure capacity is a power of 2 with strict byte-size overflow protection
        const size_t max_pow2 = static_cast<size_t>(1) << (sizeof(size_t) * 8 - 2);
        const size_t max_by_type = std::numeric_limits<size_t>::max() / sizeof(T);
        const size_t max_cap = std::min(max_pow2, max_by_type);

        size_t cap = 1024;
        if (cap > max_cap) {
            cap = 1;
            while (cap <= (max_cap >> 1)) {
                cap <<= 1;
            }
        }
        while (cap < capacity_power_of_2 && cap <= (max_cap >> 1)) {
            cap <<= 1;
        }
        capacity_ = cap;
        mask_ = capacity_ - 1;

        // Allocate 64-byte aligned buffer (guaranteed capacity_ * sizeof(T) does not overflow)
        buffer_ = static_cast<T*>(SimdDsp::aligned_alloc_64(capacity_ * sizeof(T)));
        if (buffer_) {
            is_custom_aligned_ = true;
        } else {
            // Fallback to standard allocation
            buffer_ = new T[capacity_];
            is_custom_aligned_ = false;
        }
    }

    ~LockFreeSpscRingBuffer() {
        if (buffer_) {
            if (is_custom_aligned_) {
                SimdDsp::aligned_free_64(buffer_);
            } else {
                delete[] buffer_;
            }
            buffer_ = nullptr;
        }
    }

    // Non-copyable
    LockFreeSpscRingBuffer(const LockFreeSpscRingBuffer&) = delete;
    LockFreeSpscRingBuffer& operator=(const LockFreeSpscRingBuffer&) = delete;

    // Push up to 'count' elements into buffer. Returns actual elements written without dropping.
    size_t write(const T* src, size_t count) {
        if (!src || count == 0) return 0;

        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        size_t free_slots = capacity_ - (current_write - read_idx_cache_);
        if (free_slots < count) {
            read_idx_cache_ = read_idx_.load(std::memory_order_acquire);
            free_slots = capacity_ - (current_write - read_idx_cache_);
        }
        const size_t to_write = std::min(count, free_slots);

        if (to_write == 0) {
            return 0;
        }

        // Two-part circular write
        const size_t write_pos = current_write & mask_;
        const size_t first_chunk = std::min(to_write, capacity_ - write_pos);
        const size_t second_chunk = to_write - first_chunk;

        std::memcpy(&buffer_[write_pos], src, first_chunk * sizeof(T));
        if (second_chunk > 0) {
            std::memcpy(&buffer_[0], src + first_chunk, second_chunk * sizeof(T));
        }

        write_idx_.store(current_write + to_write, std::memory_order_release);
        return to_write;
    }

    // Push all 'count' elements atomically (all-or-nothing). Returns true on full write, false if insufficient space.
    bool write_exact(const T* src, size_t count) {
        if (count == 0) return true;
        if (!src || !buffer_ || count > capacity_) return false;

        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        size_t free_slots = capacity_ - (current_write - read_idx_cache_);
        if (free_slots < count) {
            read_idx_cache_ = read_idx_.load(std::memory_order_acquire);
            free_slots = capacity_ - (current_write - read_idx_cache_);
        }
        if (free_slots < count) {
            return false;
        }

        const size_t write_pos = current_write & mask_;
        const size_t first_chunk = std::min(count, capacity_ - write_pos);
        const size_t second_chunk = count - first_chunk;

        std::memcpy(&buffer_[write_pos], src, first_chunk * sizeof(T));
        if (second_chunk > 0) {
            std::memcpy(&buffer_[0], src + first_chunk, second_chunk * sizeof(T));
        }

        write_idx_.store(current_write + count, std::memory_order_release);
        return true;
    }

    // Real-time hardware write: writes what fits, and logs unwritten remainder as dropped
    size_t write_lossy(const T* src, size_t count) {
        size_t written = write(src, count);
        if (written < count) {
            dropped_count_.fetch_add(count - written, std::memory_order_relaxed);
        }
        return written;
    }

    // Read up to 'count' elements into dst. Returns actual elements read.
    size_t read(T* dst, size_t count) {
        if (!dst || count == 0) return 0;

        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        size_t available = write_idx_cache_ - current_read;
        if (available < count) {
            write_idx_cache_ = write_idx_.load(std::memory_order_acquire);
            available = write_idx_cache_ - current_read;
        }
        const size_t to_read = std::min(count, available);

        if (to_read == 0) return 0;

        // Two-part circular read
        const size_t read_pos = current_read & mask_;
        const size_t first_chunk = std::min(to_read, capacity_ - read_pos);
        const size_t second_chunk = to_read - first_chunk;

        std::memcpy(dst, &buffer_[read_pos], first_chunk * sizeof(T));
        if (second_chunk > 0) {
            std::memcpy(dst + first_chunk, &buffer_[0], second_chunk * sizeof(T));
        }

        read_idx_.store(current_read + to_read, std::memory_order_release);
        return to_read;
    }

    // Peek without consuming
    size_t peek(T* dst, size_t count) const {
        if (!dst || count == 0) return 0;

        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        const size_t available = write_idx_.load(std::memory_order_acquire) - current_read;
        const size_t to_read = std::min(count, available);

        if (to_read == 0) return 0;

        const size_t read_pos = current_read & mask_;
        const size_t first_chunk = std::min(to_read, capacity_ - read_pos);
        const size_t second_chunk = to_read - first_chunk;

        std::memcpy(dst, &buffer_[read_pos], first_chunk * sizeof(T));
        if (second_chunk > 0) {
            std::memcpy(dst + first_chunk, &buffer_[0], second_chunk * sizeof(T));
        }

        return to_read;
    }

    // Skip / discard count elements
    void discard(size_t count) {
        if (count == 0) return;
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        size_t available = write_idx_cache_ - current_read;
        if (available < count) {
            write_idx_cache_ = write_idx_.load(std::memory_order_acquire);
            available = write_idx_cache_ - current_read;
        }
        const size_t to_discard = std::min(count, available);
        read_idx_.store(current_read + to_discard, std::memory_order_release);
    }

    size_t available_read() const {
        const size_t current_read = read_idx_.load(std::memory_order_acquire);
        const size_t current_write = write_idx_.load(std::memory_order_acquire);
        const size_t diff = current_write - current_read;
        return (diff <= capacity_) ? diff : capacity_;
    }

    size_t available_write() const {
        const size_t avail = available_read();
        return (avail <= capacity_) ? (capacity_ - avail) : 0;
    }

    size_t capacity() const { return capacity_; }

    float fill_percentage() const {
        const size_t avail = available_read();
        return (capacity_ > 0) ? (static_cast<float>(avail) / static_cast<float>(capacity_)) * 100.0f : 0.0f;
    }

    uint64_t dropped_count() const {
        return dropped_count_.load(std::memory_order_relaxed);
    }

    void reset() {
        read_idx_cache_ = 0;
        write_idx_cache_ = 0;
        read_idx_.store(0, std::memory_order_release);
        write_idx_.store(0, std::memory_order_release);
        dropped_count_.store(0, std::memory_order_release);
    }

private:
    size_t capacity_ = 0;
    size_t mask_ = 0;
    T* buffer_ = nullptr;
    bool is_custom_aligned_ = false;

    // Cache-line separation (64 bytes) to prevent false sharing between producer & consumer
    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) size_t read_idx_cache_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};
    alignas(64) size_t write_idx_cache_{0};
    alignas(64) std::atomic<uint64_t> dropped_count_{0};
};
