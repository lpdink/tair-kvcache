#include "kv_cache_manager/client/src/internal/sdk/sdk_io_stats.h"

#include <sstream>

namespace kv_cache_manager {

namespace {
constexpr size_t kMaxSdkType = 256;

size_t OpIndex(bool is_get) { return is_get ? 0 : 1; }
size_t OpIndexSafe(SdkType type) {
    size_t idx = static_cast<size_t>(type);
    return idx < kMaxSdkType ? idx : kMaxSdkType - 1;
}
} // namespace

SdkIoStats &SdkIoStats::Instance() {
    static SdkIoStats instance;
    return instance;
}

void SdkIoStats::OnTimeout(SdkType type, bool is_get, int64_t /*elapsed_ms*/, size_t /*done*/, size_t /*total*/) {
    timeout_count_[OpIndexSafe(type)][OpIndex(is_get)].fetch_add(1, std::memory_order_relaxed);
}

void SdkIoStats::OnAdmissionReject(SdkType type, bool is_get, int64_t /*overdue_ms*/) {
    admission_reject_count_[OpIndexSafe(type)][OpIndex(is_get)].fetch_add(1, std::memory_order_relaxed);
}

void SdkIoStats::OnUnsafeReturn(SdkType type, bool is_get, size_t /*inflight_blocks*/) {
    unsafe_return_count_[OpIndexSafe(type)][OpIndex(is_get)].fetch_add(1, std::memory_order_relaxed);
}

namespace {
void AppendCounts(std::ostringstream &oss, const char *name, const std::atomic<uint64_t> (&counts)[kMaxSdkType][2]) {
    oss << name << ":";
    bool first = true;
    for (size_t t = 0; t < kMaxSdkType; ++t) {
        for (size_t op = 0; op < 2; ++op) {
            uint64_t v = counts[t][op].load(std::memory_order_relaxed);
            if (v == 0) {
                continue;
            }
            if (!first) {
                oss << ",";
            }
            oss << " " << SdkTypeToString(static_cast<SdkType>(t)) << "/" << (op == 0 ? "get" : "put") << "=" << v;
            first = false;
        }
    }
}
} // namespace

std::string SdkIoStats::DebugString() const {
    std::ostringstream oss;
    AppendCounts(oss, "timeout_count", timeout_count_);
    oss << "; ";
    AppendCounts(oss, "admission_reject_count", admission_reject_count_);
    oss << "; ";
    AppendCounts(oss, "unsafe_return_count", unsafe_return_count_);
    return oss.str();
}

void SdkIoStats::Reset() {
    for (size_t t = 0; t < kMaxSdkType; ++t) {
        for (size_t op = 0; op < 2; ++op) {
            timeout_count_[t][op].store(0, std::memory_order_relaxed);
            admission_reject_count_[t][op].store(0, std::memory_order_relaxed);
            unsafe_return_count_[t][op].store(0, std::memory_order_relaxed);
        }
    }
}

std::string SdkTypeToString(SdkType type) {
    switch (type) {
    case SdkType::HF3FS:
        return "hf3fs";
    case SdkType::MOONCAKE:
        return "mooncake";
    case SdkType::TAIR_MEMPOOL:
        return "tair_mempool";
    case SdkType::LOCAL_FILE:
        return "local_file";
    default:
        return "unknown(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

} // namespace kv_cache_manager
