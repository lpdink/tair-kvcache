#include "kv_cache_manager/client/src/internal/sdk/sdk_deadline.h"

namespace kv_cache_manager {

namespace {
// 线程本地 deadline。仅当前线程读写，无锁。
thread_local std::optional<SdkDeadline::TimePoint> t_deadline;
} // namespace

SdkDeadline::Scope::Scope(TimePoint deadline) : prev_(t_deadline) { t_deadline = deadline; }

SdkDeadline::Scope::~Scope() { t_deadline = prev_; }

std::optional<SdkDeadline::TimePoint> SdkDeadline::Get() { return t_deadline; }

bool SdkDeadline::Expired() {
    if (!t_deadline.has_value()) {
        return false;
    }
    return std::chrono::steady_clock::now() >= *t_deadline;
}

int64_t SdkDeadline::RemainingMs() {
    if (!t_deadline.has_value()) {
        return -1;
    }
    auto remaining = *t_deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
        return 0;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
}

} // namespace kv_cache_manager
