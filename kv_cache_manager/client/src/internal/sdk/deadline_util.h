#pragma once

#include <chrono>

namespace kv_cache_manager {

// deadline_us 语义（契约见 docs/design/client_sdk_io_contract.md §2）：
//   - 绝对时间点，单位微秒，基于 std::chrono::steady_clock（内核 CLOCK_MONOTONIC）。
//   - 0 表示「调用方不施加 deadline」：SDK 内部不做准入/等待限制（SdkWrapper 层
//     仍会用 wrapper 级 timeout_config 兜底，见 sdk_wrapper.cc）。
//   - 时钟一致性（已实测）：Python time.monotonic_ns()//1000 与 C++ steady_clock
//     是同一个内核时钟，跨语言直接数值比较，无需校正。
inline int64_t SteadyClockUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// 准入检查：deadline_us == 0（无 deadline）时不阻断；已过期返回 true。
inline bool DeadlineExpired(int64_t deadline_us) { return deadline_us > 0 && SteadyClockUs() >= deadline_us; }

// 剩余预算：无 deadline 返回 -1；已过期返回 0；否则返回剩余毫秒（向下取整）。
inline int64_t DeadlineRemainingMs(int64_t deadline_us) {
    if (deadline_us <= 0) {
        return -1;
    }
    int64_t now_us = SteadyClockUs();
    if (now_us >= deadline_us) {
        return 0;
    }
    return (deadline_us - now_us) / 1000;
}

} // namespace kv_cache_manager
