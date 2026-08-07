#pragma once

#include <chrono>
#include <optional>

namespace kv_cache_manager {

// 在当前线程内传播一次 SDK 调用的绝对 deadline。
// SdkWrapper 在线程池任务里用 Scope 设置；各 SDK 实现读取它做逐 block/逐 key 准入检查。
// 未设置时（例如单测直接调用 SDK）所有查询返回"无 deadline"，行为与旧版一致。
//
// 为什么用 thread_local 而不是改 Get/Put 签名：3FS 需要 deadline 深入到
// Hf3fsUsrbioClient::WaitIos（4 层调用链），逐层加参数会污染大量签名；
// 且线上主力后端 PACE 的 timeout 本身是进程级配置，per-call 参数没有对应物。
// 这是 deadline 传播的标准做法（同 gRPC context）。
class SdkDeadline {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    class Scope {
    public:
        explicit Scope(TimePoint deadline);
        ~Scope();
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

    private:
        std::optional<TimePoint> prev_;
    };

    static std::optional<TimePoint> Get();
    // 未设置 deadline 时返回 false（不阻断）
    static bool Expired();
    // 未设置返回 -1；已过期返回 0
    static int64_t RemainingMs();
};

} // namespace kv_cache_manager
