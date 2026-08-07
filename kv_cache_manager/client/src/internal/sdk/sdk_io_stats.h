#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"

namespace kv_cache_manager {

// 进程内轻量的 SDK I/O 观测：超时 / 准入拒绝 / soft 级返回计数。
// 全部 std::atomic，无锁；client 侧当前没有 metrics 设施，刻意不引入 kmonitor。
// 供超时与准入拒绝路径的日志归因（01-contract.md §4.1）以及单测断言使用。
class SdkIoStats {
public:
    static SdkIoStats &Instance();

    // 一次 SDK 调用整体超时（wrapper 层 wait_until(deadline) 到期，尚未完成的组）。
    // elapsed_ms / done / total 由调用方写入日志（01-contract.md §4.1 字段要求）。
    void OnTimeout(SdkType type, bool is_get, int64_t elapsed_ms, size_t done, size_t total);
    // 线程池任务启动时发现已过 deadline，未发起任何 I/O 即拒绝。
    // 这是 00-context.md §6.1 排队超时缺陷的修复点；overdue_ms 用于日志。
    void OnAdmissionReject(SdkType type, bool is_get, int64_t overdue_ms);
    // soft 级后端（mooncake）超时返回时仍有在飞 I/O 写 caller buffer；供 W3 使用。
    void OnUnsafeReturn(SdkType type, bool is_get, size_t inflight_blocks);

    // 供测试与日志使用；格式稳定，可被单测断言。
    // 形如：timeout_count: local_file/get=1; admission_reject_count: ...; unsafe_return_count: ...
    std::string DebugString() const;
    // 仅供测试：清零所有计数。
    void Reset();

private:
    SdkIoStats() = default;

    // 按 (SdkType, op) 两维计数。SdkType 数值空间为 uint8（SDK_TYPE_MAX=255），256 上限足够且无锁。
    std::atomic<uint64_t> timeout_count_[256][2]{};
    std::atomic<uint64_t> admission_reject_count_[256][2]{};
    std::atomic<uint64_t> unsafe_return_count_[256][2]{};
};

// SdkType -> 可读字符串，用于日志归因（backend 字段）。
std::string SdkTypeToString(SdkType type);

} // namespace kv_cache_manager
