#pragma once

#include <memory>
#include <unordered_map>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// 各存储后端 SDK 的统一接口。
//
// ============================================================
// Timeout 与 buffer 生命周期契约（规范全文见
// docs/design/client_sdk_io_contract.md，即 docs/tasks/safe-timeout/01-contract.md 的副本）
// ============================================================
//
// 1) 有界性（所有实现必须满足）
//    实现必须在 I/O 预算耗尽后尽快返回，不得无界阻塞调用线程。
//    预算通过 Init 阶段的 SdkTimeoutConfig 下传，不在 Get/Put 签名中传递。
//    SdkWrapper 在每次 Get/Put 入口计算一次绝对 deadline，并经
//    SdkDeadline（thread_local + RAII）传播给线程池任务；实现方可在
//    任何深度读取 SdkDeadline::Get() / Expired() / RemainingMs()。
//
// 2) 准入（所有实现必须满足）
//    在发起任何一次会写 caller buffer 的 I/O 之前，实现必须检查 deadline。
//    已过期则直接返回超时错误，禁止发起该 I/O。
//    分批/逐 block 的实现必须在每一批/每一块前检查，而不是只在入口检查一次。
//    （SdkWrapper 的线程池任务启动前也做一次准入检查 —— 排队超时的任务
//      不允许再发起 I/O，见 sdk_wrapper.cc RunWithTimeoutParallel。）
//
// 3) Buffer 生命周期（分级，实现必须如实声明自己属于哪一级）
//    - hard：返回后 caller buffer 不再被任何执行者访问。
//      LocalFileSdk、Hf3fsSdk、TairMempoolSdk（默认 staging 路径）属于此级。
//    - soft：返回后仍可能有在飞 I/O 写 caller buffer。
//      MooncakeSdk 属于此级，因为 slices 直接指向 caller 内存且上游
//      transfer engine 不提供取消语义。
//      此级实现必须在超时路径输出可归因日志（后端、op、block 下标、buffer 地址）。
//
// 4) 调用方义务
//    在 soft 级后端参与的调用返回后立即复用 caller buffer，是已知的数据竞争。
//    本版本不提供延迟归还 / 隔离机制。该风险通过观测收敛（sdk_io_stats + 日志）。
//
// 履约矩阵（简表；各后端实现必须与其声明一致）：
//   后端            | 有界性       | 准入检查   | Buffer 级别
//   LocalFile / NFS | 逐 block     | 逐 block   | hard（abort 路径必须 sync GPU stream）
//   HF3FS           | abs_timeout  | 逐 block   | hard（数据先落我们自己的 shm；超时不 CopyIovs）
//   Mooncake        | 逐 key 前置  | 逐 key     | soft（上游无取消语义）
//   TairMempool     | PACE 9s/10s  | PACE 无队列准入 | hard（PACE 已有 cancel_and_drain）
//
// 明确否决的方案（勿"顺手改回去"）：SdkWrapper 层等待 in-flight I/O 完成；
// mooncake staging buffer；超时后 unregister 内存；Get/Put 增加 per-call deadline 参数。
class SdkInterface {
public:
    SdkInterface() {}
    virtual ~SdkInterface() = default;
    virtual ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                 const std::shared_ptr<StorageConfig> &storage_config) = 0;

    virtual SdkType Type() = 0;

    // 一个remote_uri和一个Blockbuffer对应一个block。
    // 契约：下标是 block 的唯一身份 —— remote_uris[i] ↔ local_buffers[i]。
    // 超时路径必须输出可归因日志（backend、op、deadline_ms、elapsed_ms、
    // 完成的 block 数/总数、在飞 block 下标与 caller buffer 地址）。
    virtual ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) = 0;
    // actual_remote_uris是实际存储的远端地址。
    // 保序契约：实现方必须保证 actual_remote_uris[i] 对应 remote_uris[i]（下标即身份）。
    // 分组处理后必须按 BlockGroup::indices 回填原位，禁止依赖 map 迭代序或
    // 用 assign(remote_uris) 丢弃 Alloc 的返回值（会拆掉 local alloc 管道）。
    virtual ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) = 0;

protected:
    virtual ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                                  std::vector<DataStorageUri> &alloc_uris) = 0;

    using GroupMap = std::unordered_map<std::string, BlockGroup>;
    // 按 path 分组，并记录每个元素在原始入参中的下标到 BlockGroup::indices。
    GroupMap SplitByPath(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
};

} // namespace kv_cache_manager
