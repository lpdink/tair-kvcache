#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {
class StorageConfig;
class LockFreeThreadPool;
class SdkFactory;
class SdkInterface;

class SdkWrapper {
public:
    SdkWrapper();
    ~SdkWrapper();

public:
    ClientErrorCode Init(const std::unique_ptr<ClientConfig> &client_config, const InitParams &init_params);

    ClientErrorCode
    Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers, int64_t deadline_us);
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris,
                        int64_t deadline_us);

private:
    enum class OpType : uint8_t {
        GET = 0,
        PUT = 1,
    };

    // 按 SDK 分组的 URI 和 buffer 组
    struct SdkGroup {
        std::shared_ptr<SdkInterface> sdk;
        std::vector<size_t> indices; // 原始索引，用于结果重排
        std::vector<DataStorageUri> uris;
        BlockBuffers buffers;
    };

    // 单个线程池任务及其归因元信息（日志/统计用，见 01-contract.md §4.1 字段要求）
    struct TimedTask {
        SdkType sdk_type{SdkType::LOCAL_FILE}; // 统计与日志归因用 backend 类型
        size_t group_index{0};                 // 分组序号（1-based）
        size_t group_count{0};                 // 总分组数
        size_t block_count{0};                 // 本组涉及的 block 数
        std::function<ClientErrorCode()> fn;
    };

    ClientErrorCode Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers);
    std::shared_ptr<SdkInterface> GetSdk(const DataStorageUri &remote_uri);

    // 按 URI hostname 分组，每组关联对应 SDK；任一 hostname 无对应 SDK 时返回错误
    ClientErrorCode GroupBySdk(const std::vector<DataStorageUri> &remote_uris,
                               const BlockBuffers &local_buffers,
                               std::vector<SdkGroup> &groups);

    std::string getOpTypeString(OpType op_type) const;
    // deadline 由调用方（TransferClient）逐请求传入（int64_t deadline_us，绝对
    // steady_clock 微秒，0=无外部 deadline → 回退到 wrapper 级 timeout_config）。
    // 统一向下传播：
    // 1) 线程池任务启动时的准入检查（核心修复，见 docs/design/client_sdk_io_contract.md §2）；
    // 2) 作为 Get/Put 参数传给 SDK 内部做逐 block/逐 key 检查。
    // 超时/失败即刻返回，绝不等待 in-flight 任务（见契约 §2.1 否决项）。
    ClientErrorCode RunWithTimeoutParallel(OpType op_type,
                                           std::vector<TimedTask> &&tasks,
                                           std::chrono::steady_clock::time_point deadline,
                                           int timeout_ms) const;
    ClientErrorCode UpdateMooncakeSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            RegistSpan *span,
                                            const std::string &self_location_spec_name);

private:
    SdkFactory *sdk_factory_;
    std::shared_ptr<SdkWrapperConfig> wrapper_config_;
    std::vector<std::shared_ptr<StorageConfig>> storage_configs_;
    std::unique_ptr<LockFreeThreadPool> wait_task_thread_pool_;
    // storage unique name -> storage_sdk
    std::map<std::string, std::shared_ptr<SdkInterface>> sdk_map_;
};

} // namespace kv_cache_manager