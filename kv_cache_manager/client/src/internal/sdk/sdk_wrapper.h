#pragma once

#include <chrono>
#include <functional>
#include <future>
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

    // 生命周期契约：Get/Put 内部会把 I/O 派发到线程池执行，但调用一旦返回（无论成功、失败
    // 还是 ER_SDK_TIMEOUT 超时），实现保证不再有任何线程访问 local_buffers 中 iov.base 指向的
    // 调用方内存，调用方可以立即释放或复用 buffer。
    // 超时代价：已经进入 SDK 的 I/O 无法被中途取消，超时/失败返回前会先等待这些在途任务执行
    // 完成，因此调用方被额外阻塞的时间上界 = 在途 SDK I/O 自身的完成时间（LocalFile 为对应
    // 文件读写系统调用；HF3FS/Mooncake 为其内部传输完成/超时机制），常规超时场景下即为剩余
    // I/O 时间，不会被额外放大。极端情况（存储后端僵死、I/O 永不返回）下调用将一直阻塞等待，
    // 期间按固定间隔输出 WARN 日志用于定位，需要调用方/运维基于日志告警处理。
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris);

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

    ClientErrorCode Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers);
    std::shared_ptr<SdkInterface> GetSdk(const DataStorageUri &remote_uri);

    // 按 URI hostname 分组，每组关联对应 SDK；任一 hostname 无对应 SDK 时返回错误
    ClientErrorCode GroupBySdk(const std::vector<DataStorageUri> &remote_uris,
                               const BlockBuffers &local_buffers,
                               std::vector<SdkGroup> &groups);

    std::string getOpTypeString(OpType op_type) const;
    ClientErrorCode RunWithTimeoutParallel(OpType op_type,
                                           std::vector<std::function<ClientErrorCode()>> &&tasks,
                                           int timeout_ms) const;
    // 等待所有尚未收集的在途任务执行完成（已通过 get() 收集结果的 future 跳过）。
    // 这是 Get/Put “返回后不再访问调用方内存”契约的兜底：stop flag 只能拦截队列中尚未开始
    // 的任务，已经开始的 SDK I/O 必须等其自行结束。对超过 deadline 仍未结束的任务周期性
    // 打 WARN 日志（存储后端僵死时调用方会阻塞在这里，日志是定位依据）。
    void DrainRunningTasks(std::vector<std::future<ClientErrorCode>> &futures,
                           OpType op_type,
                           std::chrono::steady_clock::time_point deadline) const;
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