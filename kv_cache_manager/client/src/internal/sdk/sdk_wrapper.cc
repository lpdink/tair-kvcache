#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"

#include <unordered_map>

#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_io_stats.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

SdkWrapper::SdkWrapper() : sdk_factory_(SdkFactory::GetInstance()) {}

SdkWrapper::~SdkWrapper() {}

ClientErrorCode SdkWrapper::Init(const std::unique_ptr<ClientConfig> &client_config, const InitParams &init_params) {
    if (!client_config) {
        KVCM_LOG_WARN("client config is null");
        return ER_INVALID_CLIENT_CONFIG;
    }
    wrapper_config_ = client_config->sdk_wrapper_config();
    if (!wrapper_config_) {
        KVCM_LOG_WARN("sdk wrapper config is null");
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    const std::string &storage_configs = init_params.storage_configs;
    if (!Jsonizable::FromJsonString(storage_configs, storage_configs_)) {
        KVCM_LOG_WARN("parse storage config failed, storage config: %s", storage_configs.c_str());
        return ER_INVALID_STORAGE_CONFIG;
    }
    if (storage_configs_.empty()) {
        KVCM_LOG_WARN("storage config is empty");
        return ER_INVALID_STORAGE_CONFIG;
    }

    wait_task_thread_pool_ = std::make_unique<LockFreeThreadPool>(
        wrapper_config_->thread_num(), wrapper_config_->queue_size(), "SdkWaitTaskPool");
    if (!wait_task_thread_pool_->start()) {
        KVCM_LOG_WARN("start wait task thread pool failed, thread num: %zu, queue size: %zu",
                      wrapper_config_->thread_num(),
                      wrapper_config_->queue_size());
        return ER_THREADPOOL_ERROR;
    }

    auto regist_span = init_params.regist_span;
    const auto &location_spec_infos = client_config->location_spec_infos();
    if (location_spec_infos.empty()) {
        KVCM_LOG_WARN("location_spec_infos is empty");
        return ER_INVALID_CLIENT_CONFIG;
    }

    // 验证 self_location_spec_name 存在于 location_spec_infos 中
    if (location_spec_infos.find(init_params.self_location_spec_name) == location_spec_infos.end()) {
        KVCM_LOG_WARN("location_spec_infos does not contain self_location_spec_name [%s]",
                      init_params.self_location_spec_name.c_str());
        return ER_INVALID_CLIENT_CONFIG;
    }

    for (const auto &storage_config : storage_configs_) {
        DataStorageType type = storage_config->type();
        const auto &sdk_backend_config = wrapper_config_->GetSdkBackendConfig(type);
        if (!sdk_backend_config) {
            KVCM_LOG_WARN("sdk backend config is null, storage config: %s", storage_config->ToString().c_str());
            return ER_INVALID_SDKBACKEND_CONFIG;
        }
        auto ec = UpdateMooncakeSdkConfig(sdk_backend_config, regist_span, init_params.self_location_spec_name);
        if (ec != ER_OK) {
            KVCM_LOG_WARN("fill span failed, storage config: %s", storage_config->ToString().c_str());
            return ec;
        }

        // 将完整的 spec → byte_size_per_block 映射传给 SDK
        sdk_backend_config->set_spec_byte_sizes_per_block(location_spec_infos);

        // Init 级 timeout 透传：各 SDK 在 Init 时可读取 get/put_timeout_ms（统一入口）。
        // 注意：PACE 等后端的 timeout 是进程级配置（如 TAIR_MEMPOOL_CLIENT_IO_TIMEOUT_MS），
        // 此处仅是 kvcm 侧的统一入口；SDK 内部主要使用 SdkDeadline 做逐 block/逐 key 准入检查。
        sdk_backend_config->set_timeout_config(wrapper_config_->timeout_config());

        auto sdk = sdk_factory_->CreateSdk(type, sdk_backend_config, storage_config);
        if (!sdk) {
            KVCM_LOG_WARN("create sdk failed, storage config: %s", storage_config->ToString().c_str());
            return ER_CREATESDK_ERROR;
        }
        sdk_map_.insert({storage_config->global_unique_name(), sdk});
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::GroupBySdk(const std::vector<DataStorageUri> &remote_uris,
                                       const BlockBuffers &local_buffers,
                                       std::vector<SdkGroup> &groups) {
    std::unordered_map<std::string, size_t> hostname_to_group;
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        std::string hostname = remote_uris[i].GetHostName();
        if (hostname.empty()) {
            KVCM_LOG_WARN("GroupBySdk: remote_uri %s has empty hostname", remote_uris[i].ToUriString().c_str());
            return ER_GETSDK_ERROR;
        }
        auto it = hostname_to_group.find(hostname);
        if (it == hostname_to_group.end()) {
            auto sdk = GetSdk(remote_uris[i]);
            if (!sdk) {
                KVCM_LOG_WARN("GroupBySdk: no sdk found for hostname: %s", hostname.c_str());
                return ER_GETSDK_ERROR;
            }
            hostname_to_group[hostname] = groups.size();
            groups.push_back({sdk, {}, {}, {}});
        }
        auto &group = groups[hostname_to_group[hostname]];
        group.indices.push_back(i);
        group.uris.push_back(remote_uris[i]);
        group.buffers.push_back(local_buffers[i]);
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        return ec;
    }
    std::vector<SdkGroup> groups;
    ec = GroupBySdk(remote_uris, local_buffers, groups);
    if (ec != ER_OK) {
        return ec;
    }

    // Build task vector for parallel dispatch
    std::vector<TimedTask> timed_tasks;
    timed_tasks.reserve(groups.size());
    int timeout_ms = wrapper_config_->timeout_config().get_timeout_ms();
    // deadline 在调用入口计算一次，向下传播（线程池准入检查 + SDK 内部逐 block 检查共用）。
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (size_t i = 0; i < groups.size(); ++i) {
        const auto &group = groups[i];
        TimedTask task;
        task.sdk_type = group.sdk->Type();
        task.group_index = i + 1;
        task.group_count = groups.size();
        task.block_count = group.uris.size();
        // Capture group by value to prevent use-after-free on timeout
        task.fn = [group]() { return group.sdk->Get(group.uris, group.buffers); };
        timed_tasks.push_back(std::move(task));
    }
    return RunWithTimeoutParallel(OpType::GET, std::move(timed_tasks), deadline, timeout_ms);
}

ClientErrorCode SdkWrapper::Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("put failed, remote_uris or local_buffers invalid.");
        return ec;
    }
    std::vector<SdkGroup> groups;
    ec = GroupBySdk(remote_uris, local_buffers, groups);
    if (ec != ER_OK) {
        return ec;
    }
    actual_remote_uris->resize(remote_uris.size());

    // Build task vector and result containers for parallel dispatch
    std::vector<TimedTask> timed_tasks;
    std::vector<std::shared_ptr<std::vector<DataStorageUri>>> group_results;
    timed_tasks.reserve(groups.size());
    group_results.reserve(groups.size());
    int timeout_ms = wrapper_config_->timeout_config().put_timeout_ms();
    // deadline 在调用入口计算一次，向下传播（线程池准入检查 + SDK 内部逐 block 检查共用）。
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (size_t i = 0; i < groups.size(); ++i) {
        const auto &group = groups[i];
        auto group_actual_uris = std::make_shared<std::vector<DataStorageUri>>();
        group_results.push_back(group_actual_uris);
        TimedTask task;
        task.sdk_type = group.sdk->Type();
        task.group_index = i + 1;
        task.group_count = groups.size();
        task.block_count = group.uris.size();
        // Capture group by value to prevent use-after-free on timeout
        task.fn = [group, group_actual_uris]() {
            return group.sdk->Put(group.uris, group.buffers, group_actual_uris);
        };
        timed_tasks.push_back(std::move(task));
    }

    ec = RunWithTimeoutParallel(OpType::PUT, std::move(timed_tasks), deadline, timeout_ms);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("put failed, sdk error: %d", static_cast<int>(ec));
        return ec;
    }

    // Result aggregation and validation
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto &group = groups[i];
        const auto &group_actual_uris = group_results[i];
        
        if (group_actual_uris->size() != group.indices.size()) {
            KVCM_LOG_WARN("sdk returned mismatched actual_uris size: %zu vs %zu",
                          group_actual_uris->size(),
                          group.indices.size());
            return ER_SDKWRITE_ERROR;
        }
        for (size_t j = 0; j < group.indices.size(); ++j) {
            (*actual_remote_uris)[group.indices[j]] = (*group_actual_uris)[j];
        }
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers) {
    if (remote_uris.empty() || local_buffers.empty() || remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_WARN(
            "Check failed, remote_uris or local_buffers invalid, remote_uris size: %zu, local_buffers size: %zu",
            remote_uris.size(),
            local_buffers.size());
        return ER_INVALID_PARAMS;
    }

    const auto it = std::find_if(remote_uris.begin(), remote_uris.end(), [](const auto &uri) { return !uri.Valid(); });
    if (it != remote_uris.end()) {
        KVCM_LOG_WARN("Check failed, remote_uri %s invalid", it->ToUriString().c_str());
        return ER_INVALID_PARAMS;
    }
    return ER_OK;
}

std::shared_ptr<SdkInterface> SdkWrapper::GetSdk(const DataStorageUri &remote_uri) {
    std::string host_name = remote_uri.GetHostName();
    if (host_name.empty()) {
        KVCM_LOG_WARN("get sdk for remote_uri %s failed, remote_uri's host name is empty",
                      remote_uri.ToUriString().c_str());
        return nullptr;
    }
    auto it = sdk_map_.find(host_name);
    if (it != sdk_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string SdkWrapper::getOpTypeString(OpType op_type) const {
    switch (op_type) {
    case OpType::GET: {
        return "get";
    }
    case OpType::PUT: {
        return "put";
    }
    }
    return "unknown";
}

ClientErrorCode SdkWrapper::RunWithTimeoutParallel(OpType op_type,
                                                   std::vector<TimedTask> &&tasks,
                                                   SdkDeadline::TimePoint deadline,
                                                   int timeout_ms) const {
    if (tasks.empty()) {
        return ER_OK;
    }

    // Check capacity before submitting any tasks
    if (wait_task_thread_pool_->isFull()) {
        KVCM_LOG_WARN("run %s parallel failed, wait task thread pool is full",
                      getOpTypeString(op_type).c_str());
        return ER_THREADPOOL_ERROR;
    }

    const bool is_get = (op_type == OpType::GET);
    const std::string op_str = getOpTypeString(op_type);
    // 调用入口时间 = deadline - timeout_ms，用于日志里的 elapsed_ms 归因。
    auto start = deadline - std::chrono::milliseconds(timeout_ms);

    std::vector<std::future<ClientErrorCode>> futures;
    futures.reserve(tasks.size());

    for (auto &task : tasks) {
        // 准入检查：任务真正开始执行时先看绝对 deadline。
        // 这是本次修复的核心（00-context.md §6.1）：排队 14.9s 的任务过去会照样发起
        // I/O，导致 caller 早已返回而 I/O 刚开始写它的 buffer。
        // 权威依据只有 deadline；不再使用 stop flag（对未启动任务与 deadline 检查冗余，
        // 对已启动任务无效），保持单一事实来源。
        auto wrapped = [deadline, timeout_ms, op_str, is_get, task]() -> ClientErrorCode {
            if (std::chrono::steady_clock::now() >= deadline) {
                auto overdue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - deadline)
                                      .count();
                SdkIoStats::Instance().OnAdmissionReject(task.sdk_type, is_get, overdue_ms);
                KVCM_LOG_WARN("sdk admission reject: backend=%s op=%s timeout_ms=%d overdue_ms=%lld group=%zu/%zu "
                              "blocks=%zu, deadline already passed, skip I/O to protect caller buffer",
                              SdkTypeToString(task.sdk_type).c_str(),
                              op_str.c_str(),
                              timeout_ms,
                              static_cast<long long>(overdue_ms),
                              task.group_index,
                              task.group_count,
                              task.block_count);
                return ER_SDK_TIMEOUT;
            }
            // 把 deadline 传播给 SDK 内部（逐 block/逐 key 准入检查用，见 sdk_deadline.h）。
            SdkDeadline::Scope scope(deadline);
            return task.fn();
        };
        futures.push_back(wait_task_thread_pool_->async(std::move(wrapped)));
    }

    // 等待：每个 future 用 wait_until(deadline)，超时即刻返回，不 drain、不等待 in-flight 任务。
    // ⚠️ future 生命周期：autil::ThreadPoolBase::async 返回的是 std::packaged_task 的 future
    //    （promise-backed），其析构不阻塞（只有 std::async 启动的 future 析构才可能阻塞）。
    //    因此超时后直接放弃未完成的 future 是安全的，调用线程不会被 in-flight I/O 拖住。
    for (size_t i = 0; i < futures.size(); ++i) {
        if (futures[i].wait_until(deadline) != std::future_status::ready) {
            const auto &task = tasks[i];
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            SdkIoStats::Instance().OnTimeout(task.sdk_type, is_get, elapsed_ms, /*done=*/i, tasks.size());
            KVCM_LOG_WARN("run %s parallel timeout: backend=%s op=%s timeout_ms=%d elapsed_ms=%lld group=%zu/%zu "
                          "blocks=%zu, return immediately without waiting in-flight I/O",
                          op_str.c_str(),
                          SdkTypeToString(task.sdk_type).c_str(),
                          op_str.c_str(),
                          timeout_ms,
                          static_cast<long long>(elapsed_ms),
                          task.group_index,
                          task.group_count,
                          task.block_count);
            return ER_SDK_TIMEOUT;
        }

        auto ec = futures[i].get();
        if (ec != ER_OK) {
            // 失败即刻返回；不等待其余 in-flight 任务（见 01-contract.md §2.1 否决项）。
            KVCM_LOG_WARN("run %s parallel failed, sdk error: %d, group %zu/%zu",
                          op_str.c_str(),
                          static_cast<int>(ec),
                          tasks[i].group_index,
                          tasks[i].group_count);
            return ec;
        }
    }

    return ER_OK;
}

ClientErrorCode SdkWrapper::UpdateMooncakeSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                                    RegistSpan *span,
                                                    const std::string &self_location_spec_name) {
    if (DataStorageType::DATA_STORAGE_TYPE_MOONCAKE != sdk_backend_config->type()) {
        return ER_OK;
    }
    auto config = std::dynamic_pointer_cast<MooncakeSdkConfig>(sdk_backend_config);
    if (!config) {
        KVCM_LOG_WARN("convert to mooncake config failed");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (span == nullptr) {
        KVCM_LOG_WARN("regist span is null but mooncake config is not null");
        return ER_INVALID_PARAMS;
    }
    if (config->local_mem_ptr() != nullptr || config->local_buffer_size() != 0) {
        KVCM_LOG_WARN("local mem ptr already set, not support register multi mooncake sdk");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    config->set_local_mem_ptr(span->base);
    config->set_local_buffer_size(span->size);
    config->set_self_location_spec_name(self_location_spec_name);
    return ER_OK;
}

} // namespace kv_cache_manager