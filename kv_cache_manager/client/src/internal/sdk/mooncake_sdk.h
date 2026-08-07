#pragma once

#include "3rdparty/mooncake/client_c.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "sdk_interface.h"

namespace kv_cache_manager {

// ============================================================
// Buffer 生命周期：soft 契约（mooncake 是唯一达不到 hard 的后端，勿误以为它安全）
// ============================================================
//
// 本 SDK 把 caller 的 iov.base 直接作为 mooncake slices 提交（见 extractSlices），
// 网卡 DMA 直接读写 caller 内存。上游 mooncake transfer engine 不提供取消语义
// （三重实证，见 docs/tasks/safe-timeout/00-context.md §4，勿再质疑）：
//   - transfer_engine.h 明确注释 "A wait timeout does not cancel the transfer.
//     Keep this operation and its CPU/GPU buffers alive until a later wait
//     reaches completion." —— 即上游要求 caller 让 buffer 活着，与"返回后可复用"冲突；
//   - Transport::freeBatchID 在任务未完成时拒绝释放（析构路径也是死路）；
//   - TENT 的 cancelTransfer 仅能取消尚未启动(PENDING)的任务，对已启动的直接返回 OK
//     （危险的假成功），且本仓库钉住的版本没有接这条路径。
// 因此本 SDK 超时返回后，caller buffer 可能仍被在飞 DMA 写入。
//
// 本版本不引入 staging buffer（会给 happy path 增加一次 memcpy 并放弃 mooncake
// 唯一的零拷贝优势），而是：
//   1) 逐 key 准入检查：每次 mooncake_client_get/put 之前检查 SdkDeadline，
//      把超时时刻的暴露面限制为最多 1 个 block（见 Get/Put 内注释）；
//   2) 超时路径输出可归因日志 + SdkIoStats::OnUnsafeReturn 计数
//      （sdk_unsafe_return_count 是决定未来是否做 staging 的判据，
//       见 docs/tasks/safe-timeout/01-contract.md §4）。
// 调用方义务：soft 级后端返回后立即复用 caller buffer 是已知数据竞争。
class MooncakeSdk : public SdkInterface {
public:
    MooncakeSdk() {}
    ~MooncakeSdk();

    ClientErrorCode Close();

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &storage_config) override;
    SdkType Type() override;
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) override;

    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override;

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                          std::vector<DataStorageUri> &alloc_uris) override;

private:
    std::pair<size_t, bool>
    extractSlices(const MooncakeRemoteItem &item, const BlockBuffer &buffer, std::vector<Slice_t> &slices) const;

private:
    client_t client_{nullptr};
    std::shared_ptr<MooncakeSdkConfig> sdk_backend_config_;
    std::shared_ptr<StorageConfig> storage_config_;
};

} // namespace kv_cache_manager