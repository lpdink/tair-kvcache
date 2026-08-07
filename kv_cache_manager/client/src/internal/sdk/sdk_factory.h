#pragma once

#include <functional>
#include <map>
#include <memory>

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class SdkFactory {
public:
    static SdkFactory *GetInstance();

    // 仅供测试使用：为某个 DataStorageType 注册自定义构造函数，Create 时优先使用。
    // 命中自定义 creator 时跳过默认 switch，但仍会走 sdk->Init(...)。
    // 生产代码不得调用。
    using SdkCreator = std::function<std::shared_ptr<SdkInterface>(
        const std::shared_ptr<SdkBackendConfig> &, const std::shared_ptr<StorageConfig> &)>;
    void RegisterCustomCreatorForTest(DataStorageType type, SdkCreator creator);
    void ClearCustomCreatorsForTest();

    std::shared_ptr<SdkInterface> CreateSdk(const DataStorageType &type,
                                            const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            const std::shared_ptr<StorageConfig> &storage_config);

private:
    SdkFactory() = default;
    // 测试注入的 creator 表；仅测试线程串行读写（gtest 默认串行），无需加锁。
    std::map<DataStorageType, SdkCreator> custom_creators_;
};

} // namespace kv_cache_manager
