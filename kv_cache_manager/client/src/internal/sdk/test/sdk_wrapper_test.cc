#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

using namespace kv_cache_manager;

class SdkWrapperTest : public TESTBASE {
public:
    void SetUp() override {
        client_config_ = CreateTestClientConfig();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = CreateTestStorageConfigs();
        root_path_ = GetPrivateTestRuntimeDataPath();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

private:
    std::unique_ptr<ClientConfig> CreateTestClientConfig() {
        auto client_config = std::make_unique<ClientConfig>();
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "address": [
                "127.0.0.1:8080"
            ],
            "block_size": 128,
            "sdk_config": {
                "thread_num": 8,
                "queue_size": 2000,
                "sdk_backend_configs": [
                    {
                        "type": "file"
                    }
                ],
                "timeout_config": {
                    "put_timeout_ms": 2000,
                    "get_timeout_ms": 2000
                }
            },
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": false,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";
        client_config->FromJsonString(client_config_str);
        return client_config;
    }

    std::string CreateTestStorageConfigs() {
        return "["
               // #ifdef ENABLE_HF3FS
               //                R"({
               //             "type": "hf3fs",
               //             "global_unique_name": "3fs_test",
               //             "storage_spec": {
               //                 "cluster_name": "3fs_cluster",
               //                 "mountpoint": "/3fs/stage/3fs",
               //                 "root_dir": "3fs_test/",
               //                 "key_count_per_file": 2
               //             }
               //         },)"
               // #endif
               R"({
            "type": "file",
            "global_unique_name": "nfs_test",
            "storage_spec": {
                "root_path": "/nfs/",
                "key_count_per_file": 2
            }
        }
    ])";
    }

private:
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::string root_path_;
};

TEST_F(SdkWrapperTest, TestInit) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
}

TEST_F(SdkWrapperTest, TestInitWithEmptyWrapperConfig) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_INVALID_CLIENT_CONFIG, sdk_wrapper.Init(nullptr, init_params_));
}

TEST_F(SdkWrapperTest, TestInitWithEmptyStorageConfigs) {
    SdkWrapper sdk_wrapper;
    InitParams init_params = init_params_;
    init_params.storage_configs = "[]";
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk_wrapper.Init(client_config_, init_params));
}

TEST_F(SdkWrapperTest, TestInitWithInvalidStorageConfigs) {
    SdkWrapper sdk_wrapper;
    InitParams init_params = init_params_;
    init_params.storage_configs = "[invalid json]";
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk_wrapper.Init(client_config_, init_params));
}

// TODO: mock mooncake
//  TEST_F(SdkWrapperTest, TestInitWithMooncake) {
//  #ifdef ENABLE_MOONCAKE
//      auto wrapper_config = CreateTestWrapperConfig();
//      auto mooncake_config = std::make_shared<MooncakeSdkConfig>();
//      mooncake_config->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
//      mooncake_config->set_location("*");
//      mooncake_config->set_put_replica_num(2);
//      wrapper_config->sdk_config_map_[DataStorageType::DATA_STORAGE_TYPE_MOONCAKE] = mooncake_config;
//      InitParams init_params;
//      {
//          SdkWrapper sdk_wrapper;
//          ASSERT_FALSE(sdk_wrapper.Init(wrapper_config, init_params));
//      }
//      {
//          SdkWrapper sdk_wrapper;
//          ASSERT_TRUE(sdk_wrapper.Init(wrapper_config, init_params_));
//      }
//      ASSERT_TRUE(false);
//  #else
//      GTEST_SKIP() << "mooncake not enabled, skipping init sdk wrapper with mooncake config";
//  #endif
//  }

TEST_F(SdkWrapperTest, TestPutAndGet) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
    std::vector<DataStorageUri> remote_uris;
    remote_uris.push_back(DataStorageUri("file://nfs_test/" + root_path_ + "/nfs/0/0/1?blkid=0&size=1024"));
    BlockBuffers local_buffers;
    BlockBuffer buffer;
    local_buffers.push_back(buffer);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperTest, TestValid) {
    SdkWrapper sdk_wrapper;
    std::vector<DataStorageUri> remote_uris;
    remote_uris.push_back(DataStorageUri("file://nfs_test/nfs/0/0/1?blkid=0"));
    BlockBuffers local_buffers;
    ASSERT_EQ(ER_INVALID_PARAMS, sdk_wrapper.Valid(remote_uris, local_buffers));
    BlockBuffer buffer;
    local_buffers.push_back(buffer);
    ASSERT_EQ(ER_OK, sdk_wrapper.Valid(remote_uris, local_buffers));
}

TEST_F(SdkWrapperTest, TestGetSdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
    DataStorageUri remote_uri("file://nfs_test/nfs/0/0/1?blkid=0");
    ASSERT_TRUE(sdk_wrapper.GetSdk(remote_uri));
    remote_uri = DataStorageUri("file://invalid/nfs/1/0/1?blkid=0");
    ASSERT_FALSE(sdk_wrapper.GetSdk(remote_uri));
#if ENABLE_HF3FS && USING_CUDA
    remote_uri = DataStorageUri("3fs://3fs_test/3fs_test/0/1?blkid=0");
    // ASSERT_TRUE(sdk_wrapper.GetSdk(remote_uri));
#endif
    remote_uri = DataStorageUri("invalid:///mnt/nfs/0/0/1?blkid=0");
    ASSERT_FALSE(sdk_wrapper.GetSdk(remote_uri));
}

TEST_F(SdkWrapperTest, TestUpdateMooncakeSdkConfig) {
    SdkWrapper sdk_wrapper;
    RegistSpan span;
    auto sdk_backend_config =
        client_config_->sdk_wrapper_config()->GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_TRUE(sdk_backend_config);
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(sdk_backend_config, nullptr, ""));
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(sdk_backend_config, &span, ""));
#ifdef ENABLE_MOONCAKE
    auto mooncake_config = std::make_shared<MooncakeSdkConfig>();
    mooncake_config->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    mooncake_config->set_location("*");
    mooncake_config->set_put_replica_num(2);
    ASSERT_EQ(ER_INVALID_PARAMS, sdk_wrapper.UpdateMooncakeSdkConfig(mooncake_config, nullptr, ""));
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(mooncake_config, &span, ""));
#endif
}

// ============================================================================
// Multi-storage test fixture
// ============================================================================
class SdkWrapperMultiStorageTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = CreateTestClientConfig();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = CreateMultiStorageConfigs();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

protected:
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::string root_path_;

private:
    std::unique_ptr<ClientConfig> CreateTestClientConfig() {
        auto client_config = std::make_unique<ClientConfig>();
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "address": ["127.0.0.1:8080"],
            "block_size": 128,
            "sdk_config": {
                "thread_num": 8,
                "queue_size": 2000,
                "sdk_backend_configs": [{"type": "file"}],
                "timeout_config": {
                    "put_timeout_ms": 2000,
                    "get_timeout_ms": 2000
                }
            },
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": false,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1
            },
            "location_spec_infos": {"tp0": 1024}
        })";
        client_config->FromJsonString(client_config_str);
        return client_config;
    }

    std::string CreateMultiStorageConfigs() {
        return "["
               R"({"type":"file","global_unique_name":"nfs_a","storage_spec":{"root_path":")" +
               root_path_ + R"(/nfs_a/","key_count_per_file":2}},)"
               R"({"type":"file","global_unique_name":"nfs_b","storage_spec":{"root_path":")" +
               root_path_ + R"(/nfs_b/","key_count_per_file":2}})"
               "]";
    }
};

TEST_F(SdkWrapperMultiStorageTest, TestMixedStoragePutAndGet) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    // 同 backend 使用同 path（不同 blkid），避免 SDK 内部 SplitByPath 导致重排
    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_b/" + root_path_ + "/nfs_b/0/0/1?blkid=1&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=2&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));

    ASSERT_EQ(actual_remote_uris->size(), 3);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(1).ToUriString(), remote_uris[1].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(2).ToUriString(), remote_uris[2].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestSingleStorageBackwardCompat) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=1&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 2);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(1).ToUriString(), remote_uris[1].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestMixedStorageWithInvalidSdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://unknown_backend/" + root_path_ + "/unknown/0/0/1?blkid=1&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_GETSDK_ERROR, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(ER_GETSDK_ERROR, sdk_wrapper.Get(remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestGroupBySdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_b/" + root_path_ + "/nfs_b/0/0/2?blkid=1&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/3?blkid=2&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    std::vector<SdkWrapper::SdkGroup> groups;
    ASSERT_EQ(ER_OK, sdk_wrapper.GroupBySdk(remote_uris, local_buffers, groups));
    ASSERT_EQ(groups.size(), 2);

    // 第一组 nfs_a: indices 0, 2
    ASSERT_EQ(groups[0].indices.size(), 2);
    ASSERT_EQ(groups[0].indices[0], 0);
    ASSERT_EQ(groups[0].indices[1], 2);
    ASSERT_EQ(groups[0].uris.size(), 2);
    ASSERT_EQ(groups[0].buffers.size(), 2);

    // 第二组 nfs_b: index 1
    ASSERT_EQ(groups[1].indices.size(), 1);
    ASSERT_EQ(groups[1].indices[0], 1);
    ASSERT_EQ(groups[1].uris.size(), 1);
    ASSERT_EQ(groups[1].buffers.size(), 1);
}

// ============================================================================
// Timeout / buffer-lifetime tests with a controllable slow fake SDK
// ============================================================================
namespace {

// 可控慢速 SDK：把每个 block 的 I/O 拆成 chunks 段、段间睡眠，模拟耗时后端传输。
// 每次访问 iov.base（模拟真实 I/O 读写调用方内存）都会记账：测试在 Get/Put 返回后
// 调用 MarkReturned()，此后任何访问都计入 accesses_after_return，即契约违例。
class SlowFakeSdk : public SdkInterface {
public:
    struct Options {
        int chunks = 1;
        std::chrono::milliseconds chunk_delay{0};
        ClientErrorCode result = ER_OK;
    };

    explicit SlowFakeSdk(Options options) : options_(std::move(options)) {}

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &, const std::shared_ptr<StorageConfig> &) override {
        return ER_OK;
    }

    SdkType Type() override { return SdkType::LOCAL_FILE; }

    ClientErrorCode Get(const std::vector<DataStorageUri> &, const BlockBuffers &local_buffers) override {
        return DoIo(local_buffers);
    }

    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override {
        actual_remote_uris->assign(remote_uris.begin(), remote_uris.end());
        return DoIo(local_buffers);
    }

    void MarkReturned() { returned_.store(true); }

    bool in_flight() const { return in_flight_.load(); }
    int invocations() const { return invocations_.load(); }
    int total_accesses() const { return total_accesses_.load(); }
    int accesses_after_return() const { return accesses_after_return_.load(); }

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &, std::vector<DataStorageUri> &) override {
        return ER_OK;
    }

private:
    ClientErrorCode DoIo(const BlockBuffers &local_buffers) {
        invocations_.fetch_add(1);
        in_flight_.store(true);
        for (int chunk = 0; chunk < options_.chunks; ++chunk) {
            if (chunk > 0) {
                std::this_thread::sleep_for(options_.chunk_delay);
            }
            for (const auto &buffer : local_buffers) {
                for (const auto &iov : buffer.iovs) {
                    if (iov.base == nullptr || iov.size == 0) {
                        continue;
                    }
                    total_accesses_.fetch_add(1);
                    if (returned_.load()) {
                        accesses_after_return_.fetch_add(1);
                    }
                    // 模拟真实 I/O 写入调用方 buffer
                    std::memset(iov.base, static_cast<int>('A' + (chunk % 26)), 1);
                }
            }
        }
        in_flight_.store(false);
        return options_.result;
    }

    Options options_;
    std::atomic<bool> returned_{false};
    std::atomic<bool> in_flight_{false};
    std::atomic<int> invocations_{0};
    std::atomic<int> total_accesses_{0};
    std::atomic<int> accesses_after_return_{0};
};

} // namespace

class SdkWrapperTimeoutTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = CreateTestClientConfig();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = CreateTimeoutStorageConfigs();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

protected:
    static void InstallFakeSdk(SdkWrapper &wrapper, const std::string &name, std::shared_ptr<SdkInterface> sdk) {
        wrapper.sdk_map_[name] = std::move(sdk);
    }

    DataStorageUri MakeUri(const std::string &storage_name, uint64_t blkid) {
        return DataStorageUri("file://" + storage_name + "/" + root_path_ + "/f?blkid=" + std::to_string(blkid) +
                              "&size=1024");
    }

    static BlockBuffer MakeBuffer(void *base, size_t size) {
        BlockBuffer buffer;
        buffer.iovs.push_back(Iov{MemoryType::CPU, base, size, false});
        return buffer;
    }

    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::string root_path_;

private:
    std::unique_ptr<ClientConfig> CreateTestClientConfig() {
        auto client_config = std::make_unique<ClientConfig>();
        // thread_num=2 用于构造“2 个在途 + 2 个排队”的场景；超时 100ms 配合 fake SDK 的
        // ~200ms I/O 时间，稳定触发超时返回路径
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "block_size": 128,
            "sdk_config": {
                "thread_num": 2,
                "queue_size": 2000,
                "sdk_backend_configs": [{"type": "file"}],
                "timeout_config": {
                    "put_timeout_ms": 100,
                    "get_timeout_ms": 100
                }
            },
            "location_spec_infos": {"tp0": 1024}
        })";
        client_config->FromJsonString(client_config_str);
        return client_config;
    }

    std::string CreateTimeoutStorageConfigs() {
        std::string configs = "[";
        for (int i = 0; i < 4; ++i) {
            if (i > 0) {
                configs += ",";
            }
            configs += R"({"type":"file","global_unique_name":"nfs_)" + std::to_string(i) +
                       R"(","storage_spec":{"root_path":"/tmp/timeout_test/","key_count_per_file":2}})";
        }
        configs += "]";
        return configs;
    }
};

TEST_F(SdkWrapperTimeoutTest, TestGetTimeoutDrainsInFlightBeforeReturn) {
    SdkWrapper wrapper;
    ASSERT_EQ(ER_OK, wrapper.Init(client_config_, init_params_));

    // 6 段 × 40ms ≈ 200ms 的在途 I/O，超过 100ms 超时
    auto slow_sdk = std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{6, std::chrono::milliseconds(40), ER_OK});
    InstallFakeSdk(wrapper, "nfs_0", slow_sdk);

    auto *buffer = malloc(64);
    ASSERT_NE(buffer, nullptr);
    std::vector<DataStorageUri> uris = {MakeUri("nfs_0", 0)};
    BlockBuffers buffers = {MakeBuffer(buffer, 64)};

    auto begin = std::chrono::steady_clock::now();
    ASSERT_EQ(ER_SDK_TIMEOUT, wrapper.Get(uris, buffers));
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    // 返回即在途任务已结束：不再有任何线程访问调用方 buffer
    slow_sdk->MarkReturned();
    EXPECT_FALSE(slow_sdk->in_flight());
    EXPECT_EQ(slow_sdk->accesses_after_return(), 0);
    EXPECT_EQ(slow_sdk->total_accesses(), 6);
    // 返回时间 = 超时点 + 等待在途 I/O 完成（≈200ms），而不是超时点直接返回
    EXPECT_GE(elapsed_ms, 190);

    // 调用方可以立即复用 buffer：再装一个快速 SDK 发起新的 Get
    auto fast_sdk = std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{});
    InstallFakeSdk(wrapper, "nfs_0", fast_sdk);
    EXPECT_EQ(ER_OK, wrapper.Get(uris, buffers));
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_EQ(slow_sdk->accesses_after_return(), 0);
    free(buffer);
}

TEST_F(SdkWrapperTimeoutTest, TestPutTimeoutDrainsInFlightBeforeReturn) {
    SdkWrapper wrapper;
    ASSERT_EQ(ER_OK, wrapper.Init(client_config_, init_params_));

    auto slow_sdk = std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{6, std::chrono::milliseconds(40), ER_OK});
    InstallFakeSdk(wrapper, "nfs_0", slow_sdk);

    auto *buffer = malloc(64);
    ASSERT_NE(buffer, nullptr);
    std::vector<DataStorageUri> uris = {MakeUri("nfs_0", 0)};
    BlockBuffers buffers = {MakeBuffer(buffer, 64)};
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();

    auto begin = std::chrono::steady_clock::now();
    ASSERT_EQ(ER_SDK_TIMEOUT, wrapper.Put(uris, buffers, actual_remote_uris));
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    slow_sdk->MarkReturned();
    EXPECT_FALSE(slow_sdk->in_flight());
    EXPECT_EQ(slow_sdk->accesses_after_return(), 0);
    EXPECT_EQ(slow_sdk->total_accesses(), 6);
    EXPECT_GE(elapsed_ms, 190);

    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_EQ(slow_sdk->accesses_after_return(), 0);
    free(buffer);
}

TEST_F(SdkWrapperTimeoutTest, TestFailFastKeepsFirstSdkErrorAndDrains) {
    SdkWrapper wrapper;
    ASSERT_EQ(ER_OK, wrapper.Init(client_config_, init_params_));

    auto error_sdk =
        std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{1, std::chrono::milliseconds(10), ER_SDKREAD_ERROR});
    auto slow_sdk = std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{6, std::chrono::milliseconds(40), ER_OK});
    InstallFakeSdk(wrapper, "nfs_0", error_sdk);
    InstallFakeSdk(wrapper, "nfs_1", slow_sdk);

    auto *buffer0 = malloc(64);
    auto *buffer1 = malloc(64);
    ASSERT_NE(buffer0, nullptr);
    ASSERT_NE(buffer1, nullptr);
    std::vector<DataStorageUri> uris = {MakeUri("nfs_0", 0), MakeUri("nfs_1", 0)};
    BlockBuffers buffers = {MakeBuffer(buffer0, 64), MakeBuffer(buffer1, 64)};

    auto begin = std::chrono::steady_clock::now();
    // fail-fast：保留首个具体 SDK 错误码，不被 ER_SDK_TIMEOUT 覆盖
    ASSERT_EQ(ER_SDKREAD_ERROR, wrapper.Get(uris, buffers));
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    // 错误返回路径同样要等在途的慢任务收尾后才返回
    slow_sdk->MarkReturned();
    EXPECT_FALSE(slow_sdk->in_flight());
    EXPECT_EQ(slow_sdk->accesses_after_return(), 0);
    EXPECT_GE(elapsed_ms, 190);

    free(buffer0);
    free(buffer1);
}

TEST_F(SdkWrapperTimeoutTest, TestQueuedTasksCanceledWithoutTouchingBuffers) {
    SdkWrapper wrapper;
    ASSERT_EQ(ER_OK, wrapper.Init(client_config_, init_params_));

    // 线程池只有 2 个线程：4 个任务中 2 个在途执行、2 个在队列中等待
    std::vector<std::shared_ptr<SlowFakeSdk>> sdks;
    std::vector<DataStorageUri> uris;
    BlockBuffers buffers;
    std::vector<void *> raw_buffers;
    for (int i = 0; i < 4; ++i) {
        auto sdk = std::make_shared<SlowFakeSdk>(SlowFakeSdk::Options{4, std::chrono::milliseconds(40), ER_OK});
        InstallFakeSdk(wrapper, "nfs_" + std::to_string(i), sdk);
        sdks.push_back(sdk);
        uris.push_back(MakeUri("nfs_" + std::to_string(i), 0));
        auto *buffer = malloc(64);
        ASSERT_NE(buffer, nullptr);
        raw_buffers.push_back(buffer);
        buffers.push_back(MakeBuffer(buffer, 64));
    }

    ASSERT_EQ(ER_SDK_TIMEOUT, wrapper.Get(uris, buffers));

    // 排队任务被 stop flag 拦截、从未执行；在途任务在返回前已完成
    int invoked = 0;
    for (auto &sdk : sdks) {
        sdk->MarkReturned();
        if (sdk->invocations() > 0) {
            ++invoked;
        }
        EXPECT_FALSE(sdk->in_flight());
        EXPECT_EQ(sdk->accesses_after_return(), 0);
    }
    EXPECT_EQ(invoked, 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    for (auto &sdk : sdks) {
        EXPECT_EQ(sdk->accesses_after_return(), 0);
    }
    for (auto *buffer : raw_buffers) {
        free(buffer);
    }
}
