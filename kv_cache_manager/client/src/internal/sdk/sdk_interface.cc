#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"

namespace kv_cache_manager {

SdkInterface::GroupMap SdkInterface::SplitByPath(const std::vector<DataStorageUri> &remote_uris,
                                                 const BlockBuffers &local_buffers) {
    GroupMap groups;
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        const auto &uri = remote_uris[i];
        const BlockBuffer &buf = local_buffers[i];
        auto &group = groups[uri.GetPath()];
        // 记录原始下标：下标是 block 的唯一身份，消费方（如 Put 回填 actual_remote_uris）
        // 必须按 indices 保序，禁止依赖 unordered_map 迭代序。
        group.indices.push_back(i);
        group.remote_uris.push_back(uri);
        group.local_buffers.push_back(buf);
    }
    return groups;
}
} // namespace kv_cache_manager