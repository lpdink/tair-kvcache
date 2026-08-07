# Client SDK I/O 契约（Timeout / 取消 / Buffer 生命周期）

> 本文件是 `docs/tasks/safe-timeout/01-contract.md` 的仓库内副本（任务卡约定：
> 任务书目录不在本仓库内时，将规范文本复制到 `docs/design/` 供代码注释引用）。
> **规范以 `docs/tasks/safe-timeout/01-contract.md` 为准；两处不一致时，改本文件，不改规范。**

---

# 01 — Timeout / 取消 / Buffer 生命周期契约（规范文本）

> 本文件是本次交付的**核心产物**。W0 负责把 §2 的措辞落到 `sdk_interface.h` 注释里，把 §3 落到能力矩阵注释里，把本文件链接进 `docs/`。
> 其他卡片实现时以本文件为准；**如果实现与本文件冲突，改实现，不改本文件**（除非人类决策者同意）。

---

## 1. 名词

- **caller buffer**：调用方（推理引擎 connector）拥有的 `BlockBuffer.iovs[].base` 指向的内存。通常是 pinned host memory，也可能是 GPU 显存。
- **I/O 预算 / deadline**：一次 `LoadKvCaches` / `SaveKvCaches` 调用允许消耗的墙钟时间上限（`get/put_timeout_ms`，默认 15000ms）。deadline = 调用进入时刻 + 预算，**是绝对时间点**。
- **hard 契约**：接口返回后，实现方及其委托的任何执行者（本进程线程、内核、fuse、网卡 DMA、远端服务）都**不再读写 caller buffer**。
- **soft 契约**：接口返回后，**仍可能**有后台执行者读写 caller buffer。调用方若立即复用该内存，构成数据竞争 / UAF。

---

## 2. 契约正文（规范措辞，供写入 `sdk_interface.h`）

```
Timeout 与 buffer 生命周期契约：

1) 有界性（所有实现必须满足）
   实现必须在 I/O 预算耗尽后尽快返回，不得无界阻塞调用线程。
   预算通过 Init 阶段的 SdkTimeoutConfig 下传（见 §5），不在 Get/Put 签名中传递。

2) 准入（所有实现必须满足）
   在发起任何一次会写 caller buffer 的 I/O 之前，实现必须检查 deadline。
   已过期则直接返回超时错误，禁止发起该 I/O。
   分批/逐 block 的实现必须在每一批/每一块前检查，而不是只在入口检查一次。

3) Buffer 生命周期（分级，实现必须如实声明自己属于哪一级）
   - hard：返回后 caller buffer 不再被任何执行者访问。
     LocalFileSdk、Hf3fsSdk、TairMempoolSdk（默认 staging 路径）属于此级。
   - soft：返回后仍可能有在飞 I/O 写 caller buffer。
     MooncakeSdk 属于此级，因为 slices 直接指向 caller 内存且上游
     transfer engine 不提供取消语义（见 00-context.md §4）。
     此级实现必须在超时路径输出可归因日志（后端、op、block 下标、buffer 地址）。

4) 调用方义务
   在 soft 级后端参与的调用返回后立即复用 caller buffer，是已知的数据竞争。
   本版本不提供延迟归还 / 隔离机制（sglang 的 host 内存池由引擎自行回收，
   kvcm 无法干预）。该风险通过观测收敛，判据见 §4。
```

### 2.1 明确否决的方案（不要"顺手改回去"）

| 方案 | 否决理由 |
|---|---|
| 在 `SdkWrapper` 层等待 in-flight I/O 完成后再返回 | 该层不持有写入能力，唯一手段是阻塞调用方。sglang 同步路径会从"15s 有界降级"变成"永久阻塞" —— 比原缺陷更严重 |
| 为 mooncake 引入 staging buffer + 超时隔离槽位 | happy path 多一次 memcpy，放弃零拷贝优势；为未证实频率的边界情况付固定成本；池写满仍会失效 |
| 超时后 `unregister_local_memory` / 撤销已提交传输 | 上游无此能力；对正在被 DMA 的内存做 unregister 更危险 |
| 给 `Get/Put` 增加 per-call deadline 参数 | 线上主力后端 PACE 的 timeout 本身就是进程级配置；改签名收益低、影响面大。改为 Init 下传（§5） |
| 新增 `ER_SDK_TIMEOUT_BUFFER_UNSAFE` 之类的分级错误码 | 本轮不做 buffer 生命周期管理，调用方拿到分级信息也无动作；日志/指标已能承载全部归因信息，协议复杂度为负收益 |

---

## 3. 后端履约矩阵（W0 写入注释，各卡实现对齐）

| 后端 | 有界性 | 准入检查 | Buffer 级别 | 备注 |
|---|---|---|---|---|
| LocalFile / NFS | ✅ 逐 block | ✅ 逐 block | **hard** | abort 路径必须 sync GPU stream，否则已入队的 `cudaMemcpyAsync` 会在返回后落到 caller GPU buffer |
| HF3FS | ✅ `hf3fs_wait_for_ios(abs_timeout)` | ✅ 逐 block | **hard** | caller buffer 天然安全（数据先落我们自己的 shm）；超时时**不执行 `CopyIovs`** |
| Mooncake | ✅ 逐 key 前置检查 | ✅ 逐 key | **soft** | 上游无取消语义；逐 key 检查把暴露面从 128 block 降到 ≤1 block |
| TairMempool (PACE) | ✅（PACE 内部 9s/10s） | PACE 侧无队列准入（见 open questions） | **hard** | PACE 已有 `cancel_and_drain` 排干；**本次不改代码** |

---

## 4. 观测与判据（决定 mooncake 后续要不要做 staging）

### 4.1 必须落地的观测

超时路径的日志**必须**包含：`backend`、`op`(get/put)、`deadline_ms`、`elapsed_ms`、`完成的 block 数 / 总数`、**在飞 block 的下标与 caller buffer 地址**、`是否 soft 级后端`。

理由：线上上次事故是从存储侧监控反推的，kvcm 侧完全不可见。这些字段是把"静默污染"变成"可归因故障"的唯一手段。

建议指标（W0 落骨架，可先只打日志）：

| 指标 | 回答的问题 |
|---|---|
| `sdk_io_latency{backend,op}` 分位数 | 15s 阈值离真实分布多远？是否该先调阈值 |
| `sdk_timeout_count{backend,op}` | 超时是否高频？集中在哪个后端 |
| `sdk_admission_reject_count{backend,op}` | §6.1 的排队超时到底有多频繁（**验证本次修复价值**） |
| `sdk_unsafe_return_count{backend}` | 真正的风险敞口（只有 mooncake 会 >0） |
| py `buffer_alloc_wait_ms` | 等槽位时间，避免与 I/O 慢混淆 |

### 4.2 判据（写死，避免下次变成主观争论）

- 若 `sdk_unsafe_return_count` 长期为 0 或极低 → **mooncake 永久不做 staging**，关闭该议题。
- 若不低且集中在 sglang → staging 从"过度设计"升级为"必需"（因为 sglang 无法靠 buffer 生命周期管理规避）。
- 若 `sdk_admission_reject_count` 显著 > 0 → 说明排队超时是真实高频问题，需进一步收窄（如降低 queue_size 或加背压）。

---

## 5. Timeout 下传方式：Init 级，不改 Get/Put 签名

- `SdkTimeoutConfig`（`client/src/internal/config/sdk_config.h:20-33`）已有 `put_timeout_ms` / `get_timeout_ms`，默认各 15000。
- W0 将其**透传进各 SDK 的 `Init`**（经 `SdkBackendConfig`），SDK 内部自行按 op 取用。
- `SdkWrapper` 在每次 `Get/Put` 入口计算**一次** `deadline = now + timeout_ms`，用于：
  1. 线程池任务的准入检查（§6.1）；
  2. 通过 SDK 内部可访问的方式传递给逐 block 检查（实现细节见 W0 卡）。
- **注意不一致点**：PACE 的进程级 timeout 是 get/put 共用一个值，而 kvcm 是分开的两个值。文档需说明；SDK 层按 op 取对应值即可。

---

## 6. 契约的可测试性

因为"复现都不好复现"（线上事故当时无法复现），**测试是本次防回归的唯一手段**。W5 必须覆盖：

1. **运行中超时**：可控 slow fake SDK，验证 wrapper 在 `deadline` 附近返回，不无界等待。
2. **排队超时被拦下**：占满线程池，让后到分组在队列里等过 deadline，断言它**从未发起 I/O**（fake SDK 记录"是否被调用"）。
3. **返回后无后台访问（hard 级）**：LocalFile 场景，返回后立即用哨兵值覆写 caller buffer，ASAN 下断言无竞争写入。
4. **保序**：交错多 path 不同 payload，断言 `actual_remote_uris[i]` 对应 `remote_uris[i]`。

---

## 7. 给未来维护者（含 AI）的警告

> 本文件的存在就是为了防止"顺手把契约改回去"。若你正准备做以下任何一件事，先读 §2.1：
>
> - 在 `SdkWrapper` 里加"等 in-flight 完成"的逻辑；
> - 把 `actual_remote_uris` 简化成 `assign(remote_uris)`；
> - 删掉 LocalFile abort 路径的 `cudaStreamSynchronize`；
> - 把 `hf3fs_wait_for_ios` 的 `abs_timeout` 改回 `nullptr`；
> - 把 mooncake 的 slices 从"逐 key 检查后发起"改成"批量一次性发起"。
>
> 以上每一条都会重新打开一个已知的静默数据损坏窗口。
