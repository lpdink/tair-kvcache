# Client SDK I/O 契约（deadline / 取消 / Buffer 生命周期）

> 本文档是 KVCM client SDK 层 timeout 与 buffer 生命周期契约的唯一仓库内规范，
> 代码注释（`sdk_interface.h`）与本文件保持一致；两处不一致时以本文件为准并修正注释。

---

## 1. 名词

- **caller buffer**：调用方（推理引擎 connector）拥有的 `BlockBuffer.iovs[].base` 指向的内存。通常是 pinned host memory，也可能是 GPU 显存。
- **deadline**：一次 `LoadKvCaches` / `SaveKvCaches` 调用允许消耗的墙钟时间上限，表示为**绝对时间点**（`int64_t deadline_us`，steady_clock / `CLOCK_MONOTONIC` 微秒）。
- **hard 契约**：接口返回后，实现方及其委托的任何执行者（本进程线程、内核、fuse、网卡 DMA、远端服务）都**不再读写 caller buffer**。
- **soft 契约**：接口返回后，**仍可能**有后台执行者读写 caller buffer。调用方若立即复用该内存，构成数据竞争 / UAF。

---

## 2. 契约正文（规范措辞，与 `sdk_interface.h` 注释一致）

```
Timeout 与 buffer 生命周期契约：

0) deadline 显式透传
   Get/Put（及上层的 LoadKvCaches/SaveKvCaches）最后一个参数是 int64_t deadline_us：
   绝对时间点（steady_clock 微秒），由调用方（Python connector）逐请求计算后一路传入。
   deadline_us == 0 表示调用方不施加 deadline：SDK 内部不做准入/等待限制，
   SdkWrapper 层仍以 wrapper 级 timeout_config（get/put_timeout_ms，默认 15000ms）兜底
   —— 即 PR 引入透传之前的行为（向后兼容）。

   为什么必须是显式参数（历史：曾否决 per-call deadline 参数，后推翻，理由）：
   a) deadline 是逐请求动态计算的值（写路径 = min(租约剩余, 自律超时)，见 §5），
      无法用 Init 级进程配置表达；
   b) thread_local 传播对 PACE 无效——PACE 的 I/O 在它自己的线程池 worker 线程上执行，
      thread_local 到不了那里；
   c) 旧设计靠「wrapper 超时(15s) > PACE 内部超时(10s)」的隐式配置不变量祈祷 PACE
      先超时，显式透传把这条不变量变成可执行语义（PACE 直接按 deadline 到点停）。

1) 有界性（所有实现必须满足）
   实现必须在 deadline 耗尽后尽快返回，不得无界阻塞调用线程。
   deadline_us == 0 时按各自后端默认行为（PACE/hf3fs 走后端自身超时）。

2) 准入（所有实现必须满足）
   在发起任何一次会写 caller buffer 的 I/O 之前，实现必须检查 deadline_us。
   已过期则直接返回超时错误，禁止发起该 I/O。
   分批/逐 block 的实现必须在每一批/每一块前检查，而不是只在入口检查一次。
   （SdkWrapper 的线程池任务启动前也做一次准入检查——排队超时的任务不允许再发起 I/O，
     见 sdk_wrapper.cc RunWithTimeoutParallel。）

3) Buffer 生命周期（分级，实现必须如实声明自己属于哪一级）
   - hard：返回后 caller buffer 不再被任何执行者访问。
     LocalFileSdk、Hf3fsSdk、TairMempoolSdk（默认 staging 路径）属于此级。
   - soft：返回后仍可能有在飞 I/O 写 caller buffer。
     MooncakeSdk 属于此级，因为 slices 直接指向 caller 内存且上游
     transfer engine 不提供取消语义。
     此级实现必须在超时路径输出可归因日志（后端、op、block 下标、buffer 地址）。

4) 调用方义务
   在 soft 级后端参与的调用返回后立即复用 caller buffer，是已知的数据竞争。
   本版本不提供延迟归还 / 隔离机制（sglang 的 host 内存池由引擎自行回收，
   kvcm 无法干预）。该风险通过观测收敛，判据见 §4。
```

### 2.1 明确否决的方案（不要"顺手改回去"）

| 方案 | 否决理由 |
|---|---|
| 在 `SdkWrapper` 层等待 in-flight I/O 完成后再返回 | 该层不持有写入能力，唯一手段是阻塞调用方。sglang 同步路径会从"15s 有界降级"变成"永久阻塞"——比原缺陷更严重 |
| 为 mooncake 引入 staging buffer + 超时隔离槽位 | happy path 多一次 memcpy，放弃零拷贝优势；为未证实频率的边界情况付固定成本；池写满仍会失效 |
| 超时后 `unregister_local_memory` / 撤销已提交传输 | 上游无此能力；对正在被 DMA 的内存做 unregister 更危险 |
| 回到 thread_local（`SdkDeadline`）隐式传播 | 传不进 PACE worker 线程（见 §2(0)b），且 deadline 是逐请求动态值，本应进签名 |
| 把 deadline 限制回 Init 级配置（`SdkBackendConfig::timeout_config`） | 无法表达"min(租约, 自律)"这类逐请求语义；`SdkBackendConfig` 的 Init 级 timeout 下传路径已删除，存储后端只认 Get/Put 传入的 deadline_us |
| 新增 `ER_SDK_TIMEOUT_BUFFER_UNSAFE` 之类的分级错误码 | 本轮不做 buffer 生命周期管理，调用方拿到分级信息也无动作；日志/指标已能承载全部归因信息，协议复杂度为负收益 |

---

## 3. 后端履约矩阵（实现必须与其声明一致）

| 后端 | 有界性 | 准入检查 | Buffer 级别 | 备注 |
|---|---|---|---|---|
| LocalFile / NFS | ✅ 逐 block | ✅ 逐 block | **hard** | abort 路径必须 sync GPU stream，否则已入队的 `cudaMemcpyAsync` 会在返回后落到 caller GPU buffer |
| HF3FS | ✅ `hf3fs_wait_for_ios(abs_timeout)` | ✅ 逐 block | **hard** | caller buffer 天然安全（数据先落我们自己的 shm）；超时时**不执行 `CopyIovs`** |
| Mooncake | ✅ 逐 key 前置检查 | ✅ 逐 key | **soft** | 上游无取消语义；逐 key 检查把暴露面从 128 block 降到 ≤1 block |
| TairMempool (PACE) | ✅（PACE 内部 9s/10s，deadline 透传后按 deadline 到点停） | PACE 侧无队列准入 | **hard**（默认 staging 路径） | PACE 已有 `cancel_and_drain` 排干 + `BufferUseGuard`（Dekker）保护 staging memcpy；deadline 由闭源 TairMempoolSdk 交给 pace 库（GARWDesc.deadline_us），**增强而非替代**现有取消机制 |

---

## 4. 观测与判据（决定 mooncake 后续要不要做 staging）

### 4.1 必须落地的观测

超时路径的日志**必须**包含：`backend`、`op`(get/put)、`deadline_ms`、`elapsed_ms`、`完成的 block 数 / 总数`、**在飞 block 的下标与 caller buffer 地址**、`是否 soft 级后端`。

理由：线上上次事故是从存储侧监控反推的，kvcm 侧完全不可见。这些字段是把"静默污染"变成"可归因故障"的唯一手段。

指标（`sdk_io_stats`）：

| 指标 | 回答的问题 |
|---|---|
| `sdk_io_latency{backend,op}` 分位数 | 15s 阈值离真实分布多远？是否该先调阈值 |
| `sdk_timeout_count{backend,op}` | 超时是否高频？集中在哪个后端 |
| `sdk_admission_reject_count{backend,op}` | 排队超时到底有多频繁（**验证准入修复价值**） |
| `sdk_unsafe_return_count{backend}` | 真正的风险敞口（只有 mooncake 会 >0） |
| py `buffer_alloc_wait_ms` | 等槽位时间，避免与 I/O 慢混淆 |

### 4.2 判据（写死，避免下次变成主观争论）

- 若 `sdk_unsafe_return_count` 长期为 0 或极低 → **mooncake 永久不做 staging**，关闭该议题。
- 若不低且集中在 sglang → staging 从"过度设计"升级为"必需"（因为 sglang 无法靠 buffer 生命周期管理规避）。
- 若 `sdk_admission_reject_count` 显著 > 0 → 说明排队超时是真实高频问题，需进一步收窄（如降低 queue_size 或加背压）。

---

## 5. deadline 的来源与计算（调用方 = Python connector）

两个性质不同的超时决定 deadline 的取值：

| | 租约超时 `write_timeout_seconds`（默认 30s） | 自律超时 `sdk_get/put_timeout_ms`（默认 15000ms） |
|---|---|---|
| 谁定 | KVCM meta 服务，`start_write_cache` 时申请到的租约 | worker 本地配置 |
| 保护谁 | **外层**：此期间 KVCM 承诺不 reclaim 该 URI；超期 URI 会被分给别人 → 跨 client 脏写，绝不能打破 | **内层**：worker 自我约束单次 transfer 时长，超时快速失败 |
| 起点 T0 | `start_write_cache` **拿到响应的时刻**（租约从这里起算） | 任务**提交到传输线程池的时刻** T_submit |
| 只在 | 写路径（读路径无租约） | 读、写路径都有 |

**deadline 计算规则（Python 侧，全部用 `time.monotonic_ns() // 1000` 取 us）：**

```
写路径: deadline_us = min(T0 + write_timeout_seconds*1e6,  T_submit + sdk_put_timeout_ms*1e3)
读路径: deadline_us = T_submit + sdk_get_timeout_ms*1e3          # 无租约项
```

- **取 min**：只用自律(15s)，任务在传输线程池排队 20s 后启动，自律 DDL=35s 会越过 30s 租约 → 外层脏写；只用租约(30s)，则放弃 worker 快速失败。取 min 两者兼顾。
- **读路径无租约**：读 uri 来自 `GetCacheLocation`（结果带 1s TTL 本地缓存），不申请租约——读不独占、KVCM 不承诺"读期间不 reclaim"。读到一半 URI 被 reclaim 是另一层 reclaimer 竞态，不归 connector deadline 管。
- **时钟一致性（已实测）**：Python `time.monotonic_ns()` 与 C++ `std::chrono::steady_clock` 同为 `CLOCK_MONOTONIC`，C++ 侧收到 deadline_us 后直接与本地 `steady_clock::now()` 换算的 us 数值比较，无需传时钟源、无需校正。

**跨 rank 一致性（sglang/trtllm 写路径）**：只有 rank0 调 `start_write_cache`，非 rank0 的"拿到响应时刻"是 broadcast 收包时刻（晚于 rank0）。**rank0 算好 `deadline_us`（租约项）随 broadcast / 元数据下发，全 rank 用同一个值**；各 rank 再各自叠加自律项取 min（自律项以本地 T_submit 为准，允许不同）。

**透传链路**：Python connector → `TransferClient.LoadKvCaches/SaveKvCaches(..., deadline_us, trace_info)` → `SdkWrapper.Get/Put(..., deadline_us)` → `SdkInterface::Get/Put(..., deadline_us)` → 各后端逐 block/逐 key 准入检查。`SdkWrapper` 把 deadline 换算成 `steady_clock::time_point`，用于线程池任务启动准入 + 等待上限；`deadline_us == 0` 时由 wrapper 级 `timeout_config`（get/put_timeout_ms）计算兜底 deadline。

---

## 6. 契约的可测试性

因为"复现都不好复现"（线上事故当时无法复现），**测试是防回归的主要手段**（`client/src/internal/sdk/test/`）。覆盖：

1. **运行中超时**：可控 slow fake SDK，验证 wrapper 在 deadline 附近返回，不无界等待。
2. **排队超时被拦下**：占满线程池，让后到分组在队列里等过 deadline，断言它**从未发起 I/O**（fake SDK 记录"是否被调用"）。
3. **deadline 透传**：fake SDK 在 Get 入口观测传入的 deadline_us（必须有值、剩余时间在预算区间）。
4. **返回后无后台访问（hard 级）**：LocalFile 场景，返回后立即用哨兵值覆写 caller buffer，ASAN 下断言无竞争写入。
5. **保序**：交错多 path 不同 payload，断言 `actual_remote_uris[i]` 对应 `remote_uris[i]`。
6. **逐 block / 逐 key 中途停下**：deadline 过期后不再触碰后续 block/key。

---

## 7. 给未来维护者（含 AI）的警告

> 本文件的存在就是为了防止"顺手把契约改回去"。若你正准备做以下任何一件事，先读 §2.1：
>
> - 在 `SdkWrapper` 里加"等 in-flight 完成"的逻辑；
> - 把 deadline 传回 thread_local（恢复 `SdkDeadline`）或 Init 级配置（恢复 `SdkBackendConfig::timeout_config`）；
> - 把 `actual_remote_uris` 简化成 `assign(remote_uris)`；
> - 删掉 LocalFile abort 路径的 `cudaStreamSynchronize`；
> - 把 `hf3fs_wait_for_ios` 的 `abs_timeout` 改回 `nullptr`；
> - 把 mooncake 的 slices 从"逐 key 检查后发起"改成"批量一次性发起"；
> - 给 `LoadKvCaches/SaveKvCaches` 的 `deadline_us` 加默认值（强制显式传，避免遗漏）。
>
> 以上每一条都会重新打开一个已知的静默数据损坏窗口。
