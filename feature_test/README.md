# Per-Group Revisit Interval Bucket Configuration — 集成测试报告

## 概述

本测试验证 per-instance-group 重访间隔直方图分桶配置功能的端到端可用性。

- **代码分支**: `feature_revisit_interval_stats_on_instance`
- **测试分支**: `feature_revisit_interval_stats_on_instance_test`（本分支）
- **测试日期**: 2026-06-17

## 快速复现

```bash
cd tair-kvcache
bash feature_test/run_test.sh
```

或手动执行：

```bash
# 1. 启动环境
docker compose -f feature_test/docker-compose-test.yaml up -d

# 2. 等待 KVCM 编译启动（首次约 2 分钟）
# 验证: curl http://localhost:6492/metrics | head

# 3. 运行测试
python3 feature_test/test_per_group_buckets.py

# 4. Grafana Dashboard
# http://localhost:12111 (admin/admin)

# 5. 停止
docker compose -f feature_test/docker-compose-test.yaml down
```

## 测试设置

### 环境

| 组件 | 说明 |
|------|------|
| KVCM | 编译运行于 dev 容器，使用 `kvcm_server_config.conf` |
| Prometheus | 5s 间隔 scrape `:6492/metrics` |
| Grafana | `:12111`，含 per-instance 变量选择器 |

### Server 级默认配置

```conf
kvcm.metrics.revisit_interval_buckets=1,5,30,60,120,300,600,1800,3600
```

### Instance Group 配置

| Group | `revisit_interval_buckets` | 预期 boundaries |
|-------|---------------------------|-----------------|
| `group-fast` | `"0.5,1,2,5"` | `[0.5, 1, 2, 5]` |
| `group-slow` | `"30,60,300,3600"` | `[30, 60, 300, 3600]` |
| `group-default` | _(未设置)_ | server default `[1, 5, 30, 60, 120, 300, 600, 1800, 3600]` |

## 测试结果

### 全部 14 项测试通过 ✅

```
Total: 14  Passed: 14  Failed: 0
```

### Phase 0: Storage Backend Setup
- ✅ 创建 dummy storage backend

### Phase 1: Create Instance Groups
- ✅ 通过 admin API `CreateInstanceGroup` 创建 3 个 group，各自配置不同 boundaries

### Phase 2: Register Instances
- ✅ 注册 3 个 instance 分别归属不同 group

### Phase 3: Generate Traffic & Verify Observations
- ✅ `inst-fast-1`: 57 observations（sleep 1s, 5 workers × 4 reads）
- ✅ `inst-slow-1`: 42 observations（sleep 10s, 3 workers × 3 reads）
- ✅ `inst-default-1`: 66 observations（sleep 5s, 5 workers × 4 reads）

### Phase 4: Per-Instance Bucket Boundaries
- ✅ `inst-fast-1` boundaries = `[0.5, 1.0, 2.0, 5.0]` — 匹配 group-fast 配置
- ✅ `inst-slow-1` boundaries = `[30.0, 60.0, 300.0, 3600.0]` — 匹配 group-slow 配置
- ✅ `inst-default-1` boundaries = `[1.0, 5.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]` — 匹配 server default
- ✅ Per-group 隔离验证：fast ≠ slow boundaries

### Phase 5: Immutability Test
- ✅ 更新 `group-fast` buckets 为 `"10,20,30"` 后，`inst-fast-1` 的 boundaries **仍为** `[0.5, 1.0, 2.0, 5.0]`
- ✅ 新注册 `inst-fast-2` 使用更新后的 boundaries `[10.0, 20.0, 30.0]`

### Phase 6: Invalid Config Handling
- ✅ 无效配置 `"5,1,30"`（非升序）被 API 接受（非阻塞），KVCM 日志输出 WARN
- ✅ 新注册 `inst-default-2` 自动 fallback 到 server default boundaries

## 验证方法

测试通过 `/metrics` 端点解析 histogram 的 `le` label 集合来验证 boundaries：

```python
def get_bucket_boundaries(instance_id):
    resp = requests.get(f"{ADMIN_URL}/metrics")
    le_values = set()
    for line in resp.text.splitlines():
        if 'revisit_interval_seconds_bucket' in line and f'instance_id="{instance_id}"' in line:
            le = re.search(r'le="([^"]+)"', line).group(1)
            if le != "+Inf":
                le_values.add(float(le))
    return sorted(le_values)
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `docker-compose-test.yaml` | 一键拉起 KVCM + Prometheus + Grafana |
| `kvcm_server_config.conf` | KVCM 配置（端口 + server 级默认 buckets） |
| `prometheus.yml` | Prometheus scrape 配置 |
| `grafana-datasource.yml` | Grafana 数据源自动配置 |
| `grafana-dashboard-provider.yml` | Grafana dashboard 加载配置 |
| `grafana-dashboard-per-group.json` | Grafana dashboard（per-instance 变量选择器） |
| `test_per_group_buckets.py` | 主测试脚本（6 个 phase，14 个验证点） |
| `run_test.sh` | 一键执行 wrapper |

## Grafana Dashboard

Dashboard 提供 3 个面板：
1. **Bucket Boundaries per Instance** — 表格展示每个 instance 的 bucket boundary 值
2. **Observation Count per Instance** — 各 instance 累计观测数
3. **Bucket Distribution (%)** — 选中 instance 的各桶占比（bar gauge）

访问: `http://<machine-ip>:12111` (admin/admin)
