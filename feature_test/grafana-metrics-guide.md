# KVCacheManager Metrics - Grafana 配置参考

## 概述

KVCacheManager 通过 Prometheus 暴露 Block 级别的缓存命中率指标。本文档说明指标的定义、标签、以及如何在 Grafana 中配置可视化面板。

## 指标定义

### 1. `kvcm_manager_get_cache_location_query_block_counter`

- **类型**: Counter
- **描述**: GetCacheLocation 请求中查询的 Block 总数（累计）
- **语义**: 每次 GetCacheLocation 调用时，累加请求的 Block 数量

### 2. `kvcm_manager_get_cache_location_hit_block_counter`

- **类型**: Counter
- **描述**: GetCacheLocation 请求中命中的 Block 总数（累计）
- **语义**: 每次 GetCacheLocation 调用时，累加命中的 Block 数量

## 标签 (Labels)

两个指标都包含以下标签：

| 标签名 | 描述 | 示例值 |
|--------|------|--------|
| `api_name` | API 名称 | `GetCacheLocation` |
| `instance_group` | 实例组名称 | `default` |
| `instance_id` | 实例 ID（推理引擎标识） | `qwen-72b-fp8` |

**重要**: `instance_id` 标签允许按推理引擎实例分别查看指标，无需使用 `sum()` 聚合。

## PromQL 查询示例

### 1. 单个实例的查询速率

```promql
rate(kvcm_manager_get_cache_location_query_block_counter{instance_id="qwen-72b-fp8"}[1m])
```

### 2. 单个实例的命中速率

```promql
rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id="qwen-72b-fp8"}[1m])
```

### 3. 单个实例的命中率（实时）

```promql
rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id="qwen-72b-fp8"}[1m]) 
  / 
rate(kvcm_manager_get_cache_location_query_block_counter{instance_id="qwen-72b-fp8"}[1m])
```

### 4. 多个实例的命中率对比

```promql
rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id=~"qwen.*|glm.*"}[1m]) 
  / 
rate(kvcm_manager_get_cache_location_query_block_counter{instance_id=~"qwen.*|glm.*"}[1m])
```

### 5. 实例组级别的聚合命中率

```promql
sum by (instance_group) (rate(kvcm_manager_get_cache_location_hit_block_counter[5m])) 
  / 
sum by (instance_group) (rate(kvcm_manager_get_cache_location_query_block_counter[5m]))
```

## Grafana Dashboard 配置

### 方式一：Per-Instance Dashboard（推荐）

使用 `grafana-dashboard-per-instance.json`，该 Dashboard 包含：

1. **Query Block Rate (per instance)** - 每个实例的查询速率时序图
2. **Hit Block Rate (per instance)** - 每个实例的命中速率时序图
3. **Hit Rate (per instance)** - 每个实例的命中率时序图（0-100%）
4. **Hit Rate Gauge (per instance)** - 每个实例的命中率仪表盘（5 分钟平均）

**特点**:
- 顶部有 `instance_id` 下拉选择器，支持多选
- 所有面板按 `instance_id` 分别展示，不使用 `sum()` 聚合
- 可以直观对比不同实例的缓存命中情况

### 方式二：全局聚合 Dashboard

使用 `grafana-dashboard.json`，该 Dashboard 使用 `sum()` 聚合所有实例：

```promql
sum(rate(kvcm_manager_get_cache_location_hit_block_counter[1m])) 
  / 
sum(rate(kvcm_manager_get_cache_location_query_block_counter[1m]))
```

**适用场景**: 只需要查看全局平均命中率，不关心单个实例的差异。

## 变量配置

在 Grafana Dashboard 中添加变量以支持动态选择实例：

1. 进入 Dashboard Settings → Variables
2. 添加新变量：
   - **Name**: `instance_id`
   - **Type**: Query
   - **Datasource**: Prometheus
   - **Query**: `label_values(kvcm_manager_get_cache_location_query_block_counter, instance_id)`
   - **Multi-value**: 启用
   - **Include All option**: 启用

## 面板配置示例

### 时序图 (Time Series)

**标题**: Hit Rate (per instance)

**PromQL**:
```promql
rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id=~"$instance_id"}[1m]) 
  / 
rate(kvcm_manager_get_cache_location_query_block_counter{instance_id=~"$instance_id"}[1m])
```

**配置**:
- Unit: `Percent (0.0-1.0)`
- Min: 0
- Max: 1
- Thresholds: 
  - Red: < 0.5
  - Yellow: 0.5 - 0.8
  - Green: > 0.8

### 仪表盘 (Gauge)

**标题**: Current Hit Rate (5m avg)

**PromQL**:
```promql
sum by (instance_id) (rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id=~"$instance_id"}[5m])) 
  / 
sum by (instance_id) (rate(kvcm_manager_get_cache_location_query_block_counter{instance_id=~"$instance_id"}[5m]))
```

**配置**:
- Unit: `Percent (0.0-1.0)`
- Min: 0
- Max: 1
- Reduce options: `Last*` (最新值)

## 常见问题

### Q: 为什么使用 `rate()` 而不是直接看 counter 值？

A: Counter 是累计值，会持续增长。使用 `rate()` 可以计算每秒的增长率，反映实时的查询/命中速率。

### Q: `rate()` 的时间窗口怎么选？

A: 
- `[1m]` - 实时性强，但波动大
- `[5m]` - 平衡实时性和平滑度（推荐）
- `[15m]` - 更平滑，适合长期趋势观察

### Q: 如何区分不同模型的缓存命中率？

A: 使用 `instance_id` 标签过滤。例如：
```promql
rate(kvcm_manager_get_cache_location_hit_block_counter{instance_id="qwen-72b-fp8"}[5m]) 
  / 
rate(kvcm_manager_get_cache_location_query_block_counter{instance_id="qwen-72b-fp8"}[5m])
```

### Q: 如何查看某个实例组的整体命中率？

A: 使用 `sum by (instance_group)` 聚合：
```promql
sum by (instance_group) (rate(kvcm_manager_get_cache_location_hit_block_counter[5m])) 
  / 
sum by (instance_group) (rate(kvcm_manager_get_cache_location_query_block_counter[5m]))
```

## 测试验证

使用 `test_hit_rate_metrics.py` 生成测试流量：

```bash
# 启动 KVCM + Prometheus + Grafana
docker compose -f docker-compose-test.yaml up -d

# 生成流量（模拟 5 个模型实例）
for i in $(seq 1 40); do python3 test_hit_rate_metrics.py; sleep 3; done

# 访问 Grafana
# http://<machine-ip>:12111 (admin/admin)
```

测试脚本会随机选择以下实例 ID：
- `qwen-72b-fp8`
- `glm-4-fp8`
- `kimi-k2-fp8`
- `deepseek-v3-fp8`
- `minimax-01-fp8`

每个实例的命中率在 80-100% 之间随机波动，可以在 Grafana 中观察到不同实例的独立曲线。

## 文件说明

| 文件 | 用途 |
|------|------|
| `grafana-dashboard.json` | 全局聚合 Dashboard（使用 `sum()`） |
| `grafana-dashboard-per-instance.json` | Per-Instance Dashboard（按 `instance_id` 分别展示） |
| `grafana-datasource.yml` | Prometheus 数据源配置 |
| `grafana-dashboard-provider.yml` | Dashboard 自动加载配置 |
| `test_hit_rate_metrics.py` | 测试流量生成脚本 |
| `docker-compose-test.yaml` | 测试环境 Docker Compose 配置 |

## 参考

- [Prometheus 指标文档](../docs/prometheus-zh_CN.md)
- [PR #XXX: Add GetCacheLocation block-level hit rate counters](https://github.com/alibaba/tair-kvcache/pull/XXX)
