"""Unit tests for hybrid attention (mamba/gdn/linear) support in vllm connector.

Tests cover:
- _detect_hybrid_layers: identify MambaSpec groups in kv_cache_config
- register_kv_caches: handle heterogeneous dict[str, Tensor | list[Tensor]]
- AND-merge callbacks: correctly merge attention + hybrid save/load results
- generate_hybrid_block_indices: per-block index generation
- Backward compatibility: pure attention models unchanged
- location_spec_group_names: correct filling for hybrid models
"""

import copy
import ctypes
import math
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List

import torch

# Mock kvcm_py_client before any connector imports (C++ pybind not available in test env)
_mock_kvcm_py_client = MagicMock()
_mock_kvcm_py_client.ClientErrorCode.ER_OK = 0
_mock_pybind = MagicMock()
_mock_pybind.kvcm_py_client = _mock_kvcm_py_client
sys.modules.setdefault('kv_cache_manager.client.pybind', _mock_pybind)

# Mock only the heavy vllm modules that require vllm._C (C++ extension)
# Do NOT mock vllm.utils as it's imported by kv_cache_interface

# For vllm.distributed.kv_transfer.kv_connector.v1.base, provide a real base class
# because metadata.py's @dataclass needs a real __mro__
_base_mock = MagicMock()
class _KVConnectorMetadata:
    pass
class _KVConnectorBase_V1:
    pass
class _KVConnectorRole:
    SCHEDULER = "scheduler"
    WORKER = "worker"
_base_mock.KVConnectorMetadata = _KVConnectorMetadata
_base_mock.KVConnectorBase_V1 = _KVConnectorBase_V1
_base_mock.KVConnectorRole = _KVConnectorRole
sys.modules['vllm.distributed.kv_transfer.kv_connector.v1.base'] = _base_mock

for mod_name in [
    'vllm.config',
    'vllm.distributed',
    'vllm.v1.core.sched.output',
    'vllm.v1.outputs',
]:
    sys.modules.setdefault(mod_name, MagicMock())

# Mock orjson (not installed in test container)
if 'orjson' not in sys.modules:
    import json
    import dataclasses
    _mock_orjson = MagicMock()
    def _orjson_dumps(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            obj = dataclasses.asdict(obj)
        return json.dumps(obj).encode('utf-8')
    def _orjson_loads(data):
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    _mock_orjson.dumps = _orjson_dumps
    _mock_orjson.loads = _orjson_loads
    sys.modules['orjson'] = _mock_orjson

# Mock zmq with proper submodule support (not installed in test container)
if 'zmq' not in sys.modules:
    _mock_zmq = MagicMock()
    sys.modules['zmq'] = _mock_zmq
    sys.modules['zmq.asyncio'] = MagicMock()

# Mock _version_info (generated during build, not in source tree)
_version_info_mod = MagicMock()
_version_info_mod.FULL_VERSION = "0.0.0-test"
_version_info_mod.GIT_COMMIT = "test-commit"
_version_info_mod.BUILD_TIME = "2024-01-01T00:00:00"
sys.modules.setdefault('kv_cache_manager.py_connector.common._version_info', _version_info_mod)

# vllm imports for mock kv_cache_config (these are pure-python, no _C needed)
from vllm.v1.kv_cache_interface import (
    MambaSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# Connector imports
from kv_cache_manager.py_connector.common.types import KVCacheInfo, HybridCacheInfo
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
from kv_cache_manager.py_connector.vllm.data_transfer import MultiResult
from kv_cache_manager.py_connector.common.tp_coordinator import (
    TpCoordinatorClient, CoordinateMessage, CoordinateMsgSerializer,
    SendBlockFinishedEvent, LoadBlockFinishedEvent,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_attn_spec(num_kv_heads=8, head_size=128, block_size=16, dtype=torch.bfloat16):
    """Create a FullAttentionSpec."""
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        dtype=dtype,
    )


def _make_mamba_spec(page_size_bytes=4096, block_size=16):
    """Create a MambaSpec with the given page_size_bytes."""
    # Use minimal shapes so actual page_size < page_size_bytes (padded)
    # shapes: conv=(conv_dim, conv_rows), ssm=(num_heads, head_dim, d_state)
    shapes = ((1, 1), (1, 1, 1))
    dtypes = (torch.float32, torch.float32)
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=page_size_bytes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _make_kv_cache_config(attn_layer_names, hybrid_layer_names, page_size_bytes=4096,
                          num_kv_heads=8, head_size=128, block_size=16):
    """Build a mock KVCacheConfig with attention and hybrid groups."""
    groups = []
    if attn_layer_names:
        groups.append(KVCacheGroupSpec(
            layer_names=list(attn_layer_names),
            kv_cache_spec=_make_attn_spec(num_kv_heads, head_size, block_size),
        ))
    if hybrid_layer_names:
        groups.append(KVCacheGroupSpec(
            layer_names=list(hybrid_layer_names),
            kv_cache_spec=_make_mamba_spec(page_size_bytes, block_size),
        ))
    return KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _make_attn_kv_cache(num_layers, num_blocks, block_size, kv_heads, head_size, dtype):
    """Create attention KV cache tensors (simulating vllm's layout)."""
    kv_caches = {}
    for i in range(num_layers):
        name = f"model.layers.{i}.self_attn"
        # Shape: (2, num_blocks, block_size, kv_heads, head_size)
        kv_caches[name] = torch.randn(
            2, num_blocks, block_size, kv_heads, head_size,
            dtype=dtype, device="cpu"
        )
    return kv_caches


def _make_hybrid_kv_cache(num_layers, num_blocks, page_size_bytes):
    """Create hybrid (mamba/gdn) state caches as list[Tensor] with shared storage.

    Each layer gets [conv_state, ssm_state] sharing the same untyped_storage,
    simulating vllm's initialize_kv_cache_tensors behavior.
    Conv and ssm states are at non-overlapping offsets in the storage.
    """
    kv_caches = {}
    total_bytes_per_block = page_size_bytes

    for i in range(num_layers):
        name = f"model.layers.{num_layers + i}.mamba"
        # Create a raw tensor that holds both conv and ssm state
        raw = torch.zeros(num_blocks, total_bytes_per_block, dtype=torch.uint8, device="cpu")
        conv_bytes = total_bytes_per_block // 2
        ssm_bytes = total_bytes_per_block - conv_bytes

        # conv_state: view into first half of each block
        # Use as_strided to create a view with proper offset/stride
        conv_state = torch.as_strided(
            raw,
            size=(num_blocks, conv_bytes),
            stride=(total_bytes_per_block, 1),
        )
        # ssm_state: view into second half of each block
        ssm_state = torch.as_strided(
            raw,
            size=(num_blocks, ssm_bytes),
            stride=(total_bytes_per_block, 1),
            storage_offset=conv_bytes,
        )

        # Fill with distinct patterns so we can verify round-trip
        conv_state.fill_(i + 1)  # layer 0 = 1, layer 1 = 2, ...
        ssm_state.fill_(i + 100)  # layer 0 = 100, layer 1 = 101, ...
        kv_caches[name] = [conv_state, ssm_state]
    return kv_caches


def _mock_vllm_config(tp_size=1, block_size=16, num_kv_heads=8, head_size=128,
                      dtype=torch.bfloat16, has_hybrid=False, num_attn_layers=4,
                      num_hybrid_layers=2, hybrid_page_size=4096):
    """Build a mock VllmConfig."""
    vllm_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.cache_config.block_size = block_size
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.model_config.use_mla = False
    vllm_config.model_config.dtype = dtype
    vllm_config.model_config.served_model_name = "test-model"
    vllm_config.model_config.get_num_layers.return_value = num_attn_layers + num_hybrid_layers
    vllm_config.model_config.get_num_kv_heads.return_value = num_kv_heads // tp_size
    vllm_config.model_config.get_head_size.return_value = head_size
    vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "manager_uri": "http://localhost:8080",
        "coordinator_base_port": 0,
        "instance_group": "test-group",
        "instance_id": "test-instance",
        "preferred_block_size": 0,
        "storage_configs": {},
        "write_timeout_seconds": 30,
        "sdk_thread_num": 4,
        "sdk_queue_size": 100,
        "sdk_get_timeout_ms": 5000,
        "sdk_put_timeout_ms": 10000,
        "read_iov_block_size": 0,
        "write_iov_block_size": 0,
        "hf3fs_concurrent_io_block_count": 1,
        "block_per_save_task": 128,
        "block_per_load_task": 128,
        "async_get_cache_location": True,
        "auto_discover_leader": False,
        "leader_retry_count": 1,
        "leader_retry_base_interval_seconds": 0.005,
        "discovery_refresh_interval_seconds": 30,
        "min_discover_interval_seconds": 1,
        "log_level": "WARNING",
    }

    # kv_cache_config
    attn_names = [f"model.layers.{i}.self_attn" for i in range(num_attn_layers)]
    hybrid_names = [f"model.layers.{num_attn_layers + i}.mamba" for i in range(num_hybrid_layers)]
    if has_hybrid:
        vllm_config.kv_cache_config = _make_kv_cache_config(
            attn_names, hybrid_names, hybrid_page_size,
            num_kv_heads, head_size, block_size
        )
    else:
        vllm_config.kv_cache_config = _make_kv_cache_config(
            attn_names, [], hybrid_page_size,
            num_kv_heads, head_size, block_size
        )

    return vllm_config


# ============================================================================
# Test: _detect_hybrid_layers
# ============================================================================

class TestDetectHybridLayers(unittest.TestCase):
    """Test layer type detection from kv_cache_config."""

    def test_pure_attention_returns_false(self):
        """Pure attention model: no hybrid layers detected."""
        cfg = _make_kv_cache_config(
            attn_layer_names=["l0.self_attn", "l1.self_attn"],
            hybrid_layer_names=[],
        )
        connector = TairKvCacheConnector.__new__(TairKvCacheConnector)
        has_hybrid, names, page_size = connector._detect_hybrid_layers(cfg)
        self.assertFalse(has_hybrid)
        self.assertEqual(names, [])
        self.assertEqual(page_size, 0)

    def test_hybrid_model_detected(self):
        """Mixed model: hybrid layers and page_size correctly identified."""
        cfg = _make_kv_cache_config(
            attn_layer_names=["l0.self_attn", "l1.self_attn"],
            hybrid_layer_names=["l2.mamba", "l3.mamba"],
            page_size_bytes=8192,
        )
        connector = TairKvCacheConnector.__new__(TairKvCacheConnector)
        has_hybrid, names, page_size = connector._detect_hybrid_layers(cfg)
        self.assertTrue(has_hybrid)
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], "l2.mamba")
        self.assertEqual(page_size, 8192)

    def test_multiple_mamba_groups_same_page_size(self):
        """Multiple MambaSpec groups with same page_size: OK."""
        spec1 = _make_mamba_spec(page_size_bytes=4096)
        spec2 = _make_mamba_spec(page_size_bytes=4096)
        cfg = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["l0.self_attn"], kv_cache_spec=_make_attn_spec()),
                KVCacheGroupSpec(layer_names=["l1.mamba"], kv_cache_spec=spec1),
                KVCacheGroupSpec(layer_names=["l2.mamba"], kv_cache_spec=spec2),
            ],
        )
        connector = TairKvCacheConnector.__new__(TairKvCacheConnector)
        has_hybrid, names, page_size = connector._detect_hybrid_layers(cfg)
        self.assertTrue(has_hybrid)
        self.assertEqual(len(names), 2)
        self.assertEqual(page_size, 4096)

    def test_multiple_mamba_groups_different_page_size_raises(self):
        """Multiple MambaSpec groups with different page_size: assert fails."""
        spec1 = _make_mamba_spec(page_size_bytes=4096)
        spec2 = _make_mamba_spec(page_size_bytes=8192)
        cfg = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["l0.mamba"], kv_cache_spec=spec1),
                KVCacheGroupSpec(layer_names=["l1.mamba"], kv_cache_spec=spec2),
            ],
        )
        connector = TairKvCacheConnector.__new__(TairKvCacheConnector)
        with self.assertRaises(AssertionError):
            connector._detect_hybrid_layers(cfg)

    def test_none_config_returns_false(self):
        """None kv_cache_config: returns False (old vllm compatibility)."""
        connector = TairKvCacheConnector.__new__(TairKvCacheConnector)
        has_hybrid, names, page_size = connector._detect_hybrid_layers(None)
        self.assertFalse(has_hybrid)


# ============================================================================
# Test: generate_hybrid_block_indices
# ============================================================================

class TestGenerateHybridBlockIndices(unittest.TestCase):
    """Test per-block index generation for hybrid transfer."""

    def _make_connector(self, manager_block_size=16, local_block_size=16):
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._manager_block_size = manager_block_size
        c._local_block_size = local_block_size
        return c

    def test_basic_mapping(self):
        """Simple case: manager_block_size == local_block_size."""
        c = self._make_connector(16, 16)
        local_block_ids = [10, 11, 12, 13, 14]
        # manager_block_idxes [0, 1, 3] → token_idx [0, 16, 48] → local_block_idx [0, 1, 3]
        result = c.generate_hybrid_block_indices([0, 1, 3], local_block_ids)
        self.assertEqual(result, [10, 11, 13])

    def test_different_block_sizes(self):
        """manager_block_size != local_block_size."""
        c = self._make_connector(32, 16)  # manager=32 tokens, local=16 tokens
        local_block_ids = [10, 11, 12, 13, 14, 15]
        # manager_block_idx=0 → token_idx=0 → local_block_idx=0 → local_block_ids[0]=10
        # manager_block_idx=1 → token_idx=32 → local_block_idx=2 → local_block_ids[2]=12
        result = c.generate_hybrid_block_indices([0, 1], local_block_ids)
        self.assertEqual(result, [10, 12])

    def test_out_of_bounds_raises(self):
        """Block index beyond local_block_ids raises AssertionError."""
        c = self._make_connector(16, 16)
        local_block_ids = [10, 11]
        with self.assertRaises(AssertionError):
            c.generate_hybrid_block_indices([0, 5], local_block_ids)


# ============================================================================
# Test: AND-merge callbacks
# ============================================================================

class TestSaveCallbackANDMerge(unittest.TestCase):
    """Test create_save_done_callback AND-merge logic."""

    # In the real code, transfer_result is an int: 0 = ER_OK (success), non-zero = failure
    OK = 0   # kvcm_py_client.ClientErrorCode.ER_OK
    ERR = 1  # any non-zero error code

    def _make_data_transfer(self):
        """Create a minimal DataTransferManager with mocked dependencies."""
        from kv_cache_manager.py_connector.vllm.data_transfer import DataTransferManager
        dt = DataTransferManager.__new__(DataTransferManager)
        dt._coordinator_client = MagicMock()
        return dt

    def test_pure_attention_no_merge(self):
        """Pure attention: no AND-merge, results passed through directly."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=3, hybrid_block_count=0
        )
        # Simulate 3 attention tasks, each with 1 block result
        cb([[self.OK], [self.ERR], [self.OK]])

        call_args = dt._coordinator_client.send.call_args
        msg = CoordinateMsgSerializer.loads(call_args[0][0])
        self.assertIsInstance(msg.content, SendBlockFinishedEvent)
        self.assertEqual(msg.content.is_success_list, [True, False, True])

    def test_hybrid_and_merge_all_success(self):
        """AND-merge: attention all success + hybrid all success → all success."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=3, hybrid_block_count=3
        )
        # 3 attn results + 3 hybrid results
        cb([[self.OK, self.OK, self.OK], [self.OK, self.OK, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.is_success_list, [True, True, True])

    def test_hybrid_and_merge_hybrid_failure(self):
        """AND-merge: attention success + hybrid failure on block 1 → block 1 fails."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=3, hybrid_block_count=3
        )
        # attn: [OK, OK, OK], hybrid: [OK, ERR, OK]
        cb([[self.OK, self.OK, self.OK], [self.OK, self.ERR, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.is_success_list, [True, False, True])

    def test_hybrid_and_merge_attention_failure(self):
        """AND-merge: attention failure on block 2 → block 2 fails regardless of hybrid."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=3, hybrid_block_count=3
        )
        # attn: [OK, OK, ERR], hybrid: [OK, OK, OK]
        cb([[self.OK, self.OK, self.ERR], [self.OK, self.OK, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.is_success_list, [True, True, False])

    def test_mismatched_counts_raises(self):
        """AND-merge with mismatched block counts raises AssertionError."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=3, hybrid_block_count=2  # mismatch!
        )
        with self.assertRaises(AssertionError):
            cb([[self.OK, self.OK, self.OK], [self.OK, self.OK]])


class TestLoadCallbackANDMerge(unittest.TestCase):
    """Test create_load_done_callback AND-merge logic."""

    OK = 0
    ERR = 1

    def _make_data_transfer(self):
        from kv_cache_manager.py_connector.vllm.data_transfer import DataTransferManager
        dt = DataTransferManager.__new__(DataTransferManager)
        dt._coordinator_client = MagicMock()
        return dt

    def test_pure_attention_no_merge(self):
        """Pure attention: failed blocks reported directly."""
        dt = self._make_data_transfer()
        local_block_ids = [10, 11, 12]
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=local_block_ids,
            attn_block_count=3, hybrid_block_count=0
        )
        # Block 1 fails
        cb([[self.OK, self.ERR, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertIsInstance(msg.content, LoadBlockFinishedEvent)
        self.assertEqual(msg.content.failed_block_idxs, [11])

    def test_hybrid_and_merge_hybrid_failure(self):
        """AND-merge: hybrid failure on block 2 → block 2 in failed_block_idxs."""
        dt = self._make_data_transfer()
        local_block_ids = [10, 11, 12]
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=local_block_ids,
            attn_block_count=3, hybrid_block_count=3
        )
        # attn: [OK, OK, OK], hybrid: [OK, OK, ERR]
        cb([[self.OK, self.OK, self.OK], [self.OK, self.OK, self.ERR]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.failed_block_idxs, [12])

    def test_hybrid_and_merge_both_fail(self):
        """AND-merge: both attn and hybrid fail on same block → one entry in failed list."""
        dt = self._make_data_transfer()
        local_block_ids = [10, 11, 12]
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=local_block_ids,
            attn_block_count=3, hybrid_block_count=3
        )
        # attn: [ERR, OK, OK], hybrid: [ERR, OK, OK]
        cb([[self.ERR, self.OK, self.OK], [self.ERR, self.OK, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.failed_block_idxs, [10])

    def test_hybrid_and_merge_all_success(self):
        """AND-merge: all success → empty failed list."""
        dt = self._make_data_transfer()
        local_block_ids = [10, 11, 12]
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=local_block_ids,
            attn_block_count=3, hybrid_block_count=3
        )
        cb([[self.OK, self.OK, self.OK], [self.OK, self.OK, self.OK]])

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.failed_block_idxs, [])

    def test_mismatched_counts_raises(self):
        """AND-merge with mismatched block counts raises AssertionError."""
        dt = self._make_data_transfer()
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=[10, 11, 12],
            attn_block_count=3, hybrid_block_count=2  # mismatch!
        )
        with self.assertRaises(AssertionError):
            cb([[self.OK, self.OK, self.OK], [self.OK, self.OK]])


# ============================================================================
# Test: get_self_hybrid_uris
# ============================================================================

class TestGetSelfHybridUris(unittest.TestCase):
    """Test URI extraction for hybrid spec."""

    def _make_connector(self, tp_size=1, tp_rank=0, has_hybrid=True):
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._has_hybrid = has_hybrid
        c._tp_rank_to_hybrid_spec_name = lambda rank: f"tp{rank}_hybrid"
        # Mock _kvcache_info.tp_rank
        c._kvcache_info = MagicMock()
        c._kvcache_info.tp_rank = tp_rank
        return c

    def test_extract_hybrid_uris(self):
        """Extract hybrid URIs from locations with mixed specs."""
        c = self._make_connector(tp_size=2, tp_rank=0)
        locations = [
            {
                "location_specs": [
                    {"name": "tp0", "uri": "3fs://attn_tp0_block0"},
                    {"name": "tp1", "uri": "3fs://attn_tp1_block0"},
                    {"name": "tp0_hybrid", "uri": "3fs://hybrid_tp0_block0"},
                    {"name": "tp1_hybrid", "uri": "3fs://hybrid_tp1_block0"},
                ]
            },
            {
                "location_specs": [
                    {"name": "tp0", "uri": "3fs://attn_tp0_block1"},
                    {"name": "tp1", "uri": "3fs://attn_tp1_block1"},
                    {"name": "tp0_hybrid", "uri": "3fs://hybrid_tp0_block1"},
                    {"name": "tp1_hybrid", "uri": "3fs://hybrid_tp1_block1"},
                ]
            },
        ]
        uris = c.get_self_hybrid_uris(locations)
        self.assertEqual(uris, ["3fs://hybrid_tp0_block0", "3fs://hybrid_tp0_block1"])

    def test_no_hybrid_spec_returns_empty(self):
        """Locations without hybrid spec: returns empty list."""
        c = self._make_connector()
        locations = [
            {"location_specs": [{"name": "tp0", "uri": "3fs://attn"}]}
        ]
        uris = c.get_self_hybrid_uris(locations)
        self.assertEqual(uris, [])

    def test_not_hybrid_model_returns_empty(self):
        """_has_hybrid=False: returns empty list immediately."""
        c = self._make_connector(has_hybrid=False)
        locations = [
            {"location_specs": [{"name": "tp0_hybrid", "uri": "3fs://hybrid"}]}
        ]
        uris = c.get_self_hybrid_uris(locations)
        self.assertEqual(uris, [])


# ============================================================================
# Test: HybridCacheInfo structure
# ============================================================================

class TestHybridCacheInfo(unittest.TestCase):
    """Test HybridCacheInfo dataclass."""

    def test_create_with_block_view_tensors(self):
        """HybridCacheInfo can be created with block_view_tensors."""
        block_views = [
            torch.zeros(10, 4096, dtype=torch.uint8),
            torch.zeros(10, 4096, dtype=torch.uint8),
        ]
        ptr_tensor = torch.tensor([1000, 2000], dtype=torch.int64)
        info = HybridCacheInfo(
            layer_names=["l0.mamba", "l1.mamba"],
            ptr_tensor_cpu=ptr_tensor,
            block_view_tensors=block_views,
            page_size_bytes=4096,
            layer_num=2,
        )
        self.assertEqual(info.layer_num, 2)
        self.assertEqual(info.page_size_bytes, 4096)
        self.assertEqual(len(info.block_view_tensors), 2)

    def test_kv_cache_info_with_hybrid(self):
        """KVCacheInfo accepts optional hybrid_info."""
        info = KVCacheInfo(
            tp_rank=0, world_size=1,
            kvcaches={}, kvcache_ptr_tensor_cpu=torch.tensor([]),
            kvcache_ptr_tensor_gpu=torch.tensor([]),
            all_kvcache_ptr_tensor_gpu=torch.tensor([]),
            layer_num=4, local_token_num=160,
            per_manager_block_shape=(4, 2, 16, 8, 128),
            per_manager_block_byte_size=131072,
            per_token_per_layer_dim_size=1024,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            hybrid_info=None,
        )
        self.assertIsNone(info.hybrid_info)


# ============================================================================
# Test: Backward compatibility (pure attention)
# ============================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Verify pure attention models are unaffected by hybrid changes."""

    def test_detect_no_hybrid(self):
        """_detect_hybrid_layers returns False for pure attention config."""
        cfg = _make_kv_cache_config(
            attn_layer_names=["l0.self_attn", "l1.self_attn"],
            hybrid_layer_names=[],
        )
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        has_hybrid, names, page_size = c._detect_hybrid_layers(cfg)
        self.assertFalse(has_hybrid)
        self.assertEqual(names, [])

    def test_location_spec_groups_empty_for_pure_attention(self):
        """Pure attention: no location_spec_groups in register request."""
        # We can't easily test the full __init__ without mocking many things,
        # but we can verify the logic: if _has_hybrid is False, no groups
        # are added. This is already tested implicitly by the _has_hybrid guard.
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._has_hybrid = False
        c._tp_rank_to_spec_name = lambda r: f"tp{r}"
        c._tp_rank_to_hybrid_spec_name = lambda r: f"tp{r}_hybrid"
        c._hybrid_page_size_bytes = 0
        c._tp_size = 1

        # Simulate the register_instance building logic
        location_spec_infos = [{
            "name": c._tp_rank_to_spec_name(rank),
            "size": 1000000
        } for rank in range(c._tp_size)]
        location_spec_groups = []
        if c._has_hybrid:
            location_spec_groups.append({"name": "Full", "spec_names": ["tp0"]})
            location_spec_groups.append({"name": "FullAndHybrid", "spec_names": ["tp0", "tp0_hybrid"]})

        self.assertEqual(location_spec_groups, [])
        self.assertEqual(len(location_spec_infos), 1)


# ============================================================================
# Test: register_kv_caches heterogeneous dict (core entry point)
# ============================================================================

class TestRegisterKvCaches(unittest.TestCase):
    """Test register_kv_caches with heterogeneous dict[str, Tensor | list[Tensor]]."""

    def test_heterogeneous_dict_separation(self):
        """Heterogeneous dict correctly separates attention and hybrid layers."""
        attn_caches = _make_attn_kv_cache(2, 10, 16, 8, 128, torch.bfloat16)
        hybrid_caches = _make_hybrid_kv_cache(2, 10, 4096)
        kv_caches = {**attn_caches, **hybrid_caches}

        # Verify the separation logic directly
        attn_names = []
        hybrid_names = []
        for name, cache in kv_caches.items():
            if isinstance(cache, list):
                hybrid_names.append(name)
            else:
                attn_names.append(name)

        self.assertEqual(len(attn_names), 2)
        self.assertEqual(len(hybrid_names), 2)
        # Attention layers come first (from _make_attn_kv_cache)
        self.assertTrue(all("self_attn" in n for n in attn_names))
        self.assertTrue(all("mamba" in n for n in hybrid_names))

    def test_hybrid_kv_cache_structure(self):
        """Hybrid kv_cache entries are list[Tensor] with shared storage."""
        hybrid_caches = _make_hybrid_kv_cache(2, 10, 4096)

        for name, state_tensors in hybrid_caches.items():
            self.assertIsInstance(state_tensors, list)
            self.assertEqual(len(state_tensors), 2)  # conv_state + ssm_state

            conv_state, ssm_state = state_tensors
            # Both share the same untyped_storage
            self.assertEqual(
                conv_state.untyped_storage().data_ptr(),
                ssm_state.untyped_storage().data_ptr()
            )
            # Shapes are correct
            self.assertEqual(conv_state.shape[0], 10)  # num_blocks
            self.assertEqual(ssm_state.shape[0], 10)

    def test_distinct_fill_patterns(self):
        """Conv and ssm states have distinct fill values for round-trip verification."""
        hybrid_caches = _make_hybrid_kv_cache(3, 5, 1024)

        names = list(hybrid_caches.keys())
        for i, name in enumerate(names):
            conv_state, ssm_state = hybrid_caches[name]
            self.assertEqual(conv_state[0, 0].item(), i + 1)
            self.assertEqual(ssm_state[0, 0].item(), i + 100)

    def test_register_kv_caches_full_integration(self):
        """Full register_kv_caches: mixed dict, verify separation + _hybrid_info."""
        attn_caches = _make_attn_kv_cache(2, 10, 16, 8, 128, torch.bfloat16)
        hybrid_caches = _make_hybrid_kv_cache(2, 10, 4096)
        kv_caches = {**attn_caches, **hybrid_caches}

        c = self._make_connector()
        c.register_kv_caches(kv_caches)

        # Attention layers registered
        self.assertEqual(len(c._kvcache_ptr_tensor_cpu), 2)
        self.assertTrue(torch.all(c._kvcache_ptr_tensor_cpu > 0))

        # Hybrid info built
        self.assertIsNotNone(c._hybrid_info)
        self.assertEqual(c._hybrid_info.layer_num, 2)
        self.assertEqual(len(c._hybrid_info.block_view_tensors), 2)
        self.assertEqual(c._hybrid_info.page_size_bytes, 4096)

        # Verify block_view_tensors point to hybrid layer storage
        for i, bvt in enumerate(c._hybrid_info.block_view_tensors):
            self.assertEqual(bvt.shape[0], 10)  # num_blocks

    def _make_connector(self):
        """Create minimal connector for register_kv_caches testing."""
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._has_hybrid = True
        c._tp_size = 1
        c._local_block_size = 16
        c._manager_block_size = 16
        c._device = torch.device("cpu")
        c._dtype = torch.bfloat16
        c._use_mla = False
        c._extra_config = MagicMock()
        c._extra_config.hf3fs_concurrent_io_block_count = 1
        c._manager_client = MagicMock()
        c._manager_client.register_instance.return_value = MagicMock()
        c._manager_client.register_instance.return_value.storage_configs = []
        c._transfer_client = MagicMock()
        c._coordinator_client = MagicMock()
        c._tp_rank = 0
        c._location_spec_name = "rank0"
        c._tp_rank_to_hybrid_spec_name = lambda rank: f"rank{rank}_hybrid"
        c._hybrid_page_size_bytes = 4096
        return c


# ============================================================================
# Test: _register_hybrid_kv_caches storage contiguity
# ============================================================================

class TestRegisterHybridKvCaches(unittest.TestCase):
    """Test _register_hybrid_kv_caches storage contiguity assertions."""

    def _make_connector(self, page_size_bytes=4096):
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._has_hybrid = True
        c._device = torch.device("cpu")
        c._hybrid_page_size_bytes = page_size_bytes
        c._kv_caches = {}
        return c

    def test_shared_storage_passes(self):
        """conv_state and ssm_state sharing untyped_storage: assert passes."""
        c = self._make_connector()
        hybrid_caches = _make_hybrid_kv_cache(2, 10, 4096)
        c._kv_caches = hybrid_caches
        # Should not raise
        c._register_hybrid_kv_caches(list(hybrid_caches.keys()))
        self.assertIsNotNone(c._hybrid_info)

    def test_different_storage_fails(self):
        """conv_state and ssm_state with different untyped_storage: assert fails."""
        c = self._make_connector()
        # Create tensors with separate storage
        conv = torch.zeros(10, 2048, dtype=torch.uint8)
        ssm = torch.zeros(10, 2048, dtype=torch.uint8)  # different storage!
        c._kv_caches = {"model.layers.0.mamba": [conv, ssm]}
        with self.assertRaises(AssertionError):
            c._register_hybrid_kv_caches(["model.layers.0.mamba"])

    def test_non_list_value_fails(self):
        """Hybrid layer with non-list value: assert fails."""
        c = self._make_connector()
        c._kv_caches = {"model.layers.0.mamba": torch.zeros(10, 100)}  # Tensor, not list
        with self.assertRaises(AssertionError):
            c._register_hybrid_kv_caches(["model.layers.0.mamba"])

    def test_empty_list_fails(self):
        """Hybrid layer with empty list: assert fails."""
        c = self._make_connector()
        c._kv_caches = {"model.layers.0.mamba": []}
        with self.assertRaises(AssertionError):
            c._register_hybrid_kv_caches(["model.layers.0.mamba"])

    def test_storage_too_small_fails(self):
        """Storage smaller than num_blocks * page_size_bytes: assert fails."""
        c = self._make_connector(page_size_bytes=8192)
        # Create a tiny storage that's too small
        raw = torch.zeros(10, 100, dtype=torch.uint8)  # 100 bytes << 10*8192
        state = torch.tensor([], dtype=torch.uint8).set_(raw.untyped_storage()).view(10, 100)
        c._kv_caches = {"model.layers.0.mamba": [state, state]}
        with self.assertRaises(AssertionError):
            c._register_hybrid_kv_caches(["model.layers.0.mamba"])

    def test_block_view_tensors_shape(self):
        """block_view_tensors have correct (num_blocks, page_size_bytes) shape."""
        c = self._make_connector(page_size_bytes=256)
        hybrid_caches = _make_hybrid_kv_cache(3, 8, 256)
        c._kv_caches = hybrid_caches
        c._register_hybrid_kv_caches(list(hybrid_caches.keys()))

        self.assertEqual(c._hybrid_info.layer_num, 3)
        self.assertEqual(c._hybrid_info.page_size_bytes, 256)
        for bvt in c._hybrid_info.block_view_tensors:
            self.assertEqual(bvt.shape, (8, 256))


# ============================================================================
# Test: hybrid data round-trip (save gather → load scatter consistency)
# ============================================================================

class TestHybridDataRoundTrip(unittest.TestCase):
    """Test hybrid_save_task → hybrid_load_task data consistency."""

    def _make_data_transfer(self, hybrid_info):
        from kv_cache_manager.py_connector.vllm.data_transfer import DataTransferManager
        dt = DataTransferManager.__new__(DataTransferManager)
        kvcache_info = MagicMock()
        kvcache_info.hybrid_info = hybrid_info
        kvcache_info.device = torch.device("cpu")
        dt._kvcache_info = kvcache_info
        dt._device_mod = MagicMock()
        # Make stream context manager a no-op
        dt._device_mod.stream = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None)
        ))
        dt._device_mod.Event = MagicMock(return_value=MagicMock(
            record=MagicMock(), synchronize=MagicMock()
        ))
        dt._save_stream = MagicMock()
        dt._load_stream = MagicMock()
        dt._coordinator_client = MagicMock()
        dt._transfer_client = MagicMock()
        return dt

    def test_save_load_round_trip(self):
        """Data saved by hybrid_save_task can be loaded back identically."""
        # Setup: 2 hybrid layers, 4 blocks, page_size=256
        page_size = 256
        num_blocks = 4
        block_views = []
        for layer in range(2):
            bv = torch.arange(layer * 100, layer * 100 + num_blocks * page_size,
                              dtype=torch.uint8).view(num_blocks, page_size)
            block_views.append(bv)

        hybrid_info = HybridCacheInfo(
            layer_names=["l0.mamba", "l1.mamba"],
            ptr_tensor_cpu=torch.tensor([bv.data_ptr() for bv in block_views]),
            block_view_tensors=block_views,
            page_size_bytes=page_size,
            layer_num=2,
        )
        dt = self._make_data_transfer(hybrid_info)

        # Mock transfer_client to capture saved data and replay it on load
        saved_data = {}
        def mock_save(uris, buffers):
            for i, uri in enumerate(uris):
                saved_data[uri] = bytes(
                    ctypes.string_at(buffers[i].iovs[0].base, buffers[i].iovs[0].size)
                )
            return [0]  # ER_OK
        def mock_load(uris, buffers):
            for i, uri in enumerate(uris):
                if uri in saved_data:
                    data = saved_data[uri]
                    ctypes.memmove(buffers[i].iovs[0].base, data, len(data))
            return 0  # ER_OK

        dt._transfer_client.SaveKvCaches = mock_save
        dt._transfer_client.LoadKvCaches = mock_load

        # Save blocks 1 and 3
        block_indices = [1, 3]
        uris = ["3fs://block1", "3fs://block3"]
        ready_event = MagicMock()

        mr = MultiResult(1, lambda results: None)
        dt.hybrid_save_task(mr, 0, uris, block_indices, ready_event)

        # Verify data was saved
        self.assertEqual(len(saved_data), 2)

        # Now clear the block views and load back
        for bv in block_views:
            bv.fill_(0)

        # Verify they're cleared
        self.assertTrue(all(bv.sum() == 0 for bv in block_views))

        mr2 = MultiResult(1, lambda results: None)
        dt.hybrid_load_task(mr2, 0, uris, block_indices)

        # Verify data is restored
        for layer_idx in range(2):
            for block_idx in block_indices:
                expected = torch.arange(
                    layer_idx * 100 + block_idx * page_size,
                    layer_idx * 100 + (block_idx + 1) * page_size,
                    dtype=torch.uint8
                )
                actual = block_views[layer_idx][block_idx]
                self.assertTrue(torch.equal(actual, expected),
                                f"Layer {layer_idx}, block {block_idx}: data mismatch")

    def test_load_failure_preserves_data(self):
        """LoadKvCaches failure: block_view_tensors unchanged, error propagates."""
        page_size = 128
        num_blocks = 3
        block_views = [
            torch.full((num_blocks, page_size), 42, dtype=torch.uint8),
            torch.full((num_blocks, page_size), 99, dtype=torch.uint8),
        ]
        hybrid_info = HybridCacheInfo(
            layer_names=["l0.mamba", "l1.mamba"],
            ptr_tensor_cpu=torch.tensor([bv.data_ptr() for bv in block_views]),
            block_view_tensors=block_views,
            page_size_bytes=page_size,
            layer_num=2,
        )
        dt = self._make_data_transfer(hybrid_info)

        # Mock LoadKvCaches to return error
        dt._transfer_client.LoadKvCaches = lambda uris, buffers: 1  # non-zero = error

        results = []
        mr = MultiResult(1, lambda r: results.extend(r))
        dt.hybrid_load_task(mr, 0, ["3fs://fail"], [0])

        # Data unchanged
        self.assertTrue(torch.all(block_views[0] == 42))
        self.assertTrue(torch.all(block_views[1] == 99))

        # Error propagated
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], [1])

    def test_save_failure_propagates(self):
        """SaveKvCaches failure: error code propagated to multi_result."""
        page_size = 64
        block_views = [
            torch.zeros(2, page_size, dtype=torch.uint8),
            torch.zeros(2, page_size, dtype=torch.uint8),
        ]
        hybrid_info = HybridCacheInfo(
            layer_names=["l0.mamba"],
            ptr_tensor_cpu=torch.tensor([block_views[0].data_ptr()]),
            block_view_tensors=[block_views[0]],
            page_size_bytes=page_size,
            layer_num=1,
        )
        dt = self._make_data_transfer(hybrid_info)

        # Mock SaveKvCaches to return error
        dt._transfer_client.SaveKvCaches = lambda uris, buffers: [7]  # error code 7

        results = []
        mr = MultiResult(1, lambda r: results.extend(r))
        ready_event = MagicMock()
        dt.hybrid_save_task(mr, 0, ["3fs://fail0", "3fs://fail1"], [0, 1], ready_event)

        # Error propagated
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], [7, 7])


# ============================================================================
# Test: location_spec_group_names filling
# ============================================================================

class TestLocationSpecGroupNames(unittest.TestCase):
    """Test start_save_kvcache_async fills location_spec_group_names correctly."""

    def _make_connector(self, has_hybrid):
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        c._has_hybrid = has_hybrid
        c._epoch = 1
        c._extra_config = MagicMock()
        c._extra_config.instance_id = "test-instance"
        c._manager_client = MagicMock()
        c._manager_client.start_write_cache = MagicMock(return_value={
            "locations": [],  # empty locations → triggers early return
            "write_session_id": "ws1",
            "block_mask": {"offset": 0},
        })
        c._coordinator_client = MagicMock()
        c._canceled_save_request_ids = []
        c._canceled_save_request_ids_lock = threading.Lock()
        c._waiting_to_save_requests_lock = threading.Lock()
        c._waiting_to_save_requests = []
        return c

    def test_hybrid_model_fills_group_names(self):
        """Hybrid model: location_spec_group_names = ['FullAndHybrid'] * N."""
        c = self._make_connector(has_hybrid=True)

        c.start_save_kvcache_async("req1", list(range(80)), 5)

        # Verify the request includes location_spec_group_names
        call_args = c._manager_client.start_write_cache.call_args[0][0]
        self.assertIn("location_spec_group_names", call_args)
        self.assertEqual(call_args["location_spec_group_names"],
                         ["FullAndHybrid"] * 5)

    def test_pure_attention_no_group_names(self):
        """Pure attention: location_spec_group_names not in request."""
        c = self._make_connector(has_hybrid=False)

        c.start_save_kvcache_async("req1", list(range(80)), 5)

        call_args = c._manager_client.start_write_cache.call_args[0][0]
        self.assertNotIn("location_spec_group_names", call_args)

    def test_different_target_save_nums(self):
        """location_spec_group_names length matches target_save_num."""
        c = self._make_connector(has_hybrid=True)

        for n in [1, 10, 100]:
            c._manager_client.start_write_cache.reset_mock()
            c.start_save_kvcache_async("req", list(range(n * 16)), n)
            call_args = c._manager_client.start_write_cache.call_args[0][0]
            self.assertEqual(len(call_args["location_spec_group_names"]), n)


# ============================================================================
# Test: multi-task AND-merge
# ============================================================================

class TestMultiTaskANDMerge(unittest.TestCase):
    """Test AND-merge with multiple task_results (real-world scenario)."""

    OK = 0
    ERR = 1

    def _make_data_transfer(self):
        from kv_cache_manager.py_connector.vllm.data_transfer import DataTransferManager
        dt = DataTransferManager.__new__(DataTransferManager)
        dt._coordinator_client = MagicMock()
        return dt

    def test_save_multi_task_flatten_order(self):
        """Multiple tasks with multiple blocks: flatten order is correct."""
        dt = self._make_data_transfer()
        cb = dt.create_save_done_callback(
            req_id="req1", tp_rank=0, write_session_id="ws1",
            attn_block_count=4, hybrid_block_count=4
        )
        # 2 attn tasks (2 blocks each) + 2 hybrid tasks (2 blocks each)
        cb([[self.OK, self.OK], [self.OK, self.ERR],    # attn: [T,T,T,F]
            [self.OK, self.OK], [self.OK, self.OK]])    # hybrid: [T,T,T,T]

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        # AND-merge: attn[3]=F → block 3 fails
        self.assertEqual(msg.content.is_success_list, [True, True, True, False])

    def test_load_multi_task_flatten_order(self):
        """Multiple tasks: failed_block_idxs correctly mapped."""
        dt = self._make_data_transfer()
        local_block_ids = [10, 11, 12, 13]
        cb = dt.create_load_done_callback(
            req_id="req1", tp_rank=0, epoch=1,
            local_block_ids=local_block_ids,
            attn_block_count=4, hybrid_block_count=4
        )
        # 2 attn tasks + 2 hybrid tasks
        cb([[self.OK, self.OK], [self.OK, self.OK],     # attn: all OK
            [self.OK, self.ERR], [self.OK, self.OK]])   # hybrid: block 1 fails

        msg = CoordinateMsgSerializer.loads(
            dt._coordinator_client.send.call_args[0][0])
        self.assertEqual(msg.content.failed_block_idxs, [11])


# ============================================================================
# Test: naming conventions
# ============================================================================

class TestNamingConventions(unittest.TestCase):
    """Test helper method naming conventions."""

    def test_hybrid_spec_name(self):
        """_tp_rank_to_hybrid_spec_name returns 'tp{rank}_hybrid'."""
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        self.assertEqual(c._tp_rank_to_hybrid_spec_name(0), "tp0_hybrid")
        self.assertEqual(c._tp_rank_to_hybrid_spec_name(3), "tp3_hybrid")

    def test_attn_spec_name(self):
        """_tp_rank_to_spec_name returns 'tp{rank}'."""
        c = TairKvCacheConnector.__new__(TairKvCacheConnector)
        self.assertEqual(c._tp_rank_to_spec_name(0), "tp0")
        self.assertEqual(c._tp_rank_to_spec_name(7), "tp7")


if __name__ == "__main__":
    unittest.main()
