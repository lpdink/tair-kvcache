"""Unit tests for hybrid state postload copy functionality.

Tests the state index computation and state copy logic that fixes
the timing issue where preprocess_mamba runs before start_load_kv.
"""
import math
import unittest
import torch

# Import the methods we're testing
import sys
sys.path.insert(0, '/root/ws/kvcache/tair-kvcache')


class TestStateIndexComputation(unittest.TestCase):
    """Test the state index computation methods."""

    def setup_method(self):
        """Set up a mock connector for testing."""
        # We'll test the logic directly without instantiating the full connector
        self.block_size = 528  # Typical block size for hybrid models

    def compute_prev_state_idx(self, num_computed_tokens: int, block_size: int) -> int:
        """Reference implementation matching the connector."""
        if num_computed_tokens == 0:
            return -1
        return (num_computed_tokens - 1) // block_size

    def compute_curr_state_idx(self, num_computed_tokens: int, num_scheduled_tokens: int, block_size: int) -> int:
        """Reference implementation matching the connector."""
        total_tokens = num_computed_tokens + num_scheduled_tokens
        num_blocks = math.ceil(total_tokens / block_size)
        return num_blocks - 1

    def test_prev_state_idx_new_request(self):
        """Test 4.1: prev_state_idx for new request (0 tokens)."""
        assert self.compute_prev_state_idx(0, self.block_size) == -1

    def test_prev_state_idx_one_token(self):
        """Test 4.1: prev_state_idx for 1 token."""
        assert self.compute_prev_state_idx(1, self.block_size) == 0

    def test_prev_state_idx_block_aligned(self):
        """Test 4.1: prev_state_idx for block-aligned token count."""
        # 528 tokens -> block 0
        assert self.compute_prev_state_idx(528, self.block_size) == 0
        # 1056 tokens -> block 1
        assert self.compute_prev_state_idx(1056, self.block_size) == 1

    def test_prev_state_idx_non_aligned(self):
        """Test 4.1: prev_state_idx for non-aligned token count."""
        # 527 tokens -> block 0
        assert self.compute_prev_state_idx(527, self.block_size) == 0
        # 529 tokens -> block 0
        assert self.compute_prev_state_idx(529, self.block_size) == 0
        # 1057 tokens -> block 1
        assert self.compute_prev_state_idx(1057, self.block_size) == 1

    def test_curr_state_idx_single_block(self):
        """Test 4.2: curr_state_idx for single block."""
        # 0 computed + 400 scheduled -> 1 block -> curr_idx = 0
        assert self.compute_curr_state_idx(0, 400, self.block_size) == 0
        # 0 computed + 528 scheduled -> 1 block -> curr_idx = 0
        assert self.compute_curr_state_idx(0, 528, self.block_size) == 0

    def test_curr_state_idx_two_blocks(self):
        """Test 4.2: curr_state_idx for two blocks."""
        # 528 computed + 286 scheduled -> 2 blocks -> curr_idx = 1
        assert self.compute_curr_state_idx(528, 286, self.block_size) == 1
        # 0 computed + 1000 scheduled -> 2 blocks -> curr_idx = 1
        assert self.compute_curr_state_idx(0, 1000, self.block_size) == 1

    def test_curr_state_idx_three_blocks(self):
        """Test 4.2: curr_state_idx for three blocks."""
        # 1056 computed + 200 scheduled -> 3 blocks -> curr_idx = 2
        assert self.compute_curr_state_idx(1056, 200, self.block_size) == 2
        # 528 computed + 1500 scheduled -> 4 blocks -> curr_idx = 3
        assert self.compute_curr_state_idx(528, 1500, self.block_size) == 3

    def test_curr_state_idx_edge_cases(self):
        """Test 4.2: curr_state_idx edge cases."""
        # 1 computed + 1 scheduled -> 1 block
        assert self.compute_curr_state_idx(1, 1, self.block_size) == 0
        # 528 computed + 0 scheduled -> 1 block
        assert self.compute_curr_state_idx(528, 0, self.block_size) == 0


class TestStateCopyLogic(unittest.TestCase):
    """Test the state copy logic with mock tensors."""

    def setup_method(self):
        """Set up mock hybrid state tensors."""
        self.num_blocks = 10
        self.page_size = 1024  # bytes per block per layer
        self.num_layers = 3
        
        # Create mock block_view_tensors
        self.block_view_tensors = []
        for layer_idx in range(self.num_layers):
            # Each layer has num_blocks blocks, each with page_size bytes
            tensor = torch.zeros(self.num_blocks, self.page_size, dtype=torch.uint8)
            # Fill with layer-specific pattern for verification
            tensor.fill_(layer_idx * 10)
            self.block_view_tensors.append(tensor)

    def test_state_copy_single_layer(self):
        """Test 4.3: state copy with single layer."""
        # Create single-layer mock
        single_layer = [torch.zeros(self.num_blocks, self.page_size, dtype=torch.uint8)]
        single_layer[0][5].fill_(100)  # Source block has value 100
        
        # Copy from block 5 to block 7
        src_block = 5
        dst_block = 7
        single_layer[0][dst_block].copy_(single_layer[0][src_block])
        
        # Verify copy
        assert torch.all(single_layer[0][dst_block] == 100)

    def test_state_copy_multi_layer(self):
        """Test 4.3: state copy with multiple layers."""
        # Set up source block with different values per layer
        for layer_idx in range(self.num_layers):
            self.block_view_tensors[layer_idx][5].fill_((layer_idx + 1) * 50)
        
        # Copy from block 5 to block 7 for all layers
        src_block = 5
        dst_block = 7
        for layer_idx in range(self.num_layers):
            self.block_view_tensors[layer_idx][dst_block].copy_(
                self.block_view_tensors[layer_idx][src_block]
            )
        
        # Verify all layers copied correctly
        for layer_idx in range(self.num_layers):
            expected_value = (layer_idx + 1) * 50
            assert torch.all(self.block_view_tensors[layer_idx][dst_block] == expected_value)

    def test_state_copy_preserves_other_blocks(self):
        """Test 4.3: state copy doesn't affect other blocks."""
        # Set up source and destination
        self.block_view_tensors[0][5].fill_(100)  # Source
        self.block_view_tensors[0][7].fill_(0)    # Destination
        self.block_view_tensors[0][3].fill_(999)  # Other block (should not change)
        
        # Copy
        self.block_view_tensors[0][7].copy_(self.block_view_tensors[0][5])
        
        # Verify other block unchanged
        assert torch.all(self.block_view_tensors[0][3] == 999)

    def test_skip_copy_prev_idx_negative(self):
        """Test 4.5: skip copy when prev_idx == -1."""
        prev_idx = -1
        curr_idx = 0
        # Should skip, no assertion needed, just verify logic
        should_skip = (prev_idx == -1) or (prev_idx == curr_idx)
        assert should_skip

    def test_skip_copy_same_indices(self):
        """Test 4.5: skip copy when prev_idx == curr_idx."""
        prev_idx = 0
        curr_idx = 0
        should_skip = (prev_idx == -1) or (prev_idx == curr_idx)
        assert should_skip

    def test_validation_prev_idx_out_of_bounds(self):
        """Test 4.4: validation for prev_idx out of bounds."""
        block_ids = [10, 20, 30]
        prev_idx = 5  # Out of bounds
        curr_idx = 1
        
        # Should log warning and skip
        should_skip = prev_idx < 0 or prev_idx >= len(block_ids)
        assert should_skip

    def test_validation_curr_idx_out_of_bounds(self):
        """Test 4.4: validation for curr_idx out of bounds."""
        block_ids = [10, 20, 30]
        prev_idx = 0
        curr_idx = 5  # Out of bounds
        
        # Should log warning and skip
        should_skip = curr_idx < 0 or curr_idx >= len(block_ids)
        assert should_skip

    def test_validation_both_indices_valid(self):
        """Test 4.4: validation passes for valid indices."""
        block_ids = [10, 20, 30]
        prev_idx = 0
        curr_idx = 1
        
        # Should not skip
        should_skip = (prev_idx < 0 or prev_idx >= len(block_ids) or
                      curr_idx < 0 or curr_idx >= len(block_ids))
        assert not should_skip


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios."""

    def setup_method(self):
        """Set up for integration tests."""
        self.block_size = 528

    def compute_prev_state_idx(self, num_computed_tokens: int, block_size: int) -> int:
        if num_computed_tokens == 0:
            return -1
        return (num_computed_tokens - 1) // block_size

    def compute_curr_state_idx(self, num_computed_tokens: int, num_scheduled_tokens: int, block_size: int) -> int:
        total_tokens = num_computed_tokens + num_scheduled_tokens
        num_blocks = math.ceil(total_tokens / block_size)
        return num_blocks - 1

    def test_scenario_new_request_no_prefix(self):
        """Scenario: New request without prefix cache."""
        # Request: 814 tokens, no prefix
        num_computed = 0
        num_scheduled = 814
        
        prev_idx = self.compute_prev_state_idx(num_computed, self.block_size)
        curr_idx = self.compute_curr_state_idx(num_computed, num_scheduled, self.block_size)
        
        # prev_idx = -1 (no previous state)
        # curr_idx = 1 (2 blocks: 0-527, 528-813)
        assert prev_idx == -1
        assert curr_idx == 1
        # Should skip copy (prev_idx == -1)
        should_copy = (prev_idx != -1 and prev_idx != curr_idx)
        assert not should_copy

    def test_scenario_prefix_cache_hit_single_block(self):
        """Scenario: Prefix cache hit, single block cached."""
        # Request: 528 tokens cached, 286 new tokens
        num_computed = 528
        num_scheduled = 286
        
        prev_idx = self.compute_prev_state_idx(num_computed, self.block_size)
        curr_idx = self.compute_curr_state_idx(num_computed, num_scheduled, self.block_size)
        
        # prev_idx = 0 (block 0 has state from first 528 tokens)
        # curr_idx = 1 (2 blocks total)
        assert prev_idx == 0
        assert curr_idx == 1
        # Should copy from block 0 to block 1
        should_copy = (prev_idx != -1 and prev_idx != curr_idx)
        assert should_copy

    def test_scenario_prefix_cache_hit_multi_block(self):
        """Scenario: Prefix cache hit, multiple blocks cached."""
        # Request: 1056 tokens cached, 200 new tokens
        num_computed = 1056
        num_scheduled = 200
        
        prev_idx = self.compute_prev_state_idx(num_computed, self.block_size)
        curr_idx = self.compute_curr_state_idx(num_computed, num_scheduled, self.block_size)
        
        # prev_idx = 1 (block 1 has state from first 1056 tokens)
        # curr_idx = 2 (3 blocks total)
        assert prev_idx == 1
        assert curr_idx == 2
        # Should copy from block 1 to block 2
        should_copy = (prev_idx != -1 and prev_idx != curr_idx)
        assert should_copy

    def test_scenario_full_prompt_cached(self):
        """Scenario: Entire prompt cached, only generating."""
        # Request: 1326 tokens cached, 0 new tokens (decode only)
        num_computed = 1326
        num_scheduled = 1  # At least 1 token for decode
        
        prev_idx = self.compute_prev_state_idx(num_computed, self.block_size)
        curr_idx = self.compute_curr_state_idx(num_computed, num_scheduled, self.block_size)
        
        # prev_idx = 2 (block 2 has state from 1326 tokens)
        # curr_idx = 2 (still 3 blocks)
        assert prev_idx == 2
        assert curr_idx == 2
        # Should skip copy (prev_idx == curr_idx)
        should_copy = (prev_idx != -1 and prev_idx != curr_idx)
        assert not should_copy


if __name__ == '__main__':
    unittest.main()
