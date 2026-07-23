"""A verification wrapper around the production KVCM vLLM connector.

This connector is injected via vLLM's ``kv_connector_module_path`` and subclasses
the production ``TairKvCacheConnector`` without modifying it. Its purpose is to
independently capture the KV data that lives in vLLM's paged KV cache so that the
test driver can verify the connector's save/load translation layer.

Why this catches translation bugs
---------------------------------
The connector translates between three block spaces:

    KVCM manager block idx  ->  global token idx  ->  vLLM physical block id
        (step 1, connector-only)   (step 2/3, shared with vLLM)

``generate_blocks_idx`` performs step 1 (``manager_block_idx * manager_block_size
+ i``) and then steps 2/3 (``local_block_ids[token // bs] * bs + token % bs``).
A bug in step 1 makes *save* gather from the wrong physical slots and *load*
scatter to the wrong physical slots. Because save and load share the same
translation, a transport-level round trip still "matches" (the bug is symmetric).

To break the symmetry we capture the KV data using ONLY steps 2/3 -- the mapping
vLLM itself uses (``slot = block_table[pos // bs] * bs + pos % bs``, where
``local_block_ids`` IS the block table). This reference is independent of the
connector's step-1 logic, so a step-1 bug makes the captured data diverge from
what the connector saved/loaded.

Capture points
--------------
* Reference (save path): in ``wait_for_save`` we read the KV of the saved token
  range straight out of the paged cache (the forward pass has completed and the
  slots are not modified by the parent's async gather).
* Loaded (load path): loads are async (``load_kv_async=True``), so the load step
  has no forward pass and the worker does not yet know the request's token ids.
  We therefore record the load's block info in ``start_load_kv`` (which is always
  called) and emit the capture in a later ``wait_for_save`` once the request's
  token ids have arrived on the worker. The loaded KV persists in the paged cache
  (its blocks are allocated to the request and prefix caching is disabled).

Captures are written to ``$KVCM_E2E_CAPTURE_DIR`` as ``.pt`` files named
``{ref|loaded}_{tp_rank}_{token_hash}.pt`` so the out-of-process test driver can
match reference vs loaded by content (the captured token ids).
"""

import hashlib
import os
import threading
import typing

import torch

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector

CAPTURE_DIR_ENV = "KVCM_E2E_CAPTURE_DIR"


class VerifyingConnector(TairKvCacheConnector):
    """Production connector + independent KV capture for e2e verification."""

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)

        self._capture_dir = os.environ.get(CAPTURE_DIR_ENV, "")
        if self._capture_dir:
            os.makedirs(self._capture_dir, exist_ok=True)

        # Track completion of async load scatters. The parent's load task already
        # CPU-synchronizes its own scatter (copy_done_event.synchronize()) before
        # reporting the task result, so a plain threading.Event set from the done
        # callback is sufficient to know the scatter is globally visible.
        self._load_done_events: dict[str, list[threading.Event]] = {}
        self._load_events_lock = threading.Lock()

        # Loads are async and their step has no forward pass, so the worker does
        # not yet have the request's token ids. Record the load's block info here
        # and emit the capture once the token ids arrive (see wait_for_save).
        # req_id -> (manager_block_idxes, local_block_ids)
        self._pending_loaded: dict[str, tuple[list, list]] = {}

        orig_factory = self._data_transfer.create_load_done_callback

        def tracking_factory(req_id, tp_rank, epoch, local_block_ids):
            orig_cb = orig_factory(req_id, tp_rank, epoch, local_block_ids)
            evt = threading.Event()
            with self._load_events_lock:
                self._load_done_events.setdefault(req_id, []).append(evt)

            def cb(task_results):
                try:
                    orig_cb(task_results)
                finally:
                    evt.set()

            return cb

        self._data_transfer.create_load_done_callback = tracking_factory
        logger.warning(
            "VerifyingConnector enabled, capture_dir=%s tp_rank=%s",
            self._capture_dir, self._tp_rank,
        )

    # ------------------------------------------------------------------ #
    # Load hook: record pending loaded captures
    # ------------------------------------------------------------------ #
    def start_load_kv(self, forward_context, **kwargs) -> None:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        load_reqs = [
            (lr.req_id, list(lr.manager_block_idxes), list(lr.local_block_ids))
            for lr in meta.to_load_requests
            if lr.local_block_ids
        ]

        super().start_load_kv(forward_context, **kwargs)

        if getattr(self, "_capture_dir", "") and load_reqs:
            for req_id, mbis, lbis in load_reqs:
                self._pending_loaded[req_id] = (mbis, lbis)
            logger.warning(
                "VerifyingConnector recorded %d pending loaded capture(s)",
                len(load_reqs),
            )

    # ------------------------------------------------------------------ #
    # Save hook: reference captures + emit pending loaded captures
    # ------------------------------------------------------------------ #
    def wait_for_save(self):
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        if getattr(self, "_capture_dir", "") and getattr(self, "_kv_caches", None):
            try:
                self._capture_refs(meta)
                self._capture_pending_loaded()
            except Exception as e:  # never break inference for a capture error
                logger.warning("VerifyingConnector capture failed: %s", e, exc_info=True)

        super().wait_for_save()

    def _capture_refs(self, meta: TairKvCacheConnectorMetadata):
        if not meta.to_save_requests:
            return
        # Make all forward-pass KV writes visible before reading the paged cache.
        torch.cuda.synchronize()
        for save_req in meta.to_save_requests:
            req = self._alive_requests.get(save_req.req_id)
            if req is None:
                continue
            self._capture_range(
                kind="ref",
                token_ids=req.token_ids,
                local_block_ids=req.local_block_ids,
                manager_block_idxes=save_req.manager_block_idxes,
            )

    def _capture_pending_loaded(self):
        if not self._pending_loaded:
            return
        done = []
        for req_id, (mbis, lbis) in self._pending_loaded.items():
            req = self._alive_requests.get(req_id)
            if req is None:
                # token ids have not arrived on this worker yet; wait for a
                # later step in which the request is scheduled.
                continue
            with self._load_events_lock:
                evts = list(self._load_done_events.get(req_id, []))
            for evt in evts:
                evt.wait(timeout=120)
            torch.cuda.synchronize()
            self._capture_range(
                kind="loaded",
                token_ids=req.token_ids,
                local_block_ids=lbis,
                manager_block_idxes=mbis,
            )
            done.append(req_id)
        for req_id in done:
            del self._pending_loaded[req_id]

    # ------------------------------------------------------------------ #
    # Capture helper
    # ------------------------------------------------------------------ #
    def _capture_range(self, kind, token_ids, local_block_ids, manager_block_idxes):
        if not manager_block_idxes or not local_block_ids:
            return
        # Capture one record per manager block. Saves are batched incrementally
        # while loads arrive all-at-once, so per-block records let the driver
        # match reference vs loaded captures by each block's token content.
        for b in manager_block_idxes:
            self._capture_block(kind, token_ids, local_block_ids, b)

    def _capture_block(self, kind, token_ids, local_block_ids, manager_block_idx):
        mbs = self._manager_block_size
        lbs = self._local_block_size

        # Global token indices covered by this manager block (step 1 done with the
        # *metadata's* block index -- this is just an index range, not the
        # connector's physical mapping).
        token_positions = list(
            range(manager_block_idx * mbs, (manager_block_idx + 1) * mbs)
        )
        if token_positions[-1] >= len(token_ids):
            # Partial trailing block; only capture the tokens that exist.
            token_positions = [p for p in token_positions if p < len(token_ids)]
        if not token_positions:
            return

        # Independent physical mapping (steps 2/3, identical to vLLM's slot kernel):
        #   slot = block_table[pos // bs] * bs + (pos % bs)
        slots = []
        for pos in token_positions:
            blk = local_block_ids[pos // lbs]
            slots.append(blk * lbs + (pos % lbs))
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self._device)

        captured_token_ids = [token_ids[p] for p in token_positions]

        kv_by_layer = {}
        for layer_name, kv_cache in self._kv_caches.items():
            # kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            assert not self._use_mla, "MLA layout not supported by e2e capture"
            num_kv = kv_cache.shape[0]
            flat = kv_cache.reshape(num_kv, -1, kv_cache.shape[3], kv_cache.shape[4])
            gathered = flat[:, slot_tensor, :, :]          # [2, n_tok, heads, head_size]
            gathered = gathered.permute(1, 0, 2, 3).contiguous()  # [n_tok, 2, heads, hs]
            kv_by_layer[layer_name] = gathered.cpu()

        token_hash = hashlib.sha256(
            torch.tensor(captured_token_ids, dtype=torch.int64).numpy().tobytes()
        ).hexdigest()[:16]

        path = os.path.join(
            self._capture_dir, f"{kind}_tp{self._tp_rank}_{token_hash}.pt"
        )
        torch.save({"token_ids": captured_token_ids, "kv": kv_by_layer}, path)
        logger.warning(
            "VerifyingConnector captured %s block=%d tokens=%d..%d tp=%s -> %s",
            kind, manager_block_idx, token_positions[0], token_positions[-1],
            self._tp_rank, path,
        )
