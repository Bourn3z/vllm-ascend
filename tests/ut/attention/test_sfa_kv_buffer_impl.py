"""Double-buffer decode KV offload — drives the REAL `SparseKVOffloadManager`
staging methods (`offload_decode_kv_double_buffer` / `flush_decode_double_buffer`)
through the REAL fused `npu_kv_rmsnorm_rope_cache` kernel, mirroring L6 of
`test_exec_kv_progression.py` (ping-pong + non-blocking side-stream D2H).

This is the acceptance test for
`D2H_BLOCK_DOUBLE_BUFFER_REQUIREMENTS.md` (§7), validating:
  * T1  fused-fill + flush lands correct K/V (== numpy golden)
  * T2  ping-pong flush count matches the batching model (512*48 tok, staging=4096)
  * T3  the FINAL host pool equals DataHelper's numpy golden (rel error <= 1e-2)
  * T4/target: the real manager methods behave as `exec_kv` dispatch expects

Run (NPU + MemFabric Hybrid deployed):
    pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_exec_kv_double_buffer_impl.py
"""

import os
import sys

import pytest
import torch
import torch_npu

import vllm_ascend.ops.fused_moe.fused_moe  # noqa: E402,F401  (break device_op<->fused_moe cycle)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exec_kv_progression import (  # noqa: E402
    STEP,
    TOKENS_PER_STEP,
    NUM_BLOCK,
    NUM_KV_HEADS,
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    EPS,
    DTYPE,
    BLOCK_SIZE,
    DataHelper,
    iter_steps,
    _assert_matches_golden,
    _host_blocks,
)

_DOC = "docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md"

STAGING_NUM_BLOCKS = NUM_BLOCK  # 32 -> capacity 32*128 = 4096


def _npu_or_skip():
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() == 0:
        pytest.skip("Ascend NPU not available")


@pytest.fixture(scope="module")
def memfabric_pool():
    _npu_or_skip()
    if not os.environ.get("MEMFABRIC_HYBRID_EXTEND_LIB_PATH"):
        pytest.skip(
            "MemFabric Hybrid not deployed: MEMFABRIC_HYBRID_EXTEND_LIB_PATH unset. "
            f"See {_DOC} (Decode Dependencies)."
        )
    from memfabric_hybrid import offload

    hb = _host_blocks()
    pool_bytes = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * DTYPE.itemsize * hb * BLOCK_SIZE
    reserve = 2 * pool_bytes + (1 << 22)  # one K/V pool-pair + margin

    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.reserve_size = reserve
    cfg.alloc_size = reserve
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.LOCAL
    rc = offload.initialize(cfg)
    if rc != 0:
        pytest.skip(f"MemFabric offload.initialize failed (rc={rc}); configure per {_DOC}.")
    yield offload
    offload.uninitialize()


def _db_manager(offload, device, num_blocks=None):
    """A real SparseKVOffloadManager configured for `decode_kv_offload_mode`
    == "double_buffer", with real MemFabric host pools and the double-buffer
    per-layer caches allocated via the real `_init_decode_double_buffer`."""
    mgr = object.__new__(SparseKVOffloadManager)
    mgr.tp_rank = 0
    mgr.block_size = BLOCK_SIZE
    mgr.num_layers = 1  # one representative layer (indexer excluded)

    hb = num_blocks if num_blocks is not None else _host_blocks()

    class _Cfg:
        keep_device_kv_cache = False
        decode_kv_offload_mode = "double_buffer"
        decode_kv_staging_num_blocks = STAGING_NUM_BLOCKS
    mgr.sparse_kv_offload_config = _Cfg()

    # Host-side big KV pool (MemFabric DRAM), matching register_kv_caches layout.
    mgr.k_caches_cpu = [offload.empty([hb, BLOCK_SIZE, NUM_KV_HEADS, KV_LORA_RANK], dtype=DTYPE).zero_()]
    mgr.v_caches_cpu = [offload.empty([hb, BLOCK_SIZE, NUM_KV_HEADS, QK_ROPE_HEAD_DIM], dtype=DTYPE).zero_()]

    # Minimal fields `_init_decode_double_buffer` reads for dtype/sizing.
    mgr.topk_buffers_k = [torch.empty(0, dtype=DTYPE, device=device)]

    mgr._init_decode_double_buffer(device, kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM)
    assert mgr._db_capacity == STAGING_NUM_BLOCKS * BLOCK_SIZE
    assert len(mgr._db_k_nope) == 1 and len(mgr._db_k_pe) == 1
    assert len(mgr._db_k_nope[0]) == 2 and len(mgr._db_k_pe[0]) == 2
    return mgr


def test_double_buffer_manager_host_matches_golden(memfabric_pool):
    """T1+T3: drive the real manager staging over 512 steps x 48 tok; the final
    host K/V pool must match DataHelper's numpy golden (rel error <= 1e-2)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    device = kv.device

    mgr = _db_manager(memfabric_pool, device, num_blocks=hb_total + STAGING_NUM_BLOCKS)
    layer_id = 0
    slots = torch.arange(B, dtype=torch.int64, device=device)  # host slot == token

    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP, kv, cos, sin):
        mgr.offload_decode_kv_double_buffer(
            layer_id=layer_id,
            kv_no_split=kv_step,
            norm_weight=weight,
            cos=cos_step,
            sin=sin_step,
            host_slots=slots[sl].clone(),
            num_kv_heads=NUM_KV_HEADS,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            variance_epsilon=EPS,
        )
    mgr.flush_decode_double_buffer(layer_id)
    torch.npu.synchronize()

    rel_k, rel_v = _assert_matches_golden(d, mgr.k_caches_cpu[0], mgr.v_caches_cpu[0], B, "impl double-buffer")
    print(f"[double-buffer-impl] {B} rows staged+flushed; K rel={rel_k:.3e}, V rel={rel_v:.3e}")


def _mid_flush_count(capacity):
    """Mid-loop D2H flushes for the workload (48 tok/step into `capacity`)
    using the same pre-check predicate as the implementation."""
    offset, cnt = 0, 0
    for _ in range(STEP):
        if offset + TOKENS_PER_STEP > capacity:
            cnt += 1
            offset = 0
        offset += TOKENS_PER_STEP
    return cnt


def test_double_buffer_flush_count_matches_model(memfabric_pool):
    """T2: ping-pong flush count equals the batching model.

    Because 48 (tokens/step) does not divide the 4096-token cache, each mid-loop
    flush moves the full 85*48=4080 staged rows, so the number of mid-loop
    flushes is computed with the same pre-check predicate as the implementation.
    """
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    device = kv.device

    mgr = _db_manager(memfabric_pool, device, num_blocks=hb_total + STAGING_NUM_BLOCKS)
    layer_id = 0
    slots = torch.arange(B, dtype=torch.int64, device=device)

    expected_mid = _mid_flush_count(mgr._db_capacity)
    assert expected_mid > 0, "workload should flush at least once (4096 < 24576 tokens)"

    n_flush = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP, kv, cos, sin):
        _, fill_before = mgr._db_active_cache(layer_id)
        if fill_before + TOKENS_PER_STEP > mgr._db_capacity:
            n_flush += 1
        mgr.offload_decode_kv_double_buffer(
            layer_id=layer_id, kv_no_split=kv_step, norm_weight=weight,
            cos=cos_step, sin=sin_step, host_slots=slots[sl].clone(),
            num_kv_heads=NUM_KV_HEADS, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, variance_epsilon=EPS,
        )
    assert n_flush == expected_mid, f"expected {expected_mid} mid-loop flushes, got {n_flush}"
    mgr.flush_decode_double_buffer(layer_id)  # trailing partial -> +1 total
    torch.npu.synchronize()
    assert mgr._db_fill[layer_id] == 0
    print(f"[double-buffer-impl] {STEP} steps x {TOKENS_PER_STEP} tok -> "
          f"{n_flush + 1} total D2H flushes ({n_flush} mid-loop + 1 final)")


def test_double_buffer_runner_hook_flushes_before_forward(memfabric_pool):
    """Runner placement: drive flush via `maybe_flush_decode_double_buffers`
    BEFORE each forward (the model_runner_v1 pre-forward hook), with exec_kv
    staging purely accumulating. Host result must still equal the numpy golden
    with the same flush count (A4/A5, §7 T-runner)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    device = kv.device

    mgr = _db_manager(memfabric_pool, device, num_blocks=hb_total + STAGING_NUM_BLOCKS)
    layer_id = 0
    slots = torch.arange(B, dtype=torch.int64, device=device)

    n_flush = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP, kv, cos, sin):
        # Runner hook, executed BEFORE each forward (model_runner_v1.py).
        n_flush += mgr.maybe_flush_decode_double_buffers(TOKENS_PER_STEP)
        # Forward: each layer stages exactly one step (fused fill into active).
        mgr.offload_decode_kv_double_buffer(
            layer_id=layer_id, kv_no_split=kv_step, norm_weight=weight,
            cos=cos_step, sin=sin_step, host_slots=slots[sl].clone(),
            num_kv_heads=NUM_KV_HEADS, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, variance_epsilon=EPS,
        )
    mgr.flush_decode_double_buffer(layer_id)
    torch.npu.synchronize()

    # Same batching model as the internal-fill flush (6 mid + 1 final = 7 total).
    expected_mid = _mid_flush_count(mgr._db_capacity)
    assert n_flush == expected_mid, f"runner hook {n_flush} != expected {expected_mid}"
    rel_k, rel_v = _assert_matches_golden(d, mgr.k_caches_cpu[0], mgr.v_caches_cpu[0], B, "runner-hook double-buffer")
    print(f"[double-buffer-impl] runner hook: {n_flush} pre-forward flushes + 1 final; "
          f"K rel={rel_k:.3e}, V rel={rel_v:.3e}")


def test_double_buffer_mtp_rollback_drops_rejected(memfabric_pool):
    """MTP adoption rollback: after staging 2 steps (2*48 tokens), a rollback of
    48 (the last step entirely rejected) must shrink `_db_fill` to 48 AND
    invalidate `_db_host_slots[48:96]` (-> -1), so the subsequent flush D2H's
    only slots 0..47 (== first-step golden) and leaves host slots 48..95 at 0
    (never written, i.e. the rejected rows are dropped before D2H)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    device = kv.device

    mgr = _db_manager(memfabric_pool, device, num_blocks=hb_total + STAGING_NUM_BLOCKS)
    layer_id = 0
    slots = torch.arange(B, dtype=torch.int64, device=device)

    it = iter_steps(STEP, TOKENS_PER_STEP, kv, cos, sin)
    for _ in range(2):
        _s, sl, kv_step, cos_step, sin_step = next(it)
        mgr.offload_decode_kv_double_buffer(
            layer_id=layer_id, kv_no_split=kv_step, norm_weight=weight,
            cos=cos_step, sin=sin_step, host_slots=slots[sl].clone(),
            num_kv_heads=NUM_KV_HEADS, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, variance_epsilon=EPS,
        )
    assert mgr._db_fill[layer_id] == 2 * TOKENS_PER_STEP

    # The 2nd step's 48 tokens are all rejected drafts -> drop 48.
    mgr.rollback_decode_double_buffers(TOKENS_PER_STEP)
    assert mgr._db_fill[layer_id] == TOKENS_PER_STEP, \
        f"fill {mgr._db_fill[layer_id]} != {TOKENS_PER_STEP} after rollback"
    active = mgr._db_active[layer_id]
    hs = mgr._db_host_slots[layer_id][active]
    assert bool((hs[:TOKENS_PER_STEP] != -1).all()), "kept host slots must stay valid"
    assert bool((hs[TOKENS_PER_STEP:2 * TOKENS_PER_STEP] == -1).all()), \
        "rejected host slots must be invalidated to -1"

    # Flush: only the kept 48 rows (slots 0..47) reach the host pool.
    mgr.flush_decode_double_buffer(layer_id)
    torch.npu.synchronize()
    ref_k, ref_v = d.standard()
    host_k = mgr.k_caches_cpu[0].view(-1, KV_LORA_RANK)[:TOKENS_PER_STEP].to(torch.float64)
    host_v = mgr.v_caches_cpu[0].view(-1, QK_ROPE_HEAD_DIM)[:TOKENS_PER_STEP].to(torch.float64)
    refk = ref_k.view(-1, KV_LORA_RANK)[:TOKENS_PER_STEP].to(torch.float64)
    refv = ref_v.view(-1, QK_ROPE_HEAD_DIM)[:TOKENS_PER_STEP].to(torch.float64)
    rel_k = (host_k - refk).abs().max().item() / refk.abs().max().item()
    rel_v = (host_v - refv).abs().max().item() / refv.abs().max().item()
    assert rel_k <= 1e-2 and rel_v <= 1e-2, f"kept rows rel K={rel_k:.3e} V={rel_v:.3e}"
    # Rejected slots 48..95 must never have been written.
    k_flat = mgr.k_caches_cpu[0].view(-1, KV_LORA_RANK)
    v_flat = mgr.v_caches_cpu[0].view(-1, QK_ROPE_HEAD_DIM)
    assert int(torch.count_nonzero(k_flat[TOKENS_PER_STEP:2 * TOKENS_PER_STEP])) == 0, \
        "rejected K rows must not be D2H'd"
    assert int(torch.count_nonzero(v_flat[TOKENS_PER_STEP:2 * TOKENS_PER_STEP])) == 0, \
        "rejected V rows must not be D2H'd"
    print(f"[double-buffer-impl] mtp-rollback: fill 96->{TOKENS_PER_STEP}, "
          f"kept-host K rel={rel_k:.3e}, V rel={rel_v:.3e}; rejected slots 48..95 zero")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

