"""Sparse-KV decode offload, refactored as a LAYERED progression.

Each test builds on the previous one, adding exactly one enhancement, so the
refactor reads top-to-bottom like a story:

    Layer 1  test_1_compute_kv_only_matches_independent_rms
             The most basic piece: `_compute_kv_only` computes k_nope/k_pe
             (verified against an independent fused-RMSNorm). No offload.

    Layer 2  test_2_baseline_exec_kv_offloads_via_memfabric
             Full baseline exec_kv: `_compute_kv_only` + `offload_new_kv`
             (MemFabric `offload.sparse_copy` D2H) -> real big host KV pool.

    Layer 3  test_3_block_accumulate_reduces_offloads
             Keep an on-device num_block=128 buffer; each step only ACCUMULATES
             the new k/v (no per-step D2H), then ONE final D2H. Fewer offloads,
             host pool unchanged vs baseline.

    Layer 4  test_4_fused_op_saves_block
             Replace `_compute_kv_only` with the fused kernel
             `npu_kv_rmsnorm_rope_cache`, which COMPUTES and SAVES k_pe/k_nope
             straight into its own on-device block caches (k_rope_block=k_pe,
             k_nope_block=k_nope), in place at the mapped slots.

    Layer 5  test_5_offload_block_replaces_offload_new_kv
             Replace the per-token `offload_new_kv` with `offload_block`: ONE
             bulk copy_ of all valid tokens (flat host rows) per flush. Same
             DataHelper 512x48 structure and numpy-golden compare as L3/L4.

    Layer 6  test_6_double_buffered_offload_guards_precision
             Double-buffer variant of L3/L4/L5: two NUM_BLOCK=32 on-device
             caches, per-step room check, non-blocking side-stream offload that
             overlaps the other cache's fused fill (flat token-row D2H, no block
             alignment). Host result must equal DataHelper's numpy golden.

Every offload test drives the REAL classes/methods and a REAL MemFabric D2H
into a REAL big host KV pool; `offload_block` lives in this file as the
block-save helper under test (per the D2H_BLOCK_LIFECYCLE design).

Runtime requirement / skip
--------------------------
MemFabric Hybrid must be deployed (install MemFabric 1.2, source set_env.sh,
set `MEMFABRIC_HYBRID_EXTEND_LIB_PATH`) -- see
docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md
(§1 Decode Dependencies). Without it the offload tests skip; Layer 1 and
Layer 4 (compute/fused only) run with no MemFabric dependency.

Run:
    pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_exec_kv_progression.py
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch_npu

# Break the device_op <-> fused_moe circular import, then load the REAL impl &
# manager classes (see sfa_kv_offload.py / sparse_kv_offload_manager.py).
import vllm_ascend.ops.fused_moe.fused_moe  # noqa: E402,F401
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

_DOC = "docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md"

# --------------------------------------------------------------------------- #
# Model / impl config (real D-node decode)
# --------------------------------------------------------------------------- #
NUM_KV_HEADS = 1          # N = 1 (MQA / PA decode) -- manager asserts == 1
KV_LORA_RANK = 512        # D_ckv (k_nope)
QK_ROPE_HEAD_DIM = 64     # D_kpe (k_pe); npu_interleave_rope requires == 64
EPS = 1e-5
DTYPE = torch.bfloat16
BLOCK_SIZE = 128          # host pool + on-device block buffer size
NUM_BLOCK = 32            # on-device block buffer: 32 * 128 = 4096 tokens before offload

# Spec-decode forward semantics (as in the latest double-buffer scheme): one
# forward step = a FULL batch of num_tokens_per_batch tokens per request
# (5 draft + 1 main), across batch_size requests -> STEP_TOKENS tokens/step.
NUM_TOKENS_PER_BATCH = 6
BATCH_SIZE = 8
STEP_TOKENS = BATCH_SIZE * NUM_TOKENS_PER_BATCH   # 48 tokens per forward step

# Layer-independent production scale: every layer runs `STEP` full-batch steps.
STEP = 512
TOKENS_PER_STEP = STEP_TOKENS             # 48 tokens per step
TOKENS_TOTAL = STEP * TOKENS_PER_STEP     # 24576 tokens over all 512 steps

_TOL_ABS = 1e-3
_TOL_REL = 1e-2

_TRACE = os.environ.get("KVDBG", "1") != "0"


def _trace(msg):
    if _TRACE:
        print(f"[progression] {msg}", flush=True)


def _host_blocks() -> int:
    return 4096  # 4096 * 128 = 524288 host tokens ("big" host KV pool)


def _npu_or_skip():
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() == 0:
        pytest.skip("Ascend NPU not available")


# --------------------------------------------------------------------------- #
# Real implementation wiring
# --------------------------------------------------------------------------- #
def _real_impl():
    """A real AscendSFAKVOffloadImpl object with the attributes `_compute_kv_only`
    reads (num_kv_heads, kv_lora_rank, qk_rope_head_dim, kv_a_layernorm) bound;
    the production method runs unchanged."""
    impl = object.__new__(AscendSFAKVOffloadImpl)
    impl.num_kv_heads = NUM_KV_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM

    class _Ln:
        weight = torch.zeros(KV_LORA_RANK, dtype=DTYPE, device="npu:0")
        variance_epsilon = EPS
    impl.kv_a_layernorm = _Ln()
    return impl


def _real_manager(offload, device, num_blocks=None):
    """A real SparseKVOffloadManager with a REAL MemFabric host KV pool and the
    D2H descriptor tensors the real `offload_new_kv` (decode branch) uses.
    Fields mirror `init_sparse_kv_offload_manager`/`register_kv_caches`."""
    mgr = object.__new__(SparseKVOffloadManager)
    mgr.tp_rank = 0  # TP0 writes decode tokens
    mgr.block_size = BLOCK_SIZE

    hb = num_blocks if num_blocks is not None else _host_blocks()

    class _Cfg:  # keep_device_kv_cache is only read when has_prefill=True
        keep_device_kv_cache = False
    mgr.sparse_kv_offload_config = _Cfg()

    # Host-side big KV cache (4D, production layout), backed by MemFabric DRAM
    # so the real sparse_copy / copy_ D2H can target it.
    k_cpu = offload.empty([hb, BLOCK_SIZE, NUM_KV_HEADS, KV_LORA_RANK],
                          dtype=DTYPE).zero_()
    v_cpu = offload.empty([hb, BLOCK_SIZE, NUM_KV_HEADS, QK_ROPE_HEAD_DIM],
                          dtype=DTYPE).zero_()
    mgr.k_caches_cpu = [k_cpu]
    mgr.v_caches_cpu = [v_cpu]

    max_num_tokens = hb * BLOCK_SIZE
    mgr.max_num_tokens = max_num_tokens
    d2h_rows = max_num_tokens * 2
    mgr.d2h_src_ptrs_npu = torch.empty(d2h_rows, dtype=torch.int64, device=device)
    mgr.d2h_dst_ptrs_npu = torch.empty(d2h_rows, dtype=torch.int64, device=device)
    mgr.d2h_lengths_npu = torch.empty(d2h_rows, dtype=torch.int32, device=device)
    mgr.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
    mgr.d2h_token_indices_npu = torch.arange(max_num_tokens, dtype=torch.int64,
                                             device=device)
    mgr.token_size_bytes_k = NUM_KV_HEADS * KV_LORA_RANK * DTYPE.itemsize
    mgr.token_size_bytes_v = NUM_KV_HEADS * QK_ROPE_HEAD_DIM * DTYPE.itemsize
    return mgr, k_cpu, v_cpu


def _offload_step(mgr, k_cpu, v_cpu, impl, kv_, cos_, sin_, sl):
    """One baseline step: compute this step's tokens with the real
    `_compute_kv_only` and ONE-SHOT MemFabric D2H of them (Layer 2 per-step
    decode offload)."""
    k_nope, k_pe = impl._compute_kv_only(kv_, cos_, sin_)
    mgr.offload_new_kv(slot_mapping=sl, k_cache_cpu=k_cpu, v_cache_cpu=v_cpu,
                       k_cache_npu=None, v_cache_npu=None, k=k_nope, v=k_pe,
                       has_prefill=False, capturing=False)
    torch.npu.synchronize()


def _sparse_copy_offload(k_src, v_src, k_dst, v_dst, slots, sync=True):
    """Scatter rows 0..n-1 of `k_src`/`v_src` (on-device block caches) into the
    host `k_dst`/`v_dst` pools at per-row host `slots` using ONE MemFabric
    `offload.sparse_copy` scater (K = k_nope and V = k_pe in a single scatter).
    `slots` is a device int64 tensor of host token destinations.

    `k_src.shape[-1]` == KV_LORA_RANK and `v_src.shape[-1]` == QK_ROPE_HEAD_DIM,
    so their per-token byte strides equal the host pool's flat row stride (slots
    index flat host rows, matching `_assert_matches_golden`). Returns the row
    count. With `sync=True` it synchronizes (blocking); `sync=False` runs fully
    async on the caller's current stream so it can overlap other compute."""
    from memfabric_hybrid import offload  # noqa: PLC0415 (lazy, gated by fixture)

    n = int(slots.numel())
    if n == 0:
        return 0
    device = slots.device
    tsize_k = k_src.shape[-1] * k_src.element_size()
    tsize_v = v_src.shape[-1] * v_src.element_size()
    idx = torch.arange(n, dtype=torch.int64, device=device)
    sp_src = torch.empty(2 * n, dtype=torch.int64, device=device)
    sp_dst = torch.empty(2 * n, dtype=torch.int64, device=device)
    sp_len = torch.empty(2 * n, dtype=torch.int32, device=device)
    sp_size = torch.empty(1, dtype=torch.int32, device=device)
    sp_src[:n].copy_(int(k_src.data_ptr()) + idx * tsize_k)
    sp_src[n:2 * n].copy_(int(v_src.data_ptr()) + idx * tsize_v)
    sp_dst[:n].copy_(int(k_dst.data_ptr()) + slots * tsize_k)
    sp_dst[n:2 * n].copy_(int(v_dst.data_ptr()) + slots * tsize_v)
    sp_len[:n].fill_(tsize_k)
    sp_len[n:2 * n].fill_(tsize_v)
    sp_size.fill_(2 * n)
    result = offload.sparse_copy(sp_src, sp_dst, sp_len, sp_size, device)
    if result not in (None, 0):
        raise RuntimeError(f"MemFabric sparse_copy failed with result={result}")
    if sync:
        torch.npu.synchronize()
    return n


def offload_block(k_block, v_block, k_cache_cpu, v_cache_cpu, num_valid, start=0):
    """Layer-5 helper: ONE bulk MemFabric `offload.sparse_copy` of `num_valid`
    token rows into the host KV pool (replaces per-token `offload_new_kv`).

    `k_block`/`v_block` hold the on-device block rows destined for the host K/V
    pools (k_nope / k_pe), row-major, with `>= num_valid` rows in dim 0. Rows
    `0..num_valid-1` are scattered to host flat token rows `[start : start +
    num_valid)` (slot == start + row), as a single sparse_copy (no block
    alignment required; num_valid may span several host blocks and/or a partial
    trailing block)."""
    slots = torch.arange(start, start + num_valid, dtype=torch.int64,
                         device=k_block.device)
    _sparse_copy_offload(
        k_src=k_block, v_src=v_block, k_dst=k_cache_cpu, v_dst=v_cache_cpu,
        slots=slots, sync=True)


def _fused_block_caches(device):
    """The fused op's two on-device block caches [NUM_BLOCK, BLOCK_SIZE, 1, D]
    (PA_BSND, addressed by buffer-local slot_mapping = block*BLOCK_SIZE + off):
    k_rope = k_pe (64), k_nope = k_nope (512)."""
    k_rope = torch.zeros(NUM_BLOCK, BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM,
                         dtype=DTYPE, device=device)
    k_nope = torch.zeros(NUM_BLOCK, BLOCK_SIZE, 1, KV_LORA_RANK,
                         dtype=DTYPE, device=device)
    return k_rope, k_nope


def _fused_fill(kv_step, weight, cos_step, sin_step, offset, k_rope, k_nope, device):
    """ONE fused `npu_kv_rmsnorm_rope_cache` (compute + in-place save): computes
    k_pe/k_nope for a 48-token step and saves them into the block caches at
    buffer-local slots [offset : offset+TOKENS_PER_STEP) (cache_mode="PA", as in
    sfa_v1.exec_kv)."""
    slot = torch.arange(offset, offset + TOKENS_PER_STEP,
                        dtype=torch.int64, device=device)
    kvb = kv_step.view(TOKENS_PER_STEP, NUM_KV_HEADS, 1,
                       KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    torch_npu.npu_kv_rmsnorm_rope_cache(kvb, weight, cos_step, sin_step, slot,
                                        k_rope, k_nope, epsilon=EPS,
                                        cache_mode="PA")


def _assert_matches_golden(d, k_cpu, v_cpu, num_tokens, label):
    """Compare the filled host KV pool against DataHelper's numpy golden and
    return (rel_k, rel_v). slots=arange(B) and B is an exact block multiple, so
    host row t == token t; the host holds bf16 output vs float32 golden, so the
    check is RELATIVE error (bf16 accuracy is relative)."""
    host_k = k_cpu.view(-1, KV_LORA_RANK)[:num_tokens].to(torch.float64)
    host_v = v_cpu.view(-1, QK_ROPE_HEAD_DIM)[:num_tokens].to(torch.float64)
    ref_k, ref_v = d.standard()
    rel_k = ((host_k - ref_k.view(-1, KV_LORA_RANK)).abs() / ref_k.abs().max()).max().item()
    rel_v = ((host_v - ref_v.view(-1, QK_ROPE_HEAD_DIM)).abs() / ref_v.abs().max()).max().item()
    assert rel_k <= _TOL_REL, f"{label}: K host vs golden rel={rel_k:.3e}"
    assert rel_v <= _TOL_REL, f"{label}: V host vs golden rel={rel_v:.3e}"
    return rel_k, rel_v


# --------------------------------------------------------------------------- #
# Fixtures / guards
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def memfabric_pool():
    """Real MemFabric offload handle (local-DRAM scene), or skip when the
    deployed runtime is unavailable. Returns the `offload` module."""
    _npu_or_skip()
    if not os.environ.get("MEMFABRIC_HYBRID_EXTEND_LIB_PATH"):
        pytest.skip(
            "MemFabric Hybrid not deployed: MEMFABRIC_HYBRID_EXTEND_LIB_PATH unset. "
            f"See {_DOC} (Decode Dependencies)."
        )
    from memfabric_hybrid import offload

    # Reserve for all host K/V pool pairs (LOCAL requires reserve==alloc; the
    # allocator does not reclaim). layer2:1, layer3:2, layer5:2 -> 5 total.
    hb = _host_blocks()
    pool_bytes = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * DTYPE.itemsize * hb * BLOCK_SIZE
    reserve = 6 * pool_bytes + (1 << 22)  # five pairs of pools + 4MiB margin

    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.reserve_size = reserve
    cfg.alloc_size = reserve
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.LOCAL
    rc = offload.initialize(cfg)
    if rc != 0:
        pytest.skip(
            f"MemFabric offload.initialize failed (rc={rc}); container lacks the "
            f"deployed runtime. Configure per {_DOC} (Decode Dependencies)."
        )
    yield offload
    offload.uninitialize()


def _rotary_cos_sin(num_tokens, head_dim=QK_ROPE_HEAD_DIM, base=10000.0):
    positions = torch.arange(num_tokens, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles).unsqueeze(1).unsqueeze(2).to(DTYPE).npu()
    sin = torch.sin(angles).unsqueeze(1).unsqueeze(2).to(DTYPE).npu()
    return cos, sin


def _inputs(B):
    torch.manual_seed(11)
    device = "npu:0"
    kv = torch.randn(B, NUM_KV_HEADS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                     dtype=DTYPE, device=device).contiguous()
    weight = torch.randn(KV_LORA_RANK, dtype=DTYPE, device=device)
    cos, sin = _rotary_cos_sin(B)
    return kv, weight, cos, sin


# ------------------------- cross-layer test data (DataHelper) --------------------- #
def iter_steps(step, tokens_per_step, kv, cos, sin):
    """Yield (s, sl, kv_step, cos_step, sin_step) for each of `step` full-batch
    steps, so every layer's per-step loop is identical and uniform."""
    for s in range(step):
        sl = slice(s * tokens_per_step, (s + 1) * tokens_per_step)
        yield s, sl, kv[sl], cos[sl], sin[sl]


def _rotary_cos_sin_numpy(num_tokens, head_dim=QK_ROPE_HEAD_DIM, base=10000.0):
    """RoPE cos/sin for `num_tokens` positions as np.float64 [N,1,1,head_dim]
    (mirrors `_rotary_cos_sin`: the head-dim pairs share one angle)."""
    positions = np.arange(num_tokens, dtype=np.float64)
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles)[:, None, None, :], np.sin(angles)[:, None, None, :]


class DataHelper:
    """Spec-decode test data for the SFA sparse-KV offload progression.

    Interface design (open-closed):
      * Layers never touch internal attributes; they use only the stable methods
        `gen_inputs()`, `standard()` (and `input_numpy()`/`standard_numpy()`) plus
        the read-only `num_tokens`/`step`/`tokens_per_step` properties. The
        internals may change without touching any layer.
      * SHARED SAVE: DataHelper is a per-(step, tokens_per_step) cached singleton, so
        every layer sees the IDENTICAL inputs and standard values -- generated
        once in-process, never regenerated. (No on-disk persistence for now.)
      * Precision: numpy generation AND golden computation are float32 -- plenty
        for the bf16-rounded kernels under test.
    """

    _cache = {}

    def __new__(cls, step=STEP, tokens_per_step=TOKENS_PER_STEP):
        key = (step, tokens_per_step)
        if key not in cls._cache:
            obj = super().__new__(cls)
            obj._init(step, tokens_per_step)
            cls._cache[key] = obj     # save the instance: all layers share it
        return cls._cache[key]

    def _init(self, step, tokens_per_step):
        self._step = step
        self._tokens_per_step = tokens_per_step
        self._num_tokens = step * tokens_per_step
        self._generate()

    def _generate(self):
        B = self._num_tokens
        rng = np.random.default_rng(11)
        self._kv_np = rng.standard_normal(
            (B, NUM_KV_HEADS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM)).astype(np.float32)
        self._weight_np = rng.standard_normal(KV_LORA_RANK).astype(np.float32)
        self._cos_np, self._sin_np = (c.astype(np.float32) for c in
                                      _rotary_cos_sin_numpy(B))
        self._k_nope_np, self._k_pe_np = self._golden(
            self._kv_np, self._weight_np, self._cos_np, self._sin_np)

    # ---- public read-only interface ----
    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def step(self):
        return self._step

    @property
    def tokens_per_step(self):
        return self._tokens_per_step

    def gen_inputs(self):
        """Inputs as torch tensors (bf16, on npu:0): (kv, weight, cos, sin)."""
        device = "npu:0"
        return (
            torch.from_numpy(self._kv_np).to(DTYPE).to(device),
            torch.from_numpy(self._weight_np).to(DTYPE).to(device),
            torch.from_numpy(self._cos_np).to(DTYPE).to(device),
            torch.from_numpy(self._sin_np).to(DTYPE).to(device),
        )

    def standard(self):
        """Standard/golden values as torch float32 (CPU): (k_nope, k_pe)."""
        return torch.from_numpy(self._k_nope_np), torch.from_numpy(self._k_pe_np)

    def input_numpy(self):
        return (self._kv_np, self._weight_np, self._cos_np, self._sin_np)

    def standard_numpy(self):
        return self._k_nope_np, self._k_pe_np

    @staticmethod
    def _golden(kv_np, w_np, cos_np, sin_np):
        """Golden (numpy float32) k_nope/k_pe mirroring `_compute_kv_only`:
          k_nope = RMSNorm(kv[..., :kv_lora_rank]) * weight
          k_pe   = npu_interleave_rope(kv[..., kv_lora_rank:], cos, sin)

        npu_interleave_rope (per the official CANN API doc):
          q = reshape(x,[B,N,S,D/2,2]).transpose(-1,-2).reshape([B,N,S,D])
              # == interleave: [x0,x2,...,x_{D-2}, x1,x3,...,x_{D-1}] (even,odd)
          RotateHalf(q) = [-q[D/2:], q[:D/2]]   # rotate-half of the last dim
          out = q * cos + RotateHalf(q) * sin
        """
        B, N, S, D = kv_np.shape
        hd = D - KV_LORA_RANK
        x = kv_np[..., :KV_LORA_RANK].reshape(B, KV_LORA_RANK)
        var = np.mean(x * x, axis=-1, keepdims=True)
        k_nope = (x / np.sqrt(var + EPS) * w_np[None, :]).reshape(B, N, S, KV_LORA_RANK)

        rr = kv_np[..., KV_LORA_RANK:].reshape(B, N, S, hd)
        q = rr.reshape(B, N, S, hd // 2, 2).transpose(0, 1, 2, 4, 3).reshape(B, N, S, hd)
        rh = np.concatenate([-q[..., hd // 2:], q[..., :hd // 2]], axis=-1)
        k_pe = (q * cos_np.reshape(B, N, S, hd)
                + rh * sin_np.reshape(B, N, S, hd))
        return k_nope, k_pe


# --------------------------------------------------------------------------- #
# Layer 1 -- most basic: _compute_kv_only vs independent RMSNorm (no offload)
# --------------------------------------------------------------------------- #
def test_1_compute_kv_only_matches_independent_rms():
    """The REAL `_compute_kv_only`: k_nope equals an independent fused-RMSNorm;
    k_pe preserves the rope vector norm (RoPE is a rotation).

    Template at production spec-decode scale: ACTUALLY run 512 steps, each a
    48-token full-batch step (batch 8 x tokens/batch 6). Precision is verified
    on the COMPLETE 512-step result (all 24 576 tokens)."""
    _npu_or_skip()
    d = DataHelper()
    B = d.num_tokens
    kv, weight, cos, sin = d.gen_inputs()
    impl = _real_impl()
    impl.kv_a_layernorm.weight = weight

    # Real forward loop: 512 steps, each computing a 48-token full-batch step.
    k_nope_parts, k_pe_parts = [], []
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        n, p = impl._compute_kv_only(kv_step, cos_step, sin_step)
        k_nope_parts.append(n)
        k_pe_parts.append(p)
    k_nope = torch.cat(k_nope_parts, dim=0)   # [24576,1,1,512]
    k_pe = torch.cat(k_pe_parts, dim=0)       # [24576,1,1,64]

    # Standard values from DataHelper's PURE-NUMPY golden: RMSNorm (k_nope) and
    # interleave RoPE (k_pe), decoupled from torch/fused kernels. bf16 accuracy
    # is RELATIVE: normalized L-inf relative error over the WHOLE 512-step result.
    g_nope, g_pe = d.standard()
    diff = (k_nope.float().cpu().to(torch.float64) - g_nope.double()).abs()
    rel = (diff / g_nope.double().abs().max()).max().item()
    diff_pe = (k_pe.float().cpu().to(torch.float64) - g_pe.double()).abs()
    rel_pe = (diff_pe / (g_pe.double().abs().max() + 1e-12)).max().item()
    assert rel <= _TOL_REL, f"k_nope vs numpy golden rel={rel:.3e}"
    assert rel_pe <= _TOL_REL, f"k_pe vs numpy golden rel={rel_pe:.3e} (interleave rope)"
    _trace(f"L1 _compute_kv_only x{STEP} steps({TOKENS_PER_STEP} tok/step, {B} tok): "
           f"k_nope rel={rel:.3e}, k_pe rel={rel_pe:.3e} (vs numpy golden)")


# --------------------------------------------------------------------------- #
# Layer 2 -- full baseline: exec_kv computes + memfabric sparse D2H to host pool
# --------------------------------------------------------------------------- #
def test_2_baseline_exec_kv_offloads_via_memfabric(memfabric_pool):
    """Real exec_kv path: REAL `_compute_kv_only` + REAL `offload_new_kv` ->
    MemFabric sparse D2H into the real big host KV pool at mapped slots.

    Template at production spec-decode scale: ACTUALLY run 512 steps, each a
    48-token full-batch step (compute + one-shot D2H). The COMPLETE 512-step
    host result is compared directly against DataHelper's numpy golden (no separate
    baseline is recomputed here)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE      # 192 host blocks
    slots = torch.arange(B, dtype=torch.int64, device=kv.device)  # token t -> slot t
    device = kv.device

    impl = _real_impl()
    impl.kv_a_layernorm.weight = weight
    mgr, k_cpu, v_cpu = _real_manager(memfabric_pool, device, num_blocks=hb_total + 16)

    # Real forward loop: 512 steps, each computes 48 tokens + one-shot D2H.
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        _offload_step(mgr, k_cpu, v_cpu, impl, kv_step, cos_step, sin_step,
                      slots[sl].clone())

    # Compare the COMPLETE 512-step host result against the shared numpy golden.
    rel_k, rel_v = _assert_matches_golden(d, k_cpu, v_cpu, B, "L2 baseline")
    _trace(f"L2 baseline: {B} rows (512 steps x 48 tok) landed; vs numpy golden "
           f"K rel={rel_k:.3e}, V rel={rel_v:.3e}")


# --------------------------------------------------------------------------- #
# Layer 3 -- add a block buffer to cut offload count: 10 offloads -> 1
# --------------------------------------------------------------------------- #
def test_3_block_accumulate_reduces_offloads(memfabric_pool):
    """On-device block buffer with IN-LOOP offload (num_block=32).

    Draw Tokens accumulate into a num_block=32 on-device buffer (32*128 = 4096
    tokens). The offload logic lives inside the step loop: BEFORE each step's
    compute, if the buffer can no longer hold this step's new tokens, the
    accumulated data is offloaded to the host pool first (releasing the buffer)
    -- exactly the production blocking scheme. The COMPLETE host result must
    match DataHelper's numpy golden (same shared input/standard as L1/L2)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE      # 192 host blocks
    slots = torch.arange(B, dtype=torch.int64, device=kv.device)  # token t -> slot t
    device = kv.device

    impl = _real_impl()
    impl.kv_a_layernorm.weight = weight

    # On-device block buffer as a FUSED LINEAR [NUM_BLOCK*BLOCK_SIZE, 1, D] token
    # axis, so fill/offload are boundary-free slices (L4/L5/L6 use the native
    # 4-D PA_BSND layout instead).
    k_rope_block = torch.zeros(NUM_BLOCK * BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM,
                               dtype=DTYPE, device=device)
    k_nope_block = torch.zeros(NUM_BLOCK * BLOCK_SIZE, 1, KV_LORA_RANK,
                               dtype=DTYPE, device=device)

    mgr, kb, vb = _real_manager(memfabric_pool, device, num_blocks=hb_total + NUM_BLOCK)
    offset = 0          # tokens currently filled in the on-device buffer
    host_t = 0          # next host token (== slot) to write

    def flush():
        """One-shot D2H of the buffer's filled [0:offset] into the host pool at
        the matching slot range, releasing the buffer (a linear fused slice)."""
        nonlocal offset, host_t
        if offset == 0:
            return
        mgr.offload_new_kv(
            slot_mapping=slots[host_t:host_t + offset],
            k_cache_cpu=kb, v_cache_cpu=vb, k_cache_npu=None, v_cache_npu=None,
            k=k_nope_block[:offset], v=k_rope_block[:offset],
            has_prefill=False, capturing=False)
        torch.npu.synchronize()
        host_t += offset
        offset = 0

    offloads = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        # BEFORE this step's compute: offload first if the new tokens don't fit.
        if offset + TOKENS_PER_STEP > NUM_BLOCK * BLOCK_SIZE:
            flush()
            offloads += 1
        k_nope, k_pe = impl._compute_kv_only(kv_step, cos_step, sin_step)
        k_nope_block[offset:offset + TOKENS_PER_STEP] = k_nope.squeeze(2)
        k_rope_block[offset:offset + TOKENS_PER_STEP] = k_pe.squeeze(2)
        offset += TOKENS_PER_STEP
    if offset:
        flush()
        offloads += 1
    assert host_t == B, f"hosted {host_t}/{B} tokens"

    rel_k, rel_v = _assert_matches_golden(d, kb, vb, B, "L3 block-buffer")
    _trace(f"L3 block-buffer(num_block={NUM_BLOCK}, {STEP}x{TOKENS_PER_STEP}): "
           f"host == numpy golden (K rel={rel_k:.3e}, V rel={rel_v:.3e}), "
           f"offloads={offloads}")


# --------------------------------------------------------------------------- #
# Layer 4 -- fuse compute+save into one op: npu_kv_rmsnorm_rope_cache
# --------------------------------------------------------------------------- #
def test_4_fused_op_saves_block(memfabric_pool):
    """THE FUSED OP replaces `_compute_kv_only` + the manual store of Layer 3.

    `npu_kv_rmsnorm_rope_cache` (cache_mode="PA", as in sfa_v1.exec_kv) both
    COMPUTES k_pe/k_nope AND SAVES them in place into its own on-device block
    caches, addressed by a buffer-local slot_mapping. Layer-3's offload-and-
    compare logic is kept verbatim; the final host result must match
    DataHelper's numpy golden (same shared input/standard as L1/L2)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE      # 192 host blocks
    slots = torch.arange(B, dtype=torch.int64, device=kv.device)  # token t -> slot t
    device = kv.device

    k_rope_block, k_nope_block = _fused_block_caches(device)

    mgr, kb, vb = _real_manager(memfabric_pool, device, num_blocks=hb_total + NUM_BLOCK)
    offset = 0          # tokens currently filled in the on-device caches
    host_t = 0          # next host token (== slot) to write

    def flush():
        """One-shot D2H of the caches' filled [0:offset] into the host pool at
        the matching slot range, releasing the buffer (a linear fused slice)."""
        nonlocal offset, host_t
        if offset == 0:
            return
        # K pool holds k_nope (512); V pool holds k_pe (64).
        mgr.offload_new_kv(
            slot_mapping=slots[host_t:host_t + offset],
            k_cache_cpu=kb, v_cache_cpu=vb, k_cache_npu=None, v_cache_npu=None,
            k=k_nope_block.view(NUM_BLOCK * BLOCK_SIZE, 1, KV_LORA_RANK)[:offset],
            v=k_rope_block.view(NUM_BLOCK * BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM)[:offset],
            has_prefill=False, capturing=False)
        torch.npu.synchronize()
        host_t += offset
        offset = 0

    offloads = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        # BEFORE this step's compute: offload first if the new tokens don't fit.
        if offset + TOKENS_PER_STEP > NUM_BLOCK * BLOCK_SIZE:
            flush()
            offloads += 1
        _fused_fill(kv_step, weight, cos_step, sin_step, offset,
                    k_rope_block, k_nope_block, device)
        offset += TOKENS_PER_STEP
    if offset:
        flush()
        offloads += 1
    assert host_t == B, f"hosted {host_t}/{B} tokens"

    rel_k, rel_v = _assert_matches_golden(d, kb, vb, B, "L4 fused-op")
    _trace(f"L4 fused-op(num_block={NUM_BLOCK}, {STEP}x{TOKENS_PER_STEP}): "
           f"host == numpy golden (K rel={rel_k:.3e}, V rel={rel_v:.3e}), "
           f"offloads={offloads}")


# --------------------------------------------------------------------------- #
# Layer 5 -- offload_block (one-shot block save) replaces offload_new_kv
# --------------------------------------------------------------------------- #
def test_5_offload_block_replaces_offload_new_kv(memfabric_pool):
    """Layer-4 scheme, but the offload is done by `offload_block` (ONE bulk
    copy_ of all valid tokens) instead of per-token `offload_new_kv`.

    Same structure as L3/L4: DataHelper 512x48, fused op saves into the block
    caches, in-loop pre-check offload. `offload_block` writes the whole filled
    range to flat host rows [host_t : host_t+offset] in a single bulk copy_ (no
    block alignment needed). The COMPLETE host result must match DataHelper's
    numpy golden (same as L1-L4)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE      # 192 host blocks
    device = kv.device

    k_rope_block, k_nope_block = _fused_block_caches(device)
    _, kb, vb = _real_manager(memfabric_pool, device, num_blocks=hb_total + NUM_BLOCK)
    offset = 0          # tokens currently filled in the on-device caches
    host_t = 0          # next host token (== slot) to write

    def flush():
        """Offload this buffer's ALL valid tokens in ONE bulk copy_ via
        `offload_block` (flat host rows [host_t : host_t+offset]); no block
        alignment or loop needed."""
        nonlocal offset, host_t
        if offset == 0:
            return
        offload_block(
            k_block=k_nope_block.view(NUM_BLOCK * BLOCK_SIZE, 1, KV_LORA_RANK)[:offset],
            v_block=k_rope_block.view(NUM_BLOCK * BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM)[:offset],
            k_cache_cpu=kb, v_cache_cpu=vb, num_valid=offset, start=host_t)
        torch.npu.synchronize()
        host_t += offset
        offset = 0

    n_flush = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        # BEFORE this step's compute: offload first if the new tokens don't fit.
        if offset + TOKENS_PER_STEP > NUM_BLOCK * BLOCK_SIZE:
            flush()
            n_flush += 1
        _fused_fill(kv_step, weight, cos_step, sin_step, offset,
                    k_rope_block, k_nope_block, device)
        offset += TOKENS_PER_STEP
    if offset:
        flush()
        n_flush += 1
    assert host_t == B, f"hosted {host_t}/{B} tokens"

    rel_k, rel_v = _assert_matches_golden(d, kb, vb, B, "L5 offload_block")
    _trace(f"L5 offload_block(num_block={NUM_BLOCK}, {STEP}x{TOKENS_PER_STEP}): "
           f"host == numpy golden (K rel={rel_k:.3e}, V rel={rel_v:.3e}), "
           f"flushes={n_flush}")


# --------------------------------------------------------------------------- #
# Layer 6 -- latest double-buffer offload (precision guard vs baseline)
# --------------------------------------------------------------------------- #
def test_6_double_buffered_offload_guards_precision(memfabric_pool):
    """L3/L4/L5 structure, but with TWO NUM_BLOCK=32 on-device caches: when the
    active cache can't hold the next step, its full content is offloaded on a
    SIDE STREAM (non-blocking) while the OTHER cache keeps filling, overlapping
    D2H with the fused compute. The COMPLETE host result must match DataHelper's
    numpy golden (same shared input/standard as L1-L5)."""
    d = DataHelper()
    B, kv, weight, cos, sin = d.num_tokens, *d.gen_inputs()
    hb_total = (B + BLOCK_SIZE - 1) // BLOCK_SIZE      # 192 host blocks
    device = kv.device

    BUFFER = NUM_BLOCK * BLOCK_SIZE               # 4096 tokens per cache
    # Two 4-D PA_BSND caches [NUM_BLOCK, BLOCK_SIZE, 1, D] (buffer-local slots);
    # flat views serve the non-blocking D2H at plain host token rows (no block
    # alignment constraint, matching L5).
    k_rope = [torch.zeros(NUM_BLOCK, BLOCK_SIZE, 1, QK_ROPE_HEAD_DIM,
                          dtype=DTYPE, device=device) for _ in range(2)]
    k_nope = [torch.zeros(NUM_BLOCK, BLOCK_SIZE, 1, KV_LORA_RANK,
                          dtype=DTYPE, device=device) for _ in range(2)]

    _, kb, vb = _real_manager(memfabric_pool, device, num_blocks=hb_total + NUM_BLOCK)
    offload_stream = torch.npu.Stream()
    compute = torch.npu.current_stream()
    done = [torch.npu.Event(), torch.npu.Event()]
    pending = [False, False]
    active, off, host_t = 0, 0, 0

    def offload_buffer(idx, num_tokens):
        """Non-blocking D2H of cache idx's filled [0:num_tokens) into host flat
        rows [host_t : host_t+num_tokens) via ONE MemFabric `offload.sparse_copy`
        (slot == host_t + row) on the side stream — no sync, so it overlaps the
        other cache's fused compute."""
        nonlocal host_t
        slots = torch.arange(host_t, host_t + num_tokens, dtype=torch.int64, device=device)
        with torch.npu.stream(offload_stream):
            offload_stream.wait_stream(compute)
            _sparse_copy_offload(
                k_src=k_nope[idx].view(-1, KV_LORA_RANK)[:num_tokens],
                v_src=k_rope[idx].view(-1, QK_ROPE_HEAD_DIM)[:num_tokens],
                k_dst=kb, v_dst=vb, slots=slots, sync=False,
            )
        done[idx].record(offload_stream)
        pending[idx] = True
        host_t += num_tokens

    n_offload = 0
    for s, sl, kv_step, cos_step, sin_step in iter_steps(STEP, TOKENS_PER_STEP,
                                                         kv, cos, sin):
        # BEFORE this step's compute: offload the active cache (non-blocking)
        # and switch; wait for the reuse target's earlier offload to finish.
        if off + TOKENS_PER_STEP > BUFFER:
            offload_buffer(active, off)
            n_offload += 1
            dest = 1 - active
            if pending[dest]:
                compute.wait_event(done[dest])
                pending[dest] = False
            active, off = dest, 0
        # ONE fused kernel into the active cache at buffer-local slots.
        _fused_fill(kv_step, weight, cos_step, sin_step, off,
                    k_rope[active], k_nope[active], device)
        off += TOKENS_PER_STEP
    if off:
        offload_buffer(active, off)
        n_offload += 1
    compute.wait_stream(offload_stream)
    torch.npu.synchronize()
    assert host_t == B, f"hosted {host_t}/{B} tokens"

    rel_k, rel_v = _assert_matches_golden(d, kb, vb, B, "L6 double-buffer")
    _trace(f"L6 double-buffer(num_block={NUM_BLOCK}, {STEP}x{TOKENS_PER_STEP}): "
           f"host == numpy golden (K rel={rel_k:.3e}, V rel={rel_v:.3e}), "
           f"offloads={n_offload}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

