"""
Unit tests for the FPBuffer logic added to modeling_llamagear.py.

Runs entirely on CPU — no CUDA, no Llama weights required.
FPBuffer is re-defined inline here to avoid the CUDA kernel imports
at the top of modeling_llamagear.py.

Run with:
    python test_fpbuffer.py
"""

import sys
import torch
from typing import Optional


# ── Inline copy of FPBuffer (identical to modeling_llamagear.py) ─────────────

class FPBuffer:
    def __init__(self, sink_tokens: int, recency_tokens: int, buffer_len: int):
        self.sink_tokens    = sink_tokens
        self.recency_tokens = recency_tokens
        self.buffer_len     = buffer_len

        self._sink    : Optional[torch.Tensor] = None
        self._recency : Optional[torch.Tensor] = None
        self._buffer  : Optional[torch.Tensor] = None
        self._initialised = False

        if recency_tokens == 0 and buffer_len > 0:
            raise ValueError(
                "recency_tokens=0 is invalid when buffer_len > 0 — "
                "the flush window would always be empty."
            )

    def _sink_len(self) -> int:
        return self._sink.shape[-2] if self._sink is not None else 0

    def _recency_len(self) -> int:
        return self._recency.shape[-2] if self._recency is not None else 0

    def _buffer_len_cur(self) -> int:
        return self._buffer.shape[-2] if self._buffer is not None else 0

    def total_len(self) -> int:
        return self._sink_len() + self._recency_len() + self._buffer_len_cur()

    def get_fp_view(self) -> Optional[torch.Tensor]:
        parts = [p for p in (self._sink, self._recency, self._buffer) if p is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=-2)

    def append(self, new_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        T = new_tokens.shape[-2]

        if not self._initialised:
            total_capacity = self.sink_tokens + self.recency_tokens + self.buffer_len

            if T > total_capacity:
                to_compress   = new_tokens[..., self.sink_tokens: T - self.recency_tokens, :]
                self._sink    = new_tokens[..., :self.sink_tokens, :]
                self._recency = new_tokens[..., T - self.recency_tokens:, :]
                self._buffer  = None
                self._initialised = True
                return to_compress

            self._sink    = new_tokens[..., :self.sink_tokens, :]
            remainder     = new_tokens[..., self.sink_tokens:, :]
            recency_part  = min(self.recency_tokens, remainder.shape[-2])
            self._recency = remainder[..., :recency_part, :]
            buf_part      = remainder[..., recency_part:, :]
            self._buffer  = buf_part if buf_part.shape[-2] > 0 else None
            self._initialised = True
            return None

        if self._recency_len() < self.recency_tokens:
            self._recency = (
                torch.cat([self._recency, new_tokens], dim=-2)
                if self._recency is not None else new_tokens
            )
            return None

        self._buffer = (
            torch.cat([self._buffer, new_tokens], dim=-2)
            if self._buffer is not None else new_tokens
        )

        if self._buffer_len_cur() < self.buffer_len:
            return None

        to_compress   = self._recency[..., :self.buffer_len, :]
        self._recency = torch.cat(
            [self._recency[..., self.buffer_len:, :], self._buffer], dim=-2
        )
        self._buffer  = None
        return to_compress


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tokens(B, H, T, D):
    """Create a [B, H, T, D] tensor with unique values so we can track identity."""
    return torch.arange(T, dtype=torch.float32).view(1, 1, T, 1).expand(B, H, T, D).clone()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  →  {detail}" if detail else ""))
        sys.exit(1)


# ── Test 1: PREFILL OK ────────────────────────────────────────────────────────

def test_prefill_ok():
    """Short prefill — everything fits in FP zones, nothing to compress."""
    print("\nTest 1: PREFILL OK")
    B, H, D = 1, 2, 8
    sink, recency, buf_len = 4, 8, 8

    buf = FPBuffer(sink, recency, buf_len)
    tokens = make_tokens(B, H, 10, D)   # 10 < 4+8+8=20 → fits
    result = buf.append(tokens)

    check("returns None (no compression)",   result is None)
    check("sink has 4 tokens",               buf._sink_len() == 4)
    check("recency has 6 tokens (10-4=6)",   buf._recency_len() == 6)
    check("buffer is empty",                 buf._buffer_len_cur() == 0)
    check("total_len == 10",                 buf.total_len() == 10)
    view = buf.get_fp_view()
    check("fp_view shape == (B,H,10,D)",     tuple(view.shape) == (B, H, 10, D))


# ── Test 2: PREFILL SPLIT ─────────────────────────────────────────────────────

def test_prefill_split():
    """Long prefill — middle chunk must be compressed, sink + recency kept in FP."""
    print("\nTest 2: PREFILL SPLIT")
    B, H, D = 1, 2, 8
    sink, recency, buf_len = 4, 8, 8

    buf = FPBuffer(sink, recency, buf_len)
    T = 100                              # 100 > 4+8+8=20 → split
    tokens = make_tokens(B, H, T, D)
    result = buf.append(tokens)

    middle_len = T - sink - recency      # 100 - 4 - 8 = 88
    check("returns to_compress tensor",      result is not None)
    check(f"to_compress has {middle_len} tokens",
          result.shape[-2] == middle_len,
          f"got {result.shape[-2]}")
    check("sink has 4 tokens",               buf._sink_len() == 4)
    check("recency has 8 tokens",            buf._recency_len() == recency)
    check("buffer is empty",                 buf._buffer_len_cur() == 0)

    # Verify sink contains the FIRST tokens and recency the LAST tokens
    check("sink = first 4 tokens",
          torch.allclose(buf._sink[0, 0, :, 0], tokens[0, 0, :sink, 0]))
    check("recency = last 8 tokens",
          torch.allclose(buf._recency[0, 0, :, 0], tokens[0, 0, -recency:, 0]))


# ── Test 3: FILLING (warm-up during decode) ───────────────────────────────────

def test_filling():
    """After a short prefill, recency is not yet full — decode tokens fill it up."""
    print("\nTest 3: FILLING (warm-up)")
    B, H, D = 1, 2, 8
    sink, recency, buf_len = 4, 8, 8

    buf = FPBuffer(sink, recency, buf_len)
    # Prefill with only sink tokens — recency will be empty
    prefill = make_tokens(B, H, sink, D)
    result = buf.append(prefill)
    check("prefill returns None",            result is None)
    check("recency empty after sink-only prefill",  buf._recency_len() == 0)

    # Decode steps fill recency (should all return None)
    for i in range(recency):
        tok = make_tokens(B, H, 1, D)
        result = buf.append(tok)
        check(f"decode step {i+1} returns None (FILLING)",  result is None,
              f"got {result}")
        check(f"recency has {i+1} tokens",   buf._recency_len() == i + 1)

    check("recency full after warm-up",      buf._recency_len() == recency)
    check("buffer still empty",              buf._buffer_len_cur() == 0)


# ── Test 4: FLUSH ─────────────────────────────────────────────────────────────

def test_flush():
    """Once recency is full, buffer accumulates and triggers a flush."""
    print("\nTest 4: FLUSH")
    B, H, D = 1, 2, 8
    sink, recency, buf_len = 4, 8, 4   # smaller buf_len for quicker flush

    buf = FPBuffer(sink, recency, buf_len)
    # Prefill filling sink + recency exactly
    prefill = make_tokens(B, H, sink + recency, D)
    result = buf.append(prefill)
    check("prefill returns None",            result is None)
    check("recency full after prefill",      buf._recency_len() == recency)

    # Decode: accumulate into buffer — no flush until buf_len tokens
    for i in range(buf_len - 1):
        tok = make_tokens(B, H, 1, D)
        result = buf.append(tok)
        check(f"step {i+1}: no flush yet",   result is None)
        check(f"buffer has {i+1} token(s)",  buf._buffer_len_cur() == i + 1)

    # Final decode token triggers flush
    tok = make_tokens(B, H, 1, D)
    result = buf.append(tok)
    check("flush triggered on buf_len-th token",   result is not None)
    check(f"flushed chunk has {buf_len} tokens",   result.shape[-2] == buf_len,
          f"got {result.shape[-2]}")

    # After flush: buffer cleared, recency slid forward
    check("buffer cleared after flush",      buf._buffer_len_cur() == 0)
    check("recency still full after flush",  buf._recency_len() == recency)
    check("sink unchanged",                  buf._sink_len() == sink)
    check("total_len = sink + recency",      buf.total_len() == sink + recency)


# ── Test 5: get_fp_view order ─────────────────────────────────────────────────

def test_fp_view_order():
    """get_fp_view returns [sink | recency | buffer] in the correct order."""
    print("\nTest 5: get_fp_view order")
    B, H, D = 1, 1, 1
    sink, recency, buf_len = 2, 3, 3

    buf = FPBuffer(sink, recency, buf_len)
    # Prefill: 7 tokens → fills sink(2) + recency(3) + buffer(2)
    tokens = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1)
    buf.append(tokens)

    view = buf.get_fp_view()
    expected = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1)
    check("fp_view = original token order",
          torch.allclose(view, expected),
          f"got {view[0,0,:,0].tolist()}")


# ── Test 6: compress_config defaults ─────────────────────────────────────────

def test_config_defaults():
    """Simulate how LlamaAttention_GEAR reads compress_config with defaults."""
    print("\nTest 6: compress_config defaults")
    compress_config = {
        "residual": 64,
        "compress_method": "gearlKIVI",
        "group_size": 64,
        "quantize_bit": 2,
        "rank": 2,
        "rankv": 2,
        "loop": 3,
    }
    residual_length = compress_config["residual"]
    sink_tokens    = compress_config.get("sink_tokens",    4)
    recency_tokens = compress_config.get("recency_tokens", residual_length)
    buffer_len     = compress_config.get("buffer_len",     residual_length)

    check("sink_tokens default = 4",            sink_tokens == 4)
    check("recency_tokens default = residual",  recency_tokens == 64)
    check("buffer_len default = residual",      buffer_len == 64)

    buf = FPBuffer(sink_tokens, recency_tokens, buffer_len)
    tokens = make_tokens(1, 1, 10, 8)
    result = buf.append(tokens)
    check("FPBuffer created and append works",  result is None)

    # With explicit overrides
    compress_config["sink_tokens"]    = 4
    compress_config["recency_tokens"] = 32
    compress_config["buffer_len"]     = 32
    buf2 = FPBuffer(
        compress_config["sink_tokens"],
        compress_config["recency_tokens"],
        compress_config["buffer_len"],
    )
    check("FPBuffer with explicit params created", buf2.sink_tokens == 4)
    check("recency_tokens = 32",                   buf2.recency_tokens == 32)
    check("buffer_len = 32",                       buf2.buffer_len == 32)


# ── Test 7: decode kv_seq_len tracking ───────────────────────────────────────

def test_kv_seq_len_tracking():
    """
    Verify that total_len() correctly tracks kv_seq_len after
    multiple prefill + decode steps including a flush.
    """
    print("\nTest 7: kv_seq_len tracking across prefill + decode + flush")
    B, H, D = 1, 2, 8
    sink, recency, buf_len = 4, 8, 4

    buf_k = FPBuffer(sink, recency, buf_len)
    buf_v = FPBuffer(sink, recency, buf_len)

    # Prefill 12 tokens
    prefill_k = make_tokens(B, H, 12, D)
    prefill_v = make_tokens(B, H, 12, D)
    buf_k.append(prefill_k)
    buf_v.append(prefill_v)
    check("total_len after prefill = 12",     buf_k.total_len() == 12)

    # Simulate 4 decode steps (fills buffer → flush on step 4)
    flushed = 0
    for i in range(4):
        tok = make_tokens(B, H, 1, D)
        r_k = buf_k.append(tok)
        r_v = buf_v.append(tok)
        if r_k is not None:
            flushed = r_k.shape[-2]

    check("flush happened during decode",     flushed == buf_len)
    # After flush: fp_view = sink(4) + recency(8) = 12 tokens
    fp_len = buf_k.total_len()
    check("fp_len after flush = sink + recency",  fp_len == sink + recency,
          f"got {fp_len}")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  FPBuffer Unit Tests")
    print("=" * 55)

    test_prefill_ok()
    test_prefill_split()
    test_filling()
    test_flush()
    test_fp_view_order()
    test_config_defaults()
    test_kv_seq_len_tracking()

    print("\n" + "=" * 55)
    print("  All tests passed.")
    print("=" * 55)
