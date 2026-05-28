"""
Single-prompt generation test using TrueCompression (FP buffer GEAR method).
Reads a prompt from a .txt file and runs GearLlamaForCausalLMNew.generate().
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from GEARLM import GearLlamaForCausalLMNew


def main():
    parser = argparse.ArgumentParser(description="Single-prompt generation with TrueCompression")

    # ── Prompt / model ──────────────────────────────────────────────
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to .txt file containing the full prompt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF hub name or local path (Llama architecture only).",
    )
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save generation output.")

    # ── Sampling ──────────────────────────────────────────────────────
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    # ── TrueCompression / FP buffer GEAR ─────────────────────────────
    parser.add_argument(
        "--compress_method",
        type=str,
        default="gear",
        help="'None' disables compression; any other value enables it.",
    )
    parser.add_argument("--compress_mode", type=str, default="gear",
                        help="gear / uniform / outlier")
    parser.add_argument("--quantize_bit", type=int, default=4)
    parser.add_argument("--rank", type=float, default=0.0,
                        help="0 = adaptive rank in true_poweriteration")
    parser.add_argument("--loop", type=int, default=3)
    parser.add_argument("--left", type=float, default=0.02)
    parser.add_argument("--buffer_len", type=int, default=20)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recency_tokens", type=int, default=64)
    parser.add_argument("--stream", action="store_true", default=False,
                        help="Use StreamCompressedCache instead of CompressedCache.")
    parser.add_argument("--streaming_gap", type=int, default=1)

    # ── Prompt formatting ─────────────────────────────────────────────
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Wrap prompt with tokenizer.apply_chat_template (for chat models).",
    )

    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── compress_config (same dict as GSM8K script) ───────────────────
    if args.compress_method == "None":
        compress_config = None
    else:
        compress_config = {
            "compress_mode": args.compress_mode,
            "quantize_bit": args.quantize_bit,
            "rank": args.rank,
            "loop": args.loop,
            "left": args.left,
            "buffer_len": args.buffer_len,
            "stream": args.stream,
            "streaming_gap": args.streaming_gap,
            "sink_tokens": args.sink_tokens,
            "recency_tokens": args.recency_tokens,
        }

    # ── Load model ────────────────────────────────────────────────────
    model_kwargs = {"torch_dtype": torch.float16, "cache_dir": "../cache"}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    config = transformers.AutoConfig.from_pretrained(
        args.model, use_flash_attn=False, trust_remote_code=True,
        **({"token": args.hf_token} if args.hf_token else {}),
    )
    model = GearLlamaForCausalLMNew.from_pretrained(
        args.model, config=config, compress_config=compress_config, **model_kwargs,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", model_max_length=args.model_max_length,
        use_fast=False, cache_dir="../cache",
        **({"token": args.hf_token} if args.hf_token else {}),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ── Read prompt ───────────────────────────────────────────────────
    prompt_path = Path(args.prompt_file)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    if args.use_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # ── Generate ────────────────────────────────────────────────────
    generate_kwargs = dict(
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=args.do_sample,
    )
    if args.do_sample:
        generate_kwargs.update(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generation = tokenizer.decode(
        outputs.sequences[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print("=" * 60)
    print("PROMPT:")
    print(prompt_path.read_text(encoding="utf-8"))
    print("=" * 60)
    print("GENERATION:")
    print(generation)
    print("=" * 60)

    if args.output_file:
        Path(args.output_file).write_text(generation, encoding="utf-8")


if __name__ == "__main__":
    main()