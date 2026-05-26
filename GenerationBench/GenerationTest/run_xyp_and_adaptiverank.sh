#!/bin/bash
# Test script to verify adaptive rank is being used in GEAR compression

# python evaluation_gsm8k_true_compression.py \
#   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 8 \
#   --quantize_bit 4 \
#   --rank 1 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


python evaluation_gsm8k_true_compression.py \
  --model meta-llama/Meta-Llama-3-8B \
  --prompt_file gsm8k_prompt_original.txt \
  # --example_subset 0:10 \
  --compress_method GEAR \
  --compress_mode gear \
  --batch_size 4 \
  --quantize_bit 4 \
  --rank 1 \
  --loop 3 \
  --left 0.02 \
  --sink_tokens 4 \
  --recency_tokens 64 \
  --buffer_len 20 \
  --max_new_tokens 256


# python evaluation_aqua_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   # --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 2 \
#   --quantize_bit 4 \
#   --rank 1 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_bbh_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   # --prompt_file gsm8k_prompt_original.txt \
#   --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 2 \
#   --quantize_bit 4 \
#   --rank 1 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256