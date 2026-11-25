# Data for study 1: Qwen3.8-27B checkpoints on one RTX PRO 6000

Every CSV in `results/` is produced by `harness/analyze.py` from the files in `raw/`. Rerun it with `uv run --with matplotlib python harness/analyze.py` from this folder. The shell scripts in `harness/` take the study folder from `STUDY_DIR` (default `$HOME/study`).

## Arms

| Arm | Label | Checkpoint (commit) | KV cache |
|---|---|---|---|
| A | BF16 | `Qwen/Qwen3.8-27B` (`37561816e66b`) | BF16 |
| B | FP8 | `Qwen/Qwen3.8-27B-FP8` (`c3a462cc3d38`) | BF16 |
| C | NVFP4 | `unsloth/Qwen3.8-27B-NVFP4` (`f0b7c9e722f5`) | BF16 (forced; the checkpoint ships an FP8 KV scheme) |
| D | INT4-AWQ | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (`e3a700906e6a`) | BF16 |
| E | FP8 + FP8 KV | same as B | FP8 |
| F | NVFP4 as shipped (FP8 KV) | same as C | `auto`, which picks the checkpoint's FP8 KV scheme |

All arms ran at `--gpu-memory-utilization 0.90` except C and F, which ran at 0.85 after running out of memory at startup at 0.90 (see the report's deviations). E and F ran GSM8K and the needle test only.

## results/quality.csv

One row per arm and eval.

| Column | Meaning |
|---|---|
| `eval` | `gsm8k` (5-shot, strict match), `mmlu_pro` (first 140 test questions of each of the 14 subjects, 5-shot CoT), `math500` (MATH-500, 4-shot Minerva prompt, math-verify scoring), `ifeval` (prompt-level strict accuracy), `niah` (all six needle tasks at 8k and 32k tokens pooled) |
| `n` | Questions scored |
| `correct` | Questions answered correctly (partial credit summed for the multivalue needle task) |
| `accuracy` | Fraction correct (0 to 1), unrounded: `correct / n` |
| `stderr` | Binomial standard error of `accuracy` |
| `delta_vs_bf16` | Mean per-question difference from arm A on the same questions (0.01 = 1 point) |
| `paired_delta_stderr` | Standard error of that paired difference |
| `paired_n` | Questions present in both this arm and arm A |
| `significant_2se` | `yes` when the absolute delta is more than two paired standard errors |

## results/niah.csv

| Column | Meaning |
|---|---|
| `task` | `single` (one needle), `multikey` (8 needles, one asked), `multivalue` (one key with 4 values, all must be returned; partial credit) |
| `length_tokens` | Prompt length in the model's own tokens, excluding chat template overhead |
| `accuracy` | Mean score over `n` prompts |

## results/throughput_runs.csv

One row per arm, concurrency and repeat. Workload: `vllm bench serve`, random token prompts, 1,024 input and 512 output tokens, `--ignore-eos`, seed = repeat number, prompts per level = max(8, 4 x concurrency), prefix caching off.

| Column | Meaning |
|---|---|
| `output_tok_s` | Output tokens per second over the whole level |
| `ttft_p50_ms`, `ttft_p95_ms` | Time to first token, median and 95th percentile |
| `tpot_p50_ms` | Median time per output token after the first |
| `itl_p50_ms` | Median gap between streamed tokens |
| `gpu_power_w` | Mean board power from `nvidia-smi` sampled every second during the level |
| `joules_per_output_token` | Mean power x level duration / output tokens. Empty for arm A: its power log was overwritten when its server restarted for MATH-500 |

## results/throughput_summary.csv

Median of the 3 repeats per arm and concurrency, plus:

| Column | Meaning |
|---|---|
| `output_tok_s_min`, `output_tok_s_max` | Spread across repeats |
| `per_user_tok_s` | 1000 / `tpot_p50_ms` |
| `meets_target` | `yes` when `tpot_p50_ms` <= 50 and `ttft_p95_ms` <= 2,000 |

## raw/

`raw/` is attached to release 2026.11.25 of this repository as `study1-qwen38-27b-quantization-raw.tar.gz` rather than kept in git. Unpack it in this folder before running `harness/analyze.py`. The VM hostname and the GPU serial numbers and UUIDs were removed from `env/`.

| Path | Contents |
|---|---|
| `env/` | `nvidia-smi -q`, pip freezes for both venvs, system info, pinned models and download times |
| `<arm>/evals/` | lm-eval results JSON and per-sample JSONL for each eval; `niah.jsonl` per prompt |
| `<arm>/bench/` | `vllm bench serve` JSON per concurrency and repeat; `windows.csv` with start and end times |
| `<arm>/logs/` | vLLM server log, power samples, per-eval logs, `timeline.log` |
| `failed-attempts/` | Runs that were stopped or failed, kept for the record (see the report's deviations) |
