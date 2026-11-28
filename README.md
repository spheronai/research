# Spheron Research

Studies run by Spheron Research on cloud GPUs. Each one asks a single question, freezes its plan before any measurement, and publishes the answer even when it is negative.

Each study is one folder with:

- `REPORT.md`: the question, the answer, how it was measured, the limits and how to cite it
- `DATA.md`: what every column in `results/` means
- `PLAN.md` and `notes/plan.sha256` (where the study was pre-registered): the frozen plan and its hashes
- `harness/`: the scripts that ran it
- `results/` and `charts/`: derived results, rebuilt from the raw data by `harness/analyze.py`

Raw data is too large for git. It is attached to the [releases](https://github.com/spheronai/research/releases) and linked from each report.

## Studies

| Study | Measured | Question | Data |
|---|---|---|---|
| [Study 1: FP8 vs NVFP4 vs AWQ](2026-11-23-qwen38-27b-quantization/REPORT.md) | 2026-11-23, 1x RTX PRO 6000, vLLM 0.34.0 | Which Qwen3.8-27B checkpoint should you serve? Accuracy on GSM8K, MMLU-Pro (subset), MATH-500, IFEval and needle retrieval, plus throughput, latency and energy, against BF16. | [raw data](https://github.com/spheronai/research/releases/download/2026.11.25/study1-qwen38-27b-quantization-raw.tar.gz); throughput runs are rows 491 to 550 of the [GPU Benchmarks catalog](https://github.com/spheronai/gpu-benchmarks) |
| [Study 2, run one: camera move reader](2026-11-24-camera-move-reader/REPORT.md) | 2026-11-24, 1x H100 PCIe | Can a small model on top of Depth Anything 3 read a camera move well enough to re-shoot it? | [Hugging Face](https://huggingface.co/datasets/spheron/camera-move-cards) (`run1` config) |
| [Study 2, run two: camera move reader](2026-11-25-camera-move-reader-2/REPORT.md) | 2026-11-24 to 25, 1x H100 PCIe | Does training on more kinds of motion make it beat Depth Anything 3 plus naming rules? | [Hugging Face](https://huggingface.co/datasets/spheron/camera-move-cards), or [one archive](https://github.com/spheronai/research/releases/download/2026.11.25/study2-camera-move-reader-dataset.tar.gz) |
| [Study 3: GLM-5.3-Flash BF16 vs FP8 vs NVFP4](2026-11-26-swe-rebench-glm-quantization/REPORT.md) | 2026-11-26 to 27, 8x B200, vLLM 0.28.1rc1 | Does FP8 or NVFP4 cost a coding agent accuracy on fresh tasks? 111 SWE-rebench tasks, 5 runs each, paired against BF16. | per-trial results in the study folder (`results/trials_classified.csv`) |

Study 1 in one line: at a latency target of 20 tok/s per user and first token p95 under 2 s, the NVFP4 checkpoint served 700 output tok/s at 16 concurrent users against 101 tok/s at 4 users for BF16, and none of the three quantized checkpoints lost a statistically significant amount of accuracy.

Study 2 in one line, a negative result across both runs: no trained head beat Depth Anything 3 plus a simple naming rule on held-out move pairs or on human camera paths, so Depth Anything 3 plus rules is the tool to use and the head's weights are not released.

Study 3 in one line: neither FP8 nor NVFP4 lost measurable accuracy against BF16 for GLM-5.3-Flash as a coding agent (FP8 +0.4 points, NVFP4 +2.7, both 95% intervals above the -5 point margin and crossing zero), and NVFP4 used the least node time per resolved trial.

Study pages: [study 1](https://www.spheron.network/research/qwen38-27b-fp8-vs-nvfp4-vs-awq/), [study 2](https://www.spheron.network/research/camera-move-reader-vs-depth-anything-3/), [study 3](https://www.spheron.network/research/glm53-flash-bf16-vs-fp8-vs-nvfp4-swe-rebench/).

## Related

- [Spheron GPU Benchmarks](https://github.com/spheronai/gpu-benchmarks): dated LLM inference throughput and latency measurements on cloud GPUs. Studies that measure inference throughput add their runs there with area `research`.

## History

These studies were first published in the GPU Benchmarks repository (releases 2026.11.24 and 2026.11.25) and moved here on 2026-11-25 with their git history.

## License

Data, charts and documentation are licensed under [Creative Commons Attribution 4.0 International](LICENSE-DATA): reuse them freely with credit to Spheron Research and a link to this repository. Code (each study's `harness/`) is licensed under the [MIT License](LICENSE). Model, GPU and software names are trademarks of their owners. This repository does not include or redistribute any model weights.
