# Nexora: Multi-GPU LLM Inference Simulator

A simulator for understanding multi-GPU LLM inference overhead. Built to answer: *Why does TP=4 (4×24GB GPUs) often feel worse than a single 96GB GPU?* and *What optimizations actually help?*

**No GPUs required** - runs on any system including laptops.

## What This Tool Is / Is Not

**This tool IS:**
- A *model* for reasoning about tensor-parallel communication overhead
- Useful for comparing hardware configurations (PCIe vs NVLink, TP=2 vs TP=4)
- A way to explore optimization tradeoffs (overlap, precision, grouping)
- An interactive visualization tool for understanding performance bottlenecks

**This tool is NOT:**
- A benchmark - it doesn't measure real hardware
- Exact tok/s predictions - real systems have many additional factors
- A replacement for profiling actual deployments

The simulator helps you understand *why* multi-GPU setups underperform and *where* to focus optimization efforts, but actual performance depends on your specific model, batch size, and system.

## Features

- **Model Presets** - Llama 7B/13B/70B, Mixtral 8x7B/8x22B configurations
- **Hardware Presets** - PCIe, NVLink, AMD Infinity Fabric, InfiniBand
- **Batch Size Modeling** - See how larger batches amortize latency overhead
- **Pipeline Parallelism** - Combined TP×PP configurations with micro-batching
- **MoE Expert Parallelism** - All-to-all communication for Mixtral-style models
- **Bottleneck Detection** - Identifies COMPUTE-BOUND, MEMORY-BOUND, or INTERCONNECT-BOUND
- **Interactive HTML Dashboard** - Real-time parameter adjustment with charts
- **ASCII Visualization** - Terminal-based bar charts for quick analysis
- **JSON Output** - Structured output for automation and scripting
- **Comparison Mode** - Side-by-side hardware preset comparison

## Installation (No Rust Required)

### Option 1: Download Prebuilt Binary

Download the binary for your platform and verify the checksum:

**macOS (Apple Silicon)**
```bash
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-arm64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-arm64.sha256
shasum -a 256 -c nexora-macos-arm64.sha256
chmod +x nexora-macos-arm64
./nexora-macos-arm64 --help
```

**macOS (Intel)**
```bash
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-x86_64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-x86_64.sha256
shasum -a 256 -c nexora-macos-x86_64.sha256
chmod +x nexora-macos-x86_64
./nexora-macos-x86_64 --help
```

**Linux (x86_64)**
```bash
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-linux-x86_64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-linux-x86_64.sha256
sha256sum -c nexora-linux-x86_64.sha256
chmod +x nexora-linux-x86_64
./nexora-linux-x86_64 --help
```

**Windows (PowerShell)**
```powershell
Invoke-WebRequest -Uri https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-windows-x86_64.exe -OutFile nexora.exe
Invoke-WebRequest -Uri https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-windows-x86_64.exe.sha256 -OutFile nexora.exe.sha256
(Get-FileHash nexora.exe -Algorithm SHA256).Hash.ToLower()
Get-Content nexora.exe.sha256
.\nexora.exe --help
```

### Option 2: Build from Source (requires Rust)

```bash
cargo build --release
# Binary at: ./target/release/nexora
```

## Quick Start

```bash
# See TP=4 overhead on typical PCIe hardware
./nexora --preset pcie4_bad --tp 4

# Use a model preset (Llama 70B on H100s)
./nexora --model llama-70b --preset nvlink4 --tp 8

# Launch interactive web dashboard
./nexora --serve 8080
# Then open http://127.0.0.1:8080

# Compare hardware side-by-side
./nexora --compare "pcie4_bad,nvlink,nvlink4" --tp 4

# ASCII visualization in terminal
./nexora --preset pcie4_bad --tp 4 --ascii
```

## Interactive Web Dashboard

Launch an interactive visualization dashboard:

```bash
./nexora --serve 8080
```

Then open http://127.0.0.1:8080 in your browser. Features:
- Real-time parameter adjustment with sliders
- Doughnut chart showing compute/comm/memory breakdown
- Bar chart comparing TP scaling
- Performance metrics cards
- Bottleneck detection with insights

## Model Presets

Pre-configured settings for popular LLM architectures:

| Model | Layers | Params | Use Case |
|-------|--------|--------|----------|
| `llama-7b` | 32 | 7B | Small model baseline |
| `llama-13b` | 40 | 13B | Medium model |
| `llama-70b` | 80 | 70B | Large model, needs multi-GPU |
| `mixtral-8x7b` | 32 | 47B | MoE architecture |
| `mixtral-8x22b` | 56 | 141B | Large MoE |

```bash
./nexora --model llama-70b --preset nvlink4 --tp 8
./nexora --model mixtral-8x7b --preset nvlink4 --tp 8 --moe-experts 8
```

## Hardware Presets

### PCIe (Consumer/Workstation)

| Preset | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `pcie4_good` | 18 GB/s | 25 µs | PCIe 4.0 with direct GPU-to-GPU P2P |
| `pcie4_bad` | 12 GB/s | 45 µs | PCIe 4.0 through CPU/switch (common!) |
| `pcie5_good` | 28 GB/s | 18 µs | PCIe 5.0 with good topology |
| `pcie5_bad` | 18 GB/s | 35 µs | PCIe 5.0 with suboptimal routing |

### NVIDIA NVLink (Datacenter)

| Preset | Bandwidth | Latency | Memory BW | Use Case |
|--------|-----------|---------|-----------|----------|
| `nvlink` | 150 GB/s | 3 µs | 2039 GB/s | NVLink 3.0 (A100) |
| `nvlink4` | 225 GB/s | 2 µs | 3350 GB/s | NVLink 4.0 (H100) |
| `nvlink5` | 450 GB/s | 1.5 µs | 8000 GB/s | NVLink 5.0 (B200) |

### AMD / InfiniBand

| Preset | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `infinity_fabric` | 200 GB/s | 3 µs | AMD Infinity Fabric (MI300) |
| `ib_hdr` | 25 GB/s | 1.5 µs | InfiniBand HDR (multi-node) |
| `ib_ndr` | 50 GB/s | 1 µs | InfiniBand NDR (multi-node) |

## Usage Examples

### Batch Size Modeling

See how larger batches amortize communication latency:

```bash
# Single token (worst case for latency)
./nexora --preset pcie4_bad --tp 4 --batch-size 1

# Batch of 32 (latency amortized)
./nexora --preset pcie4_bad --tp 4 --batch-size 32
```

### Pipeline Parallelism

Combine tensor parallelism with pipeline parallelism:

```bash
# TP=4 within each stage, PP=4 stages, 8 micro-batches
./nexora --preset nvlink4 --tp 4 --pp 4 --micro-batches 8
```

### MoE Expert Parallelism

Model Mixtral-style mixture of experts communication:

```bash
# 8 experts with top-2 routing
./nexora --model mixtral-8x7b --preset nvlink4 --tp 8 --moe-experts 8 --active-experts 2
```

### Hardware Comparison

Compare multiple hardware configurations side-by-side:

```bash
./nexora --compare "pcie4_bad,nvlink,nvlink4,nvlink5" --tp 8
```

Output:
```
=== Hardware Comparison (TP=8) ===

Preset                          Bandwidth      Latency     Time/tok     Overhead    Speedup
------------------------------------------------------------------------------------------
PCIe4 Bad Topology                12.0 GB/s  45.00 us     23.26 ms      +796.3%      1.00x
NVLink 3.0 (A100)                150.0 GB/s   3.00 us      3.95 ms       +52.1%      5.89x
NVLink 4.0 (H100)                225.0 GB/s   2.00 us      3.48 ms       +34.3%      6.67x
NVLink 5.0 (B200)                450.0 GB/s   1.50 us      3.25 ms       +25.1%      7.16x
```

### ASCII Visualization

Terminal-based bar charts:

```bash
./nexora --preset pcie4_bad --tp 4 --ascii
```

Output:
```
Time Breakdown (TP=4):
  Compute  [=========                               ]  21.9%  2.56 ms
  Comm     [==============================          ]  76.2%  8.90 ms
  KV       [=                                       ]   1.9%  225.34 us

TP Comparison vs TP=1 baseline:
  TP=1   [=========                               ] 2.59 ms  (baseline)
  TP=2   [====================                    ] 5.83 ms  +125%
  TP=4   [========================================] 11.69 ms  +350%

        ^ INTERCONNECT-BOUND: consider NVLink or reducing TP
```

### JSON Output

Structured output for automation:

```bash
./nexora --preset nvlink4 --tp 4 --json
```

### TTFT (Time To First Token) Simulation

```bash
./nexora --preset pcie4_bad --ttft --prompt-tokens 1024 --output-tokens 256
./nexora --preset pcie4_bad --ttft --cache-aware
```

### Parameter Sweep

```bash
./nexora --sweep --sweep-bw "12,16,20,24" --sweep-lat "20000,30000,45000" > sweep.csv
```

## CLI Reference

Run `./nexora --help` for full documentation. Key flags:

**Hardware / Topology**:
- `--preset` - Hardware profile (pcie4_bad, nvlink4, etc.)
- `--tp` - Tensor parallelism degree
- `--pp` - Pipeline parallelism degree
- `--micro-batches` - Micro-batches for PP

**Model Structure**:
- `--model` - Model preset (llama-70b, mixtral-8x7b, etc.)
- `--layers` - Number of transformer layers
- `--batch-size` - Tokens processed in parallel
- `--moe-experts` - Number of MoE experts (0 = dense)
- `--active-experts` - Top-k routing for MoE

**Runtime / Scheduling**:
- `--group` - Layer grouping (sync every N layers)
- `--bits` - Communication precision (16/8/4)
- `--overlap` - Comm/compute overlap fraction

**Output Modes**:
- `--json` - JSON output for automation
- `--ascii` - ASCII visualization
- `--serve <port>` - Interactive web dashboard
- `--compare "preset1,preset2"` - Side-by-side comparison

## Bottleneck Detection

The simulator automatically identifies the dominant bottleneck:

- **COMPUTE-BOUND** - Computation dominates, TP scaling is efficient
- **INTERCONNECT-BOUND** - GPU-to-GPU communication dominates, consider NVLink
- **MEMORY-BOUND** - HBM bandwidth limits performance
- **BALANCED** - No single factor dominates

## Simulation Model

### Token Time Calculation

```
token_time = compute_time + visible_comm_time + kv_overhead

compute_time = layers × compute_ns × batch_size
visible_comm_time = raw_comm × (1 - overlap)
raw_comm = num_collectives × (latency × algo_steps + transfer_time)
```

### Batch Size Amortization

Larger batches amortize the fixed latency cost:
- Compute scales linearly with batch size
- Communication payload scales linearly
- Latency is paid once per collective (not per token)
- Result: per-token time decreases with larger batches

### Pipeline Parallelism

Pipeline bubble fraction: `(pp - 1) / (micro_batches + pp - 1)`

More micro-batches reduce bubble overhead but increase memory usage.

### MoE Communication

MoE models use all-to-all communication for expert dispatch/combine:
- Two all-to-all ops per MoE layer
- Data fraction: `(tp-1)/tp` sent to other GPUs
- Lower latency than all-reduce but higher bandwidth usage

## Testing

```bash
cargo test              # Run all tests
cargo test invariants   # Run specific module
```

Tests validate monotonicity invariants, TP=1 independence, and reality checks.

## License

MIT
